# demo inspired by https://github.com/TencentARC/T2I-Adapter/tree/main
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import argparse
import copy
import cv2
import gradio as gr
import numpy as np
import torch
import pickle
import joblib
import os
# 
from PIL import Image
from functools import partial
from itertools import chain
from torch import autocast
from pytorch_lightning import seed_everything

from basicsr.utils import tensor2img
from ldm.inference_base import DEFAULT_NEGATIVE_PROMPT, diffusion_inference_from_fMRI, get_adapters, get_sd_models
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_cond_model
from ldm.modules.encoders.adapter import CoAdapterFuser

from train_soft_intro_vae import SoftIntroVAE, load_model, reparameterize

torch.set_grad_enabled(False)

supported_cond = ['sketch', 'color', 'depth']

# 配置全局参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '--sd_ckpt',
    type=str,
    #TODO 
    default='../Dataset/SD_models/sd-v1-4.ckpt',
    # default='models/v1-5-pruned-emaonly.ckpt',
    # default='models/v1-5-pruned.ckpt',
    help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
)
parser.add_argument(
    '--vae_ckpt',
    type=str,
    default=None,
    help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
)
global_opt = parser.parse_args()
#TODO 
global_opt.config = '../Dataset/SD_models/v1-inference.yaml'
# global_opt.config = 'configs/stable-diffusion/sd-v1-inference.yaml'
for cond_name in supported_cond:
    setattr(global_opt, f'{cond_name}_adapter_ckpt', f'models/coadapter-{cond_name}-sd15v1.pth')
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
global_opt.max_resolution = 512 * 512
global_opt.sampler = 'ddim'
global_opt.cond_weight = 1.0 #TODO
global_opt.C = 4
global_opt.f = 8
#TODO: expose style_cond_tau to users
global_opt.style_cond_tau = 1.0

# 配置SD模型
sd_model, sampler = get_sd_models(global_opt)
adapters = {}
cond_models = {}
torch.cuda.empty_cache()

# 配置Fuser
coadapter_fuser = CoAdapterFuser(unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3)
coadapter_fuser.load_state_dict(torch.load(f'models/coadapter-fuser-sd15v1.pth'))
coadapter_fuser = coadapter_fuser.to(global_opt.device)

# 配置测试数据和原始刺激
sub = "01"
ROIs_cond_img = ['V1', 'V2', 'V3', 'V3ab', 'VO', 'hV4', 'IPS', 'LO', 'MST', 'MT', 'PHC']
ROIs_text = ['VO', 'V1', 'V2', 'V3', 'V3ab', 'PHC', 'MT', 'MST', 'LO', 'IPS', 'hV4']
fMRIs_path = f"E:\\Dataset\\NSD_preprocessed\\sub{sub}\\val_voxel_multi_trial_data_"
ori_path = f"E:\\Dataset\\NSD_preprocessed\\sub{sub}" #TODO

# 配置FastL2LiR模型
mode = "muANDlogvar" # 选择重建流程的中间lf的类型：muANDlogvar或z
region = 11

sketch_mu_alpha = 1.0
sketch_mu_n_fea = 2000
sketch_logvar_alpha = 1.0
sketch_logvar_n_fea = 2000

depth_mu_alpha = 1.0
depth_mu_n_fea = 2000
depth_logvar_alpha = 1.0
depth_logvar_n_fea = 2000

color_mu_alpha = 1.0
color_mu_n_fea = 2000
color_logvar_alpha = 1.0
color_logvar_n_fea = 2000

sketch_mu_pth = f"sub{sub}_sketch_epoch10_reg{region}/fMRI_to_sketch_sub{sub}_mu_{sketch_mu_n_fea}_{sketch_mu_alpha}.sav"
sketch_logvar_pth = f"sub{sub}_sketch_epoch10_reg{region}/fMRI_to_sketch_sub{sub}_logvar_{sketch_logvar_n_fea}_{sketch_logvar_alpha}.sav"
sketch_mu_pth = f"../Dataset/PyFastL2LiR_models_(ImageNet_NSD)/{sketch_mu_pth}"
sketch_logvar_pth = f"../Dataset/PyFastL2LiR_models_(ImageNet_NSD)/{sketch_logvar_pth}"
pyfast_sketch_mu = joblib.load(sketch_mu_pth)
pyfast_sketch_logvar = joblib.load(sketch_logvar_pth)

depth_mu_pth = f"sub{sub}_depth_epoch23_reg{region}/fMRI_to_depth_sub{sub}_mu_{depth_mu_n_fea}_{depth_mu_alpha}.sav"
depth_logvar_pth = f"sub{sub}_depth_epoch23_reg{region}/fMRI_to_depth_sub{sub}_logvar_{depth_logvar_n_fea}_{depth_logvar_alpha}.sav"
depth_mu_pth = f"../Dataset/PyFastL2LiR_models_(ImageNet_NSD)/{depth_mu_pth}"
depth_logvar_pth = f"../Dataset/PyFastL2LiR_models_(ImageNet_NSD)/{depth_logvar_pth}"
pyfast_depth_mu = joblib.load(depth_mu_pth)
pyfast_depth_logvar = joblib.load(depth_logvar_pth)

color_mu_pth = f"sub{sub}_color_epoch17_reg{region}/fMRI_to_color_sub{sub}_mu_{color_mu_n_fea}_{color_mu_alpha}.sav"
color_logvar_pth = f"sub{sub}_color_epoch17_reg{region}/fMRI_to_color_sub{sub}_logvar_{color_logvar_n_fea}_{color_logvar_alpha}.sav"
color_mu_pth = f"../Dataset/PyFastL2LiR_models_(ImageNet_NSD)/{color_mu_pth}"
color_logvar_pth = f"../Dataset/PyFastL2LiR_models_(ImageNet_NSD)/{color_logvar_pth}"
pyfast_color_mu = joblib.load(color_mu_pth)
pyfast_color_logvar = joblib.load(color_logvar_pth)

# 配置Soft-IntroVAE
sketch_vae_pth = "sketch_soft_intro_betas_0.5_1024.0_1.0_model_epoch_10_iter_74260.pth"
depth_vae_pth = "depth_soft_intro_betas_0.5_1024.0_1.0_model_epoch_23_iter_170798.pth"
color_vae_pth = "color_soft_intro_betas_0.5_1024.0_1.0_model_epoch_17_iter_126242.pth"
sketch_vae_pth = f"../Dataset/SoftIntroVAE_(ImageNet_NSD)_checkpoints/{sketch_vae_pth}"
depth_vae_pth = f"../Dataset/SoftIntroVAE_(ImageNet_NSD)_checkpoints/{depth_vae_pth}"
color_vae_pth = f"../Dataset/SoftIntroVAE_(ImageNet_NSD)_checkpoints/{color_vae_pth}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae_sketch = SoftIntroVAE(cdim=1, zdim=256, channels=[64, 128, 256, 512, 512, 512], image_size=256).to(device)
vae_depth = SoftIntroVAE(cdim=1, zdim=256, channels=[64, 128, 256, 512, 512, 512], image_size=256).to(device)
vae_color = SoftIntroVAE(cdim=3, zdim=256, channels=[64, 128, 256, 512, 512, 512], image_size=256).to(device)
load_model(vae_sketch, sketch_vae_pth, device)
load_model(vae_depth, depth_vae_pth, device)
load_model(vae_color, color_vae_pth, device)


def run(*args):
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):

        inps = []
        for i in range(0, len(args) - 10, len(supported_cond)):
            inps.append(args[i:i + len(supported_cond)])

        opt = copy.deepcopy(global_opt)
        opt.prompt, opt.neg_prompt, opt.scale, opt.n_samples, opt.seed, opt.steps, opt.resize_short_edge, opt.cond_tau, opt.n_sample, opt.btn2 \
            = args[-10:]

        # TODO 获取opt.n_sample对应的原始刺激图像
        originals = []
        ori_extract_path = os.path.join(ori_path, 'val_stim_multi_trial_data', f'{opt.n_sample:07}_RGB.png')
        if not os.path.exists(ori_extract_path):
            print(f"文件未找到: {ori_extract_path}")
        original = cv2.imread(ori_extract_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        originals.append(original)

        # TODO 解码fMRI获取sketch、depth、color、cap四个模态的信息
        n_sample = opt.n_sample # TODO 应该暴露给用户，目前不能批量重建
        fMRIs_cond_img = fetch_ROI_voxel(fMRIs_path, ROIs_cond_img)
        fMRIs_text = fetch_ROI_voxel(fMRIs_path, ROIs_text)
        skecths_pth, depths_pth, colors_pth, caps = fMRI2condition(fMRIs_cond_img[n_sample:n_sample+1, :], fMRIs_text, index=n_sample)

        # 获取condition images (tensor)以及对应的condition names (str)
        conds = []
        activated_conds = []
        prev_size = None
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            cond_name = supported_cond[idx]
            if b == 'Nothing':
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].cpu()
            else:
                activated_conds.append(cond_name)
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].to(opt.device)
                else:
                    adapters[cond_name] = get_adapters(opt, getattr(ExtraCondition, cond_name))
                adapters[cond_name]['cond_weight'] = cond_weight

                process_cond_module = getattr(api, f'get_cond_{cond_name}') # sketch\depth\color extracting

                if b == 'Image':
                    if cond_name not in cond_models:
                        cond_models[cond_name] = get_cond_model(opt, getattr(ExtraCondition, cond_name)) # sketch\depth\color extracting models
                    if prev_size is not None:
                        image = cv2.resize(im1, prev_size, interpolation=cv2.INTER_LANCZOS4)
                    else:
                        image = im1 #TODO
                    conds.append(process_cond_module(opt, image, 'image', cond_models[cond_name]))
                    if idx != 0 and prev_size is None:  # skip style since we only care spatial cond size
                        h, w = image.shape[:2]
                        prev_size = (w, h)
                elif b == 'fMRI': # TODO
                    if cond_name == 'sketch':
                        image = skecths_pth[0]
                    elif cond_name == 'depth':
                        image = depths_pth[0]
                    elif cond_name == 'color':
                        image = colors_pth[0]
                    else:
                        print("Error: undified cond_name")
                    conds.append(process_cond_module(opt, image, cond_name, None))     
                    # if idx != 0 and prev_size is None:  # skip style since we only care spatial cond size
                    #     h, w = image.shape[:2]
                    #     prev_size = (w, h)                                 
                else:
                    if prev_size is not None:
                        image = cv2.resize(im2, prev_size, interpolation=cv2.INTER_LANCZOS4)
                    else:
                        image = im2
                    conds.append(process_cond_module(opt, image, cond_name, None))
                    if idx != 0 and prev_size is None:  # skip style since we only care spatial cond size
                        h, w = image.shape[:2]
                        prev_size = (w, h)

        # 获取对应于condition images的condition features
        features = dict()
        for idx, cond_name in enumerate(activated_conds):
            cur_feats = adapters[cond_name]['model'](conds[idx])
            if isinstance(cur_feats, list):
                for i in range(len(cur_feats)):
                    cur_feats[i] *= adapters[cond_name]['cond_weight']
            else:
                cur_feats *= adapters[cond_name]['cond_weight']
            features[cond_name] = cur_feats

        # 获取fuser feature
        adapter_features, append_to_context = coadapter_fuser(features)

        # 将cond (tensor)转换为cond (bgr)
        output_conds = []
        for cond in conds:
            output_conds.append(tensor2img(cond, rgb2bgr=False))

        # 开始SD推理
        ims = []
        seed_everything(opt.seed)
        for _ in range(opt.n_samples):
            result = diffusion_inference_from_fMRI(opt, sd_model, sampler, adapter_features, caps[0], append_to_context, fMRI = (opt.btn2 == "fMRI"))
            ims.append(tensor2img(result, rgb2bgr=False))

        # Clear GPU memory cache so less likely to OOM
        torch.cuda.empty_cache()
        return output_conds, ims, originals

# TODO 把两个fMRI合二为一
def fMRI2condition(fMRIs_cond_img, fMRIs_text, index, output_dir = 'recon_demo_inter_results',
                text_model_path='model_cond/', text_mean_path_without_cut='z-score/text_lf_mean_without_cut.npy', 
                text_std_path_without_cut='z-score/text_lf_std_without_cut.npy', cls_token_path='z-score/cls_token.npy', device='cuda'):
    
    skecths_pth = []
    depths_pth = []
    colors_pth = []
    caps = []

    text_alpha = 0.15
    text_n = 450
    
    for _, fMRI_cond_img in enumerate(fMRIs_cond_img):        
        ### fMRI2sketch ###
        sketch_decoded = fMRI2sketch(fMRI_cond_img.reshape(1, -1))
        sketch_decoded_np = sketch_decoded.detach().cpu().numpy().squeeze()
        if len(sketch_decoded_np.shape) == 3 and sketch_decoded_np.shape[0] == 3:
            sketch_decoded_np = np.transpose(sketch_decoded_np, (1, 2, 0))
        sketch_decoded_image = Image.fromarray((sketch_decoded_np * 255).astype(np.uint8))
        
        sketch_pth = os.path.join(output_dir, f'sub{sub}_sketch', f'{index:05}_sketch.png')
        sketch_decoded_image.save(sketch_pth)
        skecths_pth.append(sketch_pth)
        
        ### fMRI2depth ###
        depth_decoded = fMRI2depth(fMRI_cond_img.reshape(1, -1))
        depth_decoded_np = depth_decoded.detach().cpu().numpy().squeeze()
        if len(depth_decoded_np.shape) == 3 and depth_decoded_np.shape[0] == 3:
            depth_decoded_np = np.transpose(depth_decoded_np, (1, 2, 0))
        depth_decoded_image = Image.fromarray((depth_decoded_np * 255).astype(np.uint8))
        
        depth_pth = os.path.join(output_dir, f'sub{sub}_depth', f'{index:05}_depth.png')
        depth_decoded_image.save(depth_pth)
        depths_pth.append(depth_pth)
        
        ### fMRI2color ###
        color_decoded = fMRI2color(fMRI_cond_img.reshape(1, -1))
        color_decoded_np = color_decoded.detach().cpu().numpy().squeeze()
        if len(color_decoded_np.shape) == 3 and color_decoded_np.shape[0] == 3:
            color_decoded_np = np.transpose(color_decoded_np, (1, 2, 0))
        color_decoded_image = Image.fromarray((color_decoded_np * 255).astype(np.uint8))
        
        color_pth = os.path.join(output_dir, f'sub{sub}_color', f'{index:05}_color.png')
        color_decoded_image.save(color_pth)
        colors_pth.append(color_pth)
        
        ### fMRI2text ###
        x_test = scaler.fit_transform(fMRIs_text)
        mean = (np.load(text_mean_path_without_cut)).reshape(1, -1)
        std = (np.load(text_std_path_without_cut))
        std[:768] = 0
        std = std.reshape(1, -1)
        
        text_model_name = "fastl2_semantic_{}_{}.pickle".format(text_n, text_alpha)
        f_save = open(text_model_path + text_model_name, 'rb')
        model = pickle.load(f_save)
        f_save.close()
        
        z = np.zeros((1, 11520))
        cls = (np.load(cls_token_path))
        text_pred = model.predict(x_test)[index:index + 1, :] #TODO
        z[:,:768] = cls
        z[:,768:] = text_pred
        cap_numpy = reverse_reshape(z, mean, std)
        cap = torch.tensor(cap_numpy.astype(np.float32)).to(device)
        caps.append(cap)
        
    return skecths_pth, depths_pth, colors_pth, caps

def fMRI2sketch(fMRI):
    if mode == "z":
        # 使用PyFastL2LiR解码fMRI到z
        # z = linear_regression_z.predict(fMRI)
        # z = torch.tensor(z).float().to(device)
        z = 0 #TODO
    elif mode == "muANDlogvar":
        # 使用PyFastL2LiR解码fMRI到mu和logvar
        mu = pyfast_sketch_mu.predict(fMRI)
        logvar = pyfast_sketch_logvar.predict(fMRI)

        # 将mu和logvar转换为PyTorch张量
        mu = torch.tensor(mu).float().to(device)
        logvar = torch.tensor(logvar).float().to(device)
        z = reparameterize(mu, logvar)
    else:
        print("Error: undified mode")
        
    # 重构深度图像
    vae_sketch.eval()
    with torch.no_grad():
        sketch = vae_sketch.decoder(z)
    
    return sketch

def fMRI2depth(fMRI):
    if mode == "z":
        # 使用PyFastL2LiR解码fMRI到z
        # z = linear_regression_z.predict(fMRI)
        # z = torch.tensor(z).float().to(device)
        z = 0 #TODO
    elif mode == "muANDlogvar":
        # 使用PyFastL2LiR解码fMRI到mu和logvar
        mu = pyfast_depth_mu.predict(fMRI)
        logvar = pyfast_depth_logvar.predict(fMRI)

        # 将mu和logvar转换为PyTorch张量
        mu = torch.tensor(mu).float().to(device)
        logvar = torch.tensor(logvar).float().to(device)
        z = reparameterize(mu, logvar)
    else:
        print("Error: undified mode")
        
    # 重构深度图像
    vae_depth.eval()
    with torch.no_grad():
        depth = vae_depth.decoder(z)
    
    return depth

def fMRI2color(fMRI):
    if mode == "z":
        # 使用PyFastL2LiR解码fMRI到z
        # z = linear_regression_z.predict(fMRI)
        # z = torch.tensor(z).float().to(device)
        z = 0 #TODO
    elif mode == "muANDlogvar":
        # 使用PyFastL2LiR解码fMRI到mu和logvar
        mu = pyfast_color_mu.predict(fMRI)
        logvar = pyfast_color_logvar.predict(fMRI)

        # 将mu和logvar转换为PyTorch张量
        mu = torch.tensor(mu).float().to(device)
        logvar = torch.tensor(logvar).float().to(device)
        z = reparameterize(mu, logvar)
    else:
        print("Error: undified mode")
        
    # 重构深度图像
    vae_color.eval()
    with torch.no_grad():
        color = vae_color.decoder(z)
    
    return color

def fetch_ROI_voxel(file_ex, ROIs):
    file_paths = [file_ex + roi + '.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

def reverse_reshape(l , mean, std): #(8859,15*768)>>(8859,15,768);revers_z_score
    b = []
    l = l * std + mean
    for i in range(l.shape[0]):
        a = np.concatenate([(l[i])[np.newaxis, 768 * j:768 * (j + 1)] for j in range(15)], axis=0)
        b.append(a)
    b_after_reverse = np.array(b)
    return b_after_reverse

def change_visible(im1, im2, val):
    outputs = {}
    if val == "Image":
        outputs[im1] = gr.update(visible=True)
        outputs[im2] = gr.update(visible=False)
    elif val == "Nothing" or val == "fMRI":
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=False)
    else:
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=True)
    return outputs

def update_visibility(prompt, val):
    outputs = {}
    if val == "Prompt":
        outputs[prompt] = gr.update(visible=True)
    else:
        outputs[prompt] = gr.update(visible=False)
    return outputs

DESCRIPTION = '''# VisualDiffuser: 
## <span style="color: #950660;"> a flexible image reconstruction framework for analyzing the contributions of various conditions in human brain visual information processing

### [GitHub](https://github.com/Gengsheng-Li/VisualDiffuser) [Details](https://github.com/Gengsheng-Li/VisualDiffuser/tree/main#readme)

### This gradio demo is for a simple experience of VisualDiffuser. Following are some useful tips:
- **Four types of conditions are offered**. `Sketch`, `Color`, `Depth` and `Text`.
- **Four ways are provided for extracting these conditions**. `fMRI`: Automatically decode the corresponding condition from the selected fMRI sample; `Sketch/Color/Depth/Prompt`: Use your input directly as the corresponding condition; `Image`: Extract the corresponding condition from the image you upload; `Nothing`: Indicate that the corresponding condition will not participate in image reconstruction.
- `fMRI sample num` **is designed to be used in conjunction with** `fMRI`. You can slide `fMRI sample num` to select a real fMRI sample from the dataset `NSD subject 01 val_multi`. However, if you do not wish to use any fMRI-based condition extraction, then you don't need to care about it.
- **Start with fewer conditions**. If you plan to use more than two conditions in image reconstruction, it is recommended to start with only one condition. Once you have found the suitable `Condition weight` for existing conditions, gradually append the new conditions. **Note**: Based on our experience, including Text as one of your conditions usually results in a more accurate image reconstruction.
- **Condition weight is important**.  If the reconstructed image is not well aligned with the condition, increase the corresponding `Condition weight`. If increasing `Condition weight` is ineffective or degrades image quality, try decreasing the `Condition weight` of other conditions.
- It is recommended to use a step size of 0.1 to adjust `Condition weight`. From experience, `Condition weight` will not be less than 0.5 or greater than 1.5.

'''
with gr.Blocks(title="VisualDiffuser", css=".gr-box {border-color: #8136e2}") as demo:
    gr.Markdown(DESCRIPTION)

    btns = []
    ims1 = []
    ims2 = []
    cond_weights = []

    with gr.Box():
        with gr.Column():
            with gr.Row():
                for cond_name in supported_cond:
                    with gr.Box():
                        with gr.Column():
                            btn1 = gr.Radio(
                                choices=["fMRI", "Image", cond_name, "Nothing"],
                                label=f"Input type for {cond_name}",
                                interactive=True,
                                value="Nothing",
                            )
                            im1 = gr.Image(source='upload', label="Image", interactive=True, visible=False, type="numpy")
                            im2 = gr.Image(source='upload', label=cond_name, interactive=True, visible=False, type="numpy")
                            cond_weight = gr.Slider(
                                label="Condition weight", minimum=0, maximum=5, step=0.05, value=1, interactive=True)

                            fn = partial(change_visible, im1, im2)
                            btn1.change(fn=fn, inputs=[btn1], outputs=[im1, im2], queue=False)

                            btns.append(btn1)
                            ims1.append(im1)
                            ims2.append(im2)
                            cond_weights.append(cond_weight)
            
            with gr.Column():
                btn2 = gr.Radio(
                    choices=["fMRI", "Prompt"],
                    label=f"Input type for text",
                    interactive=True,
                    value="fMRI",
                )
                prompt = gr.Textbox(label="Prompt", visible=False)
                
                btn2.change(fn=partial(update_visibility, prompt), inputs=[btn2], outputs=[prompt], queue=False)
    
    with gr.Box():
        with gr.Column():
            n_sample = gr.Slider(label="fMRI sample num (NSD subject 01)", value=0, minimum=0, maximum=981, step=1, interactive=True)
                
            with gr.Accordion('Advanced options', open=False):
                neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
                scale = gr.Slider(label="Guidance Scale (Classifier free guidance)", value=7.5, minimum=1, maximum=20, step=0.1)
                n_samples = gr.Slider(label="Num samples", value=1, minimum=1, maximum=8, step=1)
                seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1)
                steps = gr.Slider(label="Steps", value=50, minimum=10, maximum=100, step=1)
                resize_short_edge = gr.Slider(label="Image resolution", value=512, minimum=320, maximum=1024, step=1)
                cond_tau = gr.Slider(
                    label="timestamp parameter that determines until which step the adapter is applied",
                    value=1.0,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05)

            submit = gr.Button("Generate")
    
    with gr.Box():
        with gr.Column():
            cond = gr.Gallery(label="Condition images").style(grid=3, height='auto')
            with gr.Row():
                output = gr.Gallery(label="Reconstructed images").style(grid=3, height='auto')
                original = gr.Gallery(label="Original stimulations").style(grid=3, height='auto')
    
    inps = list(chain(btns, ims1, ims2, cond_weights))
    inps.extend([prompt, neg_prompt, scale, n_samples, seed, steps, resize_short_edge, cond_tau, n_sample, btn2])
    submit.click(fn=run, inputs=inps, outputs=[cond, output, original])
demo.launch(share=True, auth=("demo", "casia-bjut"))