# 函数外一块

import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from train_soft_intro_vae_color import SoftIntroVAE, ColorDataset, DepthDataset, SketchDataset, load_model, reparameterize
from torchvision import transforms

# 
def encode_and_save(dataset_path, model_path, output_path, batch_size=1, device=torch.device("cpu"), 
                    type="val", trial="_multi_trail", cond="color", sub="sub01"):
    # Define dataset and load model
    if cond == "color":
        train_set = ColorDataset(file_path=dataset_path, transform=transforms.ToTensor())
        model = SoftIntroVAE(cdim=3, zdim=256, channels=[64, 128, 256, 512, 512, 512], image_size=256).to(device)
    elif cond == "depth":
        train_set = DepthDataset(file_path=dataset_path, transform=transforms.ToTensor())
        model = SoftIntroVAE(cdim=1, zdim=256, channels=[64, 128, 256, 512, 512, 512], image_size=256).to(device)
    elif cond == "sketch":
        train_set = SketchDataset(file_path=dataset_path, transform=transforms.ToTensor())
        model = SoftIntroVAE(cdim=1, zdim=256, channels=[64, 128, 256, 512, 512, 512], image_size=256).to(device)
    else:
        print("Error: undified condition")
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    load_model(model, model_path, device)
    
    # Lists to store mu and logvar
    all_mu = []
    all_logvar = []
    all_z = []

    # Iterate through dataset
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)
            batch = batch.to(device)

            # Encode
            mu, logvar = model.encode(batch)
            z = reparameterize(mu, logvar)
            
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_z.append(z.cpu().numpy())

    # Concatenate results
    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)
    all_z = np.concatenate(all_z, axis=0)

    # Save to .npy file
    np.save(output_path + f'\\{type}{trial}_{cond}_mu_{sub}.npy', all_mu)
    np.save(output_path + f'\\{type}{trial}_{cond}_logvar_{sub}.npy', all_logvar)
    np.save(output_path + f'\\{type}{trial}_{cond}_z_{sub}.npy', all_z)

    print("Encoding and saving completed.")

# Parameters to be modified !!!
type = "val"
trial = "_single_trial" # trn: non; val: _multi_trial or _single_trial
cond = "sketch"
sub = "sub01"
model = "sketch_soft_intro_betas_0.5_1024.0_1.0_model_epoch_10_iter_74260.pth"
epoch = "epoch_test"

dataset_path = f"../../Dataset/T2I_preprocessed_(NSD)_256/{type}_stim{trial}_data_{cond}_{sub}_256.npy"
model_path = f"../../Dataset/SoftIntroVAE_(ImageNet_NSD)_checkpoints/{model}"
output_path = f"../../Dataset/SoftIntroVAE_(ImageNet_NSD)_latent_features/{sub}_{cond}_{epoch}" # replace with your desired output path
if not os.path.exists(output_path):
    os.mkdir(output_path)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encode_and_save(dataset_path, model_path, output_path, device = device, type = type, trial = trial, cond = cond, sub = sub)