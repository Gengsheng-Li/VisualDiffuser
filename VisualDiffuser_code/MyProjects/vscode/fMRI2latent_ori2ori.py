# 共计一块

import numpy as np
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import joblib
import fastl2lir


# 
# 选择合适的参数 !!!
type = 'sketch'
lf_type = 'logvar'
trial = 'multi_trial' # 选择用于测试的测试数据：multi_trial/single_trial
n_region = 11
sub = 'sub01'
alpha = 1.0
n_feat = 450
model_epoch = "epoch54"
Y_folder = f"{sub}_{type}_{model_epoch}"

# Set data storage directory
X_dir = f"../../Dataset/NSD_preprocessed/{sub}/"
Y_dir = f"../../Dataset/SoftIntroVAE_(ImageNet_NSD)_latent_features/{Y_folder}/"

# ----------------------- Training ----------------------
# Data for training
hV4_data = np.load(X_dir + "trn_voxel_data_hV4.npy")
IPS_data = np.load(X_dir + "trn_voxel_data_IPS.npy")
LO_data = np.load(X_dir + "trn_voxel_data_LO.npy")
MST_data = np.load(X_dir + "trn_voxel_data_MST.npy")
MT_data = np.load(X_dir + "trn_voxel_data_MT.npy")
PHC_data = np.load(X_dir + "trn_voxel_data_PHC.npy")
V1_data = np.load(X_dir + "trn_voxel_data_V1.npy")
V2_data = np.load(X_dir + "trn_voxel_data_V2.npy")
V3_data = np.load(X_dir + "trn_voxel_data_V3.npy")
V3ab_data = np.load(X_dir + "trn_voxel_data_V3ab.npy")
VO_data = np.load(X_dir + "trn_voxel_data_VO.npy")

X_train = VO_data
if n_region == 11:
    X_train_combined = np.hstack([V1_data, V2_data, V3_data, V3ab_data, VO_data, hV4_data, IPS_data, LO_data, MST_data, MT_data, PHC_data])
elif n_region == 6:
    X_train_combined = np.hstack([V1_data, V2_data, V3_data, V3ab_data, hV4_data, MT_data]) 
else:
    print("Error: Undefined brain regions")

Y_train = np.load(Y_dir + f"trn_{type}_{lf_type}_{sub}.npy")


# Create and train the model
print(f'Training: X_trn shape: {X_train_combined.shape}, Y_trn shape: {Y_train.shape}')
model = fastl2lir.FastL2LiR()
model.fit(X_train_combined, Y_train, alpha, n_feat)


# Predict
Y_train_predicted = model.predict(X_train_combined)

# ------------------- Valuating -------------------
# Data for valuating
val_hV4_data = np.load(X_dir + "val_voxel_multi_trial_data_hV4.npy")
val_IPS_data = np.load(X_dir + "val_voxel_multi_trial_data_IPS.npy")
val_LO_data = np.load(X_dir + "val_voxel_multi_trial_data_LO.npy")
val_MST_data = np.load(X_dir + "val_voxel_multi_trial_data_MST.npy")
val_MT_data = np.load(X_dir + "val_voxel_multi_trial_data_MT.npy")
val_PHC_data = np.load(X_dir + "val_voxel_multi_trial_data_PHC.npy")
val_V1_data = np.load(X_dir + "val_voxel_multi_trial_data_V1.npy")
val_V2_data = np.load(X_dir + "val_voxel_multi_trial_data_V2.npy")
val_V3_data = np.load(X_dir + "val_voxel_multi_trial_data_V3.npy")
val_V3ab_data = np.load(X_dir + "val_voxel_multi_trial_data_V3ab.npy")
val_VO_data = np.load(X_dir + "val_voxel_multi_trial_data_VO.npy")

X_val = val_VO_data
if n_region == 11:
    X_val_combined = np.hstack([val_V1_data, val_V2_data, val_V3_data, val_V3ab_data, val_VO_data, val_hV4_data, val_IPS_data, val_LO_data, val_MST_data, val_MT_data, val_PHC_data])
elif n_region == 6:
    X_val_combined = np.hstack([val_V1_data, val_V2_data, val_V3_data, val_V3ab_data, val_hV4_data, val_MT_data])
else:
    print("Error: Undefined brain regions")
Y_val = np.load(Y_dir + f"val_{trial}_{type}_{lf_type}_{sub}.npy")


# Predict
print(f'X_val shape: {X_val_combined.shape}   Y_val shape: {Y_val.shape}')
Y_val_predicted = model.predict(X_val_combined)

# ------------------- Valuation --------------------
# Check if there are any NaN or infinite values in Y or Y_predicted
temp = np.isnan(Y_train).any() or np.isinf(Y_train).any() or np.isnan(Y_train_predicted).any() or np.isinf(Y_train_predicted).any() or np.isnan(Y_val).any() or np.isinf(Y_val).any() or np.isnan(Y_val_predicted).any() or np.isinf(Y_val_predicted).any()
if temp != False:
    print("NaN in Y_train: ", np.isnan(Y_train).any())
    print("Infinite in Y_train: ", np.isinf(Y_train).any())
    print("NaN in Y_train_predicted: ", np.isnan(Y_train_predicted).any())
    print("Infinite in Y_train_predicted: ", np.isinf(Y_train_predicted).any())
    print("NaN in Y_val: ", np.isnan(Y_val).any())
    print("Infinite in Y_val: ", np.isinf(Y_val).any())
    print("NaN in Y_val_predicted: ", np.isnan(Y_val_predicted).any())
    print("Infinite in Y_val_predicted: ", np.isinf(Y_val_predicted).any())


# Check predicted Y shape
print('Y_pred_trn shape: %s   Y_pred_val shape: %s' % (Y_train_predicted.shape, Y_val_predicted.shape,))


# Calculate MSE and PCC
mse_trn = mean_squared_error(Y_train, Y_train_predicted)
mse_val = mean_squared_error(Y_val, Y_val_predicted)
corr_trn, _ = pearsonr(Y_train.flatten(), Y_train_predicted.flatten())
corr_val, _ = pearsonr(Y_val.flatten(), Y_val_predicted.flatten())
print(f"sub: {sub}   type: {type}   lf_type: {lf_type}")
print(f"n_feat: {n_feat} alpha: {alpha}")
print(f"Trn MSE/Val MSE: {mse_trn:.4f}/{mse_val:.4f}   Trn PCC/Val PCC: {corr_trn:.4f}/{corr_val:.4f}")

# ------------------- Saving Model --------------------
model_saving_folder = f"{sub}_{type}_{model_epoch}_reg{n_region}"
model_saving_pth = f'D:\\GitHub\\Dataset\\PyFastL2LiR_models_(ImageNet_NSD)\\{model_saving_folder}'
if not os.path.exists(model_saving_pth):
    os.mkdir(model_saving_pth)
filename = model_saving_pth + f'\\fMRI_to_{type}_{sub}_{lf_type}_{n_feat}_{alpha}__{mse_trn:.4f}_{mse_val:.4f}__{corr_trn:.4f}_{corr_val:.4f}.sav'
joblib.dump(model, filename)