import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from ResMLPModel import ResMLP
from Scheduler import GradualWarmupScheduler
from Dataset import VectorDataset


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # MODIFIED: Use the new SineWaveDataset
    dataset = VectorDataset(data_dir=modelConfig["data_dir"])
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    net_model = ResMLP(T=modelConfig["T"], data_dim=modelConfig["data_len"], num_blocks=4).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    best_loss = float("inf")
    for e in range(modelConfig["epoch"]):
        e_loss = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data, labels in tqdmDataLoader:
                optimizer.zero_grad()
                x_0 = data.to(device)
                # MODIFIED: Using mean() is more standard than sum() / 1000
                loss = trainer(x_0).mean()
                loss.backward()
                e_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "data shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if e_loss < best_loss:
            best_loss = e_loss
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_1d_' + "_.pt"))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # Load Model
    model = ResMLP(T=modelConfig["T"], data_dim=modelConfig["data_len"], num_blocks=4).to(device)
    ckpt = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]),
                      map_location=device)
    model.load_state_dict(ckpt)
    print("Model loaded successfully.")
    model.eval()

    sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # 1. Get original data from the directory
    print("Loading original data for comparison...")
    original_dataset = VectorDataset(data_dir=modelConfig["data_dir"])
    # Use a DataLoader to efficiently load all data points
    original_dataloader = DataLoader(original_dataset, batch_size=modelConfig["batch_size"], shuffle=False,
                                     num_workers=4)

    original_data_list = []
    for batch, _ in tqdm(original_dataloader, desc="Loading original vectors"):
        original_data_list.append(batch.cpu().numpy())

    # Concatenate all batches and remove the channel dimension
    original_data = np.concatenate(original_data_list, axis=0).squeeze()
    num_samples = len(original_data)  # Get the exact number of samples from the dataset
    print(f"Loaded {num_samples} original data samples.")

    # 2. Generate the same number of sampled data
    sampled_data_list = []
    batch_size = modelConfig["batch_size"]
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
            # Handle the last batch which might be smaller
            current_batch_size = min(batch_size, num_samples - i)
            noisy_data = torch.randn(size=[current_batch_size, 1, modelConfig["data_len"]], device=device)
            sampled_batch = sampler(noisy_data)
            sampled_data_list.append(sampled_batch.cpu().numpy().squeeze())

    sampled_data = np.concatenate(sampled_data_list, axis=0)
    print(f"Generated {len(sampled_data)} new data samples.")

    # --- PCA and Plotting (remains the same) ---

    # 3. Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    combined_data = np.vstack((original_data, sampled_data))
    transformed_data = pca.fit_transform(combined_data)
    original_pca = transformed_data[:num_samples]
    sampled_pca = transformed_data[num_samples:]

    # 4. Create the scatter plot
    print("Creating plot...")
    plt.figure(figsize=(12, 12))
    plt.scatter(original_pca[:, 0], original_pca[:, 1], alpha=0.6,
                s=15, label='Original Data (from .pt files)', color='dodgerblue')
    plt.scatter(sampled_pca[:, 0], sampled_pca[:, 1], alpha=0.6,
                s=15, label='Sampled Data', color='orangered')

    plt.title('PCA of Original vs. Sampled Data Distributions', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')

    output_path = os.path.join(modelConfig["sampled_dir"], modelConfig["sampledDataName"])
    plt.savefig(output_path)
    print(f"Saved PCA distribution plot to {output_path}")


if __name__ == '__main__':
    modelConfig = {
        "state": "train",  # or "eval"
        "epoch": 50,
        "batch_size": 128,  # Can be larger for 1D data
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 4],  # For 256 -> 128 -> 64 -> 32 -> 16
        # REMOVED: "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "data_len": 256,  # MODIFIED: from img_size
        "grad_clip": 1.,
        "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
        "training_load_weight": None,
        "data_dir": "../Data/latent_vectors/",
        "save_weight_dir": "./Checkpoints_1D/",
        "test_load_weight": "ckpt_1d_47_.pt",  # MODIFIED
        "sampled_dir": "./SampledData/",  # MODIFIED
        "sampledDataName": "Sampled1DData.png",  # MODIFIED
    }

    # Create directories if they don't exist
    if not os.path.exists(modelConfig["save_weight_dir"]):
        os.makedirs(modelConfig["save_weight_dir"])
    if not os.path.exists(modelConfig["sampled_dir"]):
        os.makedirs(modelConfig["sampled_dir"])

    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)