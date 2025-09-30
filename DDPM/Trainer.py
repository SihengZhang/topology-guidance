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
from DDPMModel import UNet
from Scheduler import GradualWarmupScheduler
from Dataset import VectorDataset


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    dataset = VectorDataset(data_dir=modelConfig["data_dir"])
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    if modelConfig["training_load_weight"] is not None:
        # Load from the new checkpoint format
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device)
        net_model.load_state_dict(ckpt['model_state_dict'])
        print("Loaded model weights for continued training.")

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # MODIFICATION: Variables for tracking best model and losses
    best_loss = float('inf')
    epoch_losses = []

    # start training
    for e in range(modelConfig["epoch"]):
        epoch_loss = 0.0
        num_batches = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data, labels in tqdmDataLoader:
                optimizer.zero_grad()
                x_0 = data.to(device)
                loss = trainer(x_0).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()

                # Accumulate loss for the epoch average
                epoch_loss += loss.item()
                num_batches += 1

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "data shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        # Calculate and record the average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {e} | Average Loss: {avg_epoch_loss:.4f}")

        warmUpScheduler.step()

        # MODIFICATION: Save only the best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

            # Create checkpoint dictionary
            checkpoint = {
                'epoch': e,
                'model_state_dict': net_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }

            # Save the single best model checkpoint
            save_path = os.path.join(modelConfig["save_weight_dir"], 'best_model.pt')
            torch.save(checkpoint, save_path)
            print(f"✨ New best model saved at epoch {e} with loss {best_loss:.4f} to {save_path}")

    # MODIFICATION: Plot and save the loss curve after training is finished
    print("Training finished. Plotting loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses)
    plt.yscale('log')  # Use log scale for the y-axis
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plot_path = os.path.join(modelConfig["save_weight_dir"], 'training_loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

    # MODIFICATION: Save the recorded losses to a .pt file
    losses_save_path = os.path.join(modelConfig["save_weight_dir"], 'training_losses.pt')
    torch.save(epoch_losses, losses_save_path)
    print(f"Loss history saved to {losses_save_path}")


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # Load Model
    model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                 num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)

    # MODIFICATION: Load state_dict from the checkpoint dictionary
    ckpt_path = os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"])
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(
        f"Model loaded successfully from {ckpt_path} (trained for {ckpt['epoch']} epochs with loss {ckpt['loss']:.4f}).")
    model.to(device)  # Make sure model is on the correct device
    model.eval()

    sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # 1. Get original data from the directory
    print("Loading original data for comparison...")
    original_dataset = VectorDataset(data_dir=modelConfig["data_dir"])
    original_dataloader = DataLoader(original_dataset, batch_size=modelConfig["batch_size"], shuffle=False,
                                     num_workers=4)

    original_data_list = []
    for batch, _ in tqdm(original_dataloader, desc="Loading original vectors"):
        original_data_list.append(batch.cpu().numpy())

    original_data = np.concatenate(original_data_list, axis=0).squeeze()
    num_samples = len(original_data)
    print(f"Loaded {num_samples} original data samples.")

    # 2. Generate the same number of sampled data
    sampled_data_list = []
    batch_size = modelConfig["batch_size"]
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i)
            noisy_data = torch.randn(size=[current_batch_size, 1, modelConfig["data_len"]], device=device)
            sampled_batch = sampler(noisy_data)
            sampled_data_list.append(sampled_batch.cpu().numpy().squeeze())

    sampled_data = np.concatenate(sampled_data_list, axis=0)
    print(f"Generated {len(sampled_data)} new data samples.")

    # --- PCA and Plotting ---

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
        "state": "eval",
        "epoch": 500,
        "batch_size": 128,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 4],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "data_len": 256,
        "grad_clip": 1.,
        "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
        "training_load_weight": None,
        "data_dir": "../Data/latent_vectors/",
        "save_weight_dir": "./Checkpoints_1D/",
        # MODIFICATION: Point to the single best model file
        "test_load_weight": "best_model.pt",
        "sampled_dir": "./SampledData/",
        "sampledDataName": "Sampled1DData.png",
    }

    if not os.path.exists(modelConfig["save_weight_dir"]):
        os.makedirs(modelConfig["save_weight_dir"])
    if not os.path.exists(modelConfig["sampled_dir"]):
        os.makedirs(modelConfig["sampled_dir"])

    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)