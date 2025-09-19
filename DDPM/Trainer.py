import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from Scheduler import GradualWarmupScheduler
from Dataset import SineWaveDataset


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # MODIFIED: Use the new SineWaveDataset
    dataset = SineWaveDataset(num_samples=20000, seq_len=modelConfig["data_len"])
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # MODIFIED: model setup for 1D UNet (removed attn)
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
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
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data, labels in tqdmDataLoader:
                optimizer.zero_grad()
                x_0 = data.to(device)
                # MODIFIED: Using mean() is more standard than sum() / 1000
                loss = trainer(x_0).mean()
                loss.backward()
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
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_1d_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        # MODIFIED: model setup for 1D UNet (removed attn)
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        # MODIFIED: Sample 1D noise
        noisy_data = torch.randn(
            size=[modelConfig["batch_size"], 1, modelConfig["data_len"]], device=device)

        sampled_data = sampler(noisy_data)

        # MODIFIED: Plotting results instead of saving images
        sampled_data = sampled_data.cpu().numpy().squeeze()  # Move to CPU and remove channel dim

        # Plot a few samples
        num_to_plot = min(modelConfig["batch_size"], 8)
        fig, axes = plt.subplots(num_to_plot, 1, figsize=(10, 2 * num_to_plot))
        for i in range(num_to_plot):
            axes[i].plot(sampled_data[i])
            axes[i].set_title(f"Sample {i + 1}")
            axes[i].set_ylim(-1.1, 1.1)
        plt.tight_layout()

        output_path = os.path.join(modelConfig["sampled_dir"], modelConfig["sampledDataName"])
        plt.savefig(output_path)
        print(f"Saved generated samples plot to {output_path}")


if __name__ == '__main__':
    modelConfig = {
        "state": "train",  # or "eval"
        "epoch": 200,
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
        "save_weight_dir": "./Checkpoints_1D/",
        "test_load_weight": "ckpt_1d_199_.pt",  # MODIFIED
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