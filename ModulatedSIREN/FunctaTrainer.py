import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from DataLoader import VectorFieldDataset
from FunctaModel import SIRENWithShift
from Utilities.SampleAndNormalization import normalized_random_sample, interpolate_vector_field


def fit(model, device, data_loader, outer_optimizer, outer_criterion, current_epoch, inner_steps, inner_lr):
    losses = []
    latent_features = model.latent_features
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))

    for fields in prog_bar:
        fields = fields.to(device)
        batch_size = fields.size(0)
        # fields(batch_size, 2, H, W) -> fields(batch_size, 2, H*W)
        # fields = fields.view(batch_size, 2, -1).moveaxis(1, -1).to(device)
        coordinates = []
        latent_vectors = []
        # Inner loop.
        for batch_id in range(batch_size):
            coords = normalized_random_sample(16500).to(device)
            latent = torch.zeros(latent_features, device=device, requires_grad=True).float()
            inner_optimizer = optim.SGD([latent], lr=inner_lr)
            # Inner Optimization.
            for step in range(inner_steps):
                # Inner optimizer step.
                inner_optimizer.zero_grad()
                fitted = model(coords, latent)
                inner_loss = inner_criterion(fitted, interpolate_vector_field(fields[batch_id], coords).to(device))
                inner_loss.backward()
                # Clip the gradient.
                torch.nn.utils.clip_grad_norm_([latent], 1)
                # Update.
                inner_optimizer.step()
            latent.requires_grad = False
            coordinates.append(coords)
            latent_vectors.append(latent)

        outer_optimizer.zero_grad()
        outer_loss = torch.tensor(0).to(device).float()
        for batch_id in range(batch_size):
            latent = latent_vectors[batch_id]
            latent = latent.to(device)
            # Outer Optimization.
            fitted = model(coordinates[batch_id],latent)
            outer_loss += outer_criterion(fitted, interpolate_vector_field(fields[batch_id], coordinates[batch_id]).to(device)) / batch_size
        # Outer optimizer step.
        outer_loss.backward()
        # Clip the gradient.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        outer_optimizer.step()
        losses.append(outer_loss.item())

    print(f'epoch: {current_epoch}, loss: {sum(losses) / len(losses)}')
    return sum(losses) / len(losses)



if __name__ == '__main__':
    CONFIG = {
        "input_dim": 2,
        "output_dim": 2,
        "latent_dim": 256,
        "hidden_dim": 512,
        "hidden_layers": 5,
        "batch_size": 72,
        "epochs": 50,
        "outer_learning_rate": 3e-6,
        "inner_learning_rate": 1e-2,
        "inner_steps": 7,
        "num_workers": 4,
        "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
        "dataset_dir" : '../Data/cropped_and_sampled_pt_data',
        "pretrained_dir" : '../Trained_models',
        "checkpoint_path" : '../Trained_models/SIRENWithShift_b72_i7.pth',
        "using_checkpoint": False,
        "normalize_vectors": False,
    }

    print(f"Using device: {CONFIG['device']}")

    dataset = VectorFieldDataset(root_dir=CONFIG['dataset_dir'], normalize_vectors=CONFIG['normalize_vectors'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)

    model = SIRENWithShift(CONFIG["input_dim"], CONFIG["latent_dim"], CONFIG["hidden_dim"], CONFIG["hidden_layers"], CONFIG["output_dim"])
    model.to(CONFIG["device"])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["outer_learning_rate"])
    criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

    os.makedirs(CONFIG['pretrained_dir'], exist_ok=True)

    start_epoch = 0
    best_loss = float('Inf')

    if CONFIG["using_checkpoint"]:
        print(f"✅ Found checkpoint! Loading from {CONFIG['checkpoint_path']}")
        # Note: When loading, wrap with map_location to handle device differences
        checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        best_loss = checkpoint['loss']

        print(f"▶️ Resuming training from epoch {start_epoch}")
    else:
        print("ℹ️ No checkpoint found. Starting training from scratch.")


    for epoch in range(start_epoch, CONFIG["epochs"]):
        loss = fit(model, CONFIG["device"], dataloader, optimizer, criterion, epoch, inner_steps=CONFIG["inner_steps"], inner_lr=CONFIG["inner_learning_rate"])
        if loss < best_loss:
            print(f"🎉 New best loss! Saving model at epoch {epoch}...")
            best_loss = loss
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        }, CONFIG['checkpoint_path'])