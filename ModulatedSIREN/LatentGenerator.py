import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.xpu import device
from vector_field_sampler import normalized_random_sample, normalized_meshgrid_sample, interpolate_vector_field
from dataloader import get_vector_loader
from FunctaModel import SIRENWithShift
import joblib
from tqdm import tqdm

def create_functaset(
        model,
        data_loader,
        inner_steps=1000,
        inner_lr=0.01,
):
    """
    :param model:
    :param data_loader:
    :param inner_steps:
    :param inner_lr:
    """
    assert data_loader.batch_size == 1
    device = next(iter(model.parameters())).device
    latent_features = model.latent_features
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))
    i = 1
    for fields in prog_bar:
        fields = fields.to(device)
        coords = normalized_meshgrid_sample(256, 256).to(device)
        latent = torch.zeros(latent_features, requires_grad=True).float().to(device)
        inner_optimizer = optim.SGD([latent], lr=inner_lr)
        mse = 0
        # Inner Optimization.
        for step in range(inner_steps):
            fitted = model(coords, latent)
            inner_loss = inner_criterion(fitted, interpolate_vector_field(fields[0], coords).to(device))
            mse = inner_loss.item()

            # Inner optimizer step.
            inner_optimizer.zero_grad()
            inner_loss.backward()
            # Clip the gradient.
            torch.nn.utils.clip_grad_norm_([latent], 1)
            # Update.
            inner_optimizer.step()
        prog_bar.set_description(f'MSE: {mse}')

        pt_tensor = torch.from_numpy(latent.detach().cpu().numpy())
        torch.save(pt_tensor, os.path.join('./results', f'latent_vector_{i}.pt'))
        i = i + 1

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = SIRENWithShift(2, 256, 512, 5, 2)
    pretrained = torch.load('trained_models/SIRENWithShift.pth', map_location=device)
    model.load_state_dict(pretrained['state_dict'])
    dataloader=get_vector_loader('./sample', train=True, batch_size=1)
    # dataloader = get_vector_loader('./data_to_get_latent', train=True, batch_size=1)
    create_functaset(model, dataloader, inner_steps=10)