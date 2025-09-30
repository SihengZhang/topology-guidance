import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Utilities.SampleAndNormalization import normalized_meshgrid_sample, interpolate_vector_field
from DataLoader import VectorFieldDataset
from FunctaModel import SIRENWithShift


def create_latent_vector_set(
        model,
        device,
        data_loader,
        inner_steps=10,
        inner_lr=0.01,
        result_dir = './'
):
    assert data_loader.batch_size == 1

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
        torch.save(pt_tensor, os.path.join(result_dir, f'latent_vector_{i}.pt'))
        i = i + 1

if __name__ == '__main__':
    CONFIG = {
        "input_dim" : 2,
        "output_dim" : 2,
        "latent_dim" : 256,
        "hidden_dim" : 512,
        "hidden_layers" : 5,
        "inner_steps" : 10,
        "inner_learning_rate" : 1e-2,
        "device" : 'cuda:0' if torch.cuda.is_available() else 'cpu',
        "dataset_dir" : '../Data/cropped_and_sampled_pt_data',
        "pretrained_path" : '../Trained_models/SIRENWithShift_b48_i5.pth',
        "latent_vector_dir" : '../Data/latent_vectors',
        "normalize_vectors" : False,
    }
    print(f"Using device: {CONFIG['device']}")

    model = SIRENWithShift(CONFIG["input_dim"], CONFIG["latent_dim"], CONFIG["hidden_dim"], CONFIG["hidden_layers"], CONFIG["output_dim"])

    pretrained = torch.load(CONFIG['pretrained_path'], map_location=CONFIG['device'])
    model.load_state_dict(pretrained['state_dict'])

    dataset = VectorFieldDataset(root_dir=CONFIG['dataset_dir'], normalize_vectors=CONFIG['normalize_vectors'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    create_latent_vector_set(model, CONFIG['device'], dataloader, inner_steps=CONFIG['inner_steps'],
                             inner_lr=CONFIG['inner_learning_rate'], result_dir=CONFIG['latent_vector_dir'])