import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyvista as pv
import torch

from FunctaModel import SIRENWithShift
from Utilities.SampleAndNormalization import normalized_meshgrid_sample

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # latent = torch.load('./results/latent_vector_2.pt', map_location=device)
    latent = torch.load('./results/generated_500.pt', map_location=device)[0]
    print(latent)
    print(latent.shape)
    model = SIRENWithShift(2, 256, 512, 5, 2)
    pretrained = torch.load('trained_models/SIRENWithShift_normalized.pth', map_location=device)
    model.load_state_dict(pretrained['state_dict'])
    coords = normalized_meshgrid_sample(256, 256)
    evaluated=model(coords, latent)
    print(evaluated.shape)
    evaluated=evaluated.detach().cpu().numpy().reshape(256, 256, 2)

    height, width, _ = evaluated.shape  # Assumes shape is (height, width, components)

    # 2. Create the PyVista StructuredGrid
    # Create coordinate arrays. The grid will be on the Z=0 plane.
    x = np.arange(width)
    y = np.arange(height)
    z = np.array([0])

    # Use np.meshgrid to create 3D coordinate matrices
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Create the grid object directly from the coordinate matrices
    grid = pv.StructuredGrid(xx, yy, zz)
    print("PyVista grid created.")

    # 3. Prepare vector data and add it to the grid
    # Create a 3D vector array by adding a zero Z-component
    vectors_3d = np.zeros((height, width, 3), dtype=np.float32)
    vectors_3d[..., :2] = evaluated

    # Add the vectors to the grid's point data.
    # PyVista handles the correct ordering and flattening.
    # The name 'VectorField' will be used in visualization software like ParaView.
    grid.point_data['VectorField'] = vectors_3d.reshape(-1, 3)

    # 4. Save the grid to a .vts file
    vts_file_path = './results/out_3.vts'
    print(f"Writing .vts file to '{vts_file_path}'...")
    # The .save() method is simple. binary=True is recommended for smaller files.
    grid.save(vts_file_path, binary=True)

    print("✅ Conversion successful!")