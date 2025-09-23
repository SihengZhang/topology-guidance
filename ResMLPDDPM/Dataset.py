import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class VectorDataset(Dataset):
    """
    A dataset that loads vectors from a directory of .pt files.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Create a list of all .pt file paths in the directory
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        if not self.file_paths:
            raise ValueError(f"No .pt files found in the directory: {data_dir}")
        print(f"Found {len(self.file_paths)} vector files.")

    def __len__(self):
        """Returns the total number of files."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Loads and returns a vector from a .pt file."""
        file_path = self.file_paths[idx]
        # Load the tensor from the file
        vector = torch.load(file_path)

        # Return 0 for the label, as it's unused
        return vector, 0
