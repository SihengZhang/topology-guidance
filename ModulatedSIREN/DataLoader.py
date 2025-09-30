import torch
import os
import glob
from ..Utilities.SampleAndNormalization import normalize_vector_field
from torch.utils.data import Dataset
import torchvision.transforms as T

class VectorFieldDataset(Dataset):
    """Custom Dataset for loading 2D vector fields."""
    def __init__(self, root_dir, normalize_vectors=False):
        """
        Args:
            root_dir (string): Directory with all the vector field files.
        """
        # Get a list of all file paths
        self.file_paths = glob.glob(os.path.join(root_dir, '*.pt'))
        self.normalize_vectors = normalize_vectors

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Loads and returns a sample from the dataset at the given index."""
        # Get the file path
        file_path = self.file_paths[idx]
        # Load the tensor from the file
        vector_field = torch.load(file_path)

        if self.normalize_vectors:
            vector_field = normalize_vector_field(vector_field)

        # Permute dimensions from (H, W, C) to (C, H, W)
        vector_field = vector_field.permute(2, 0, 1)

        return vector_field


class TensorFileDataset(Dataset):
    """
    A custom PyTorch Dataset to load tensors from individual .pt files.

    Each .pt file in the specified directory is considered a single data sample.
    """

    def __init__(self, root_dir):
        """
        Initializes the dataset.

        Args:
            root_dir (str): The path to the directory containing the .pt files.
        """
        self.root_dir = root_dir
        # Create a sorted list of all files ending with .pt for deterministic order
        self.file_list = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f)) and f.endswith('.pt')
        ])

        if not self.file_list:
            raise ValueError(f"No .pt files found in the directory: {root_dir}")

    def __len__(self):
        """
        Returns the total number of tensor files.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Fetches the tensor at the given index.

        Args:
            idx (int): The index of the file to load.

        Returns:
            torch.Tensor: The loaded tensor from the .pt file.
        """
        if not 0 <= idx < len(self.file_list):
            raise IndexError(f"Index {idx} is out of bounds for the dataset with size {len(self)}")

        file_path = os.path.join(self.root_dir, self.file_list[idx])

        try:
            # Load the tensor from the file
            tensor = torch.load(file_path)
            return tensor
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # Return a zero tensor or handle the error as appropriate
            return torch.zeros(256)