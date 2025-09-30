import os
import random
import torch
from tqdm import tqdm

from DataTypeConvertion import vts_to_tensor

def random_square_crop(vector_data_2d: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Randomly crops a square section from a 2D vector field tensor.

    Args:
        vector_data_2d (torch.Tensor): The input tensor with shape (height, width, 2).
        crop_size (int): The side length of the square crop.

    Returns:
        torch.Tensor: The cropped square tensor with shape (crop_size, crop_size, 2).
    """
    # Get the original dimensions
    original_height, original_width, _ = vector_data_2d.shape

    # Check if the crop size is valid
    if crop_size > original_height or crop_size > original_width:
        raise ValueError("Crop size cannot be larger than the original dimensions.")

    # Determine the latest possible starting point for the crop
    max_start_y = original_height - crop_size
    max_start_x = original_width - crop_size

    # Randomly select the top-left corner using torch.randint
    # .item() extracts the integer value from the tensor
    start_y = torch.randint(0, max_start_y + 1, (1,)).item()
    start_x = torch.randint(0, max_start_x + 1, (1,)).item()

    # Perform the crop using tensor slicing ✂️
    cropped_field = vector_data_2d[
        start_y : start_y + crop_size,
        start_x : start_x + crop_size,
        :
    ]

    return cropped_field

def create_dataset(input_dir: str, output_dir: str, number_of_samples: int, crop_size: int):
    """
        Converts a 2D vector field from .vts files to PyTorch .pt files using PyVista.

        Args:
            input_dir(str): path to the directory containing the .vts files.
            output_dir (str): Path to the output .pt dir.
            number_of_samples(int): number of random samples to create.
            crop_size (int): resolution of the cropped field.
        """
    try:
        # os.path.basename gets the final component
        dir_name = os.path.basename(os.path.normpath(input_dir))
        os.makedirs(output_dir, exist_ok=True)

        # 1. Create a list of all files ending with .vts
        vts_files = [f for f in os.listdir(input_dir) if f.endswith('.vts')]

        # 2. If the list is not empty, choose one file randomly
        if len(vts_files) == 0:
            raise ValueError(f"No .vts files found in '{input_dir}'")

        for i in tqdm(range(number_of_samples)):
            random_filename = random.choice(vts_files)
            filename = os.path.join(input_dir, random_filename)
            cropped_field = random_square_crop(vts_to_tensor(filename), crop_size)
            torch.save(cropped_field, os.path.join(output_dir, f'vector_field_{dir_name}_{i:04d}.pt'))

    except FileNotFoundError:
        print(f"Error: Directory not found at '{input_dir}'")
        return None

if __name__ == "__main__":
    create_dataset( '../Data/vector_field_in_vts/0000','../Data/cropped_and_sampled_pt_data', 5000, 256)