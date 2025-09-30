import torch
import torch.nn.functional as F


def normalized_meshgrid_sample(height: int, width: int) -> torch.Tensor:
    """
    Generates a structured grid of (x, y) coordinates and normalizes them to [-1, 1].

    This function creates a grid of coordinates corresponding to pixel locations
    for a given height and width. It then transforms these coordinates from the
    pixel space (e.g., [0, width-1]) to a normalized device coordinate space [-1, 1].

    Args:
        height (int): The height of the grid.
        width (int): The width of the grid.

    Returns:
        torch.Tensor: A tensor of shape (height * width, 2) containing the
                      normalized (x, y) coordinates. Each row is an [x, y] pair.
    """
    # --- Mesh Grid Generation ---
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    # Stack the flattened coordinate tensors along dimension 1 to get (H*W, 2) shape
    meshgrid = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).float()

    # --- Normalization Step ---
    # Create scale and bias tensors that will broadcast correctly.
    scale = torch.tensor([2.0 / (width - 1), 2.0 / (height - 1)])
    bias = torch.tensor([-1.0, -1.0])

    normalized_meshgrid = meshgrid * scale + bias
    return normalized_meshgrid

def normalized_random_sample(n: int) -> torch.Tensor:
    """
    Generates n random (x, y) coordinates uniformly sampled from the [-1, 1] domain.

    Args:
        n (int): The number of random coordinate pairs to generate.

    Returns:
        torch.Tensor: A tensor of shape (n, 2) containing the random normalized
                      (x, y) coordinates. Each row is an [x, y] pair.
    """
    # torch.rand(n, 2) directly creates the desired (n, 2) shape.
    # We then scale by 2.0 and subtract 1.0 to shift the range from [0, 1) to [-1, 1).
    return torch.rand(n, 2) * 2.0 - 1.0

def normalize_vector_field(vector_field: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Normalizes a 2D vector field represented by a PyTorch tensor.

    This function converts each vector in the field to a unit vector by dividing
    it by its L2 norm (magnitude).

    Args:
        vector_field (torch.Tensor): A tensor of shape (height, width, 2)
                                     representing the 2D vector field.
        epsilon (float): A small value added to the denominator to prevent
                         division by zero for zero-magnitude vectors.

    Returns:
        torch.Tensor: The normalized vector field with the same shape as the input.
                      Each vector will have the length of approximately 1, except
                      for original zero vectors, which will remain zero.
    """
    # Calculate the L2 norm (magnitude) along the last dimension (dim=2).
    # The shape of the input is (height, width, 2).
    # We calculate the norm of each of the 2-element vectors.
    # `keepdim=True` is essential. It makes the output shape (height, width, 1)
    # instead of (height, width), which allows for broadcasting during division.
    magnitudes = torch.norm(vector_field, p=2, dim=2, keepdim=True)

    # Add epsilon to the magnitudes to avoid division by zero.
    # This ensures numerical stability.
    stable_magnitudes = magnitudes + epsilon

    # Divide the original vector field by the magnitudes.
    # Thanks to broadcasting, each (vx, vy) pair is divided by its corresponding magnitude.
    normalized_field = vector_field / stable_magnitudes

    return normalized_field

def interpolate_vector_field(vector_field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Samples a vector field at given normalized coordinates using bilinear interpolation.

    Args:
        vector_field (torch.Tensor): A tensor of shape (2, H, W) representing the
                                     2D vector field. The first channel is the
                                     x-component, the second is the y-component.
        coords (torch.Tensor): A tensor of shape (N, 2) with N coordinates in the
                               range [-1, 1]. Each row is an [x, y] pair.

    Returns:
        torch.Tensor: A tensor of shape (N, 2) containing the interpolated
                      vector values at the given coordinates.
    """
    # --- Input Reshaping for grid_sample ---
    # grid_sample expects the input tensor (vector_field) to be in the format
    # (N, C, Hin, Win), so we add a batch dimension of 1.
    # Input shape: (2, H, W) -> (1, 2, H, W)
    vector_field_batch = vector_field.unsqueeze(0)

    # grid_sample expects the grid (coords) to be in the format (N, Hout, Wout, 2).
    # We are sampling N individual points, so Hout=1, Wout=N.
    # Coords shape: (N, 2) -> Reshape to (1, 1, N, 2)
    num_points = coords.shape[0]
    coords_grid = coords.view(1, 1, num_points, 2)

    # --- Interpolation ---
    # `align_corners=True` is crucial. It means the [-1, 1] range corresponds
    # to the centers of the corner pixels, which matches our normalization logic.
    interpolated_values = F.grid_sample(
        vector_field_batch,
        coords_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # --- Output Reshaping ---
    # The output of grid_sample is (N, C, Hout, Wout), which is (1, 2, 1, N).
    # We squeeze and transpose to get the desired (N, 2) shape.
    return interpolated_values.squeeze().T
