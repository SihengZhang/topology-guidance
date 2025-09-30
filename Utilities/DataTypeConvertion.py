import sys
import re
import os
import random
import torch
import vtk
from vtkmodules.util import numpy_support as nps
import numpy as np
import pyvista as pv
from tqdm import tqdm

from SampleAndNormalization import normalize_vector_field

def amira_mesh_to_vts(filename: str, output_dir: str="vector_field_in_vts"):
    """
        Reads an AmiraMesh file that defines a scalar/vector field on a uniform grid.

        This function mimics the logic of the original C code but uses Pythonic
        constructs like the `re` module for parsing and NumPy for efficient
        numerical data handling.

        Args:
            filename (str): The path to the AmiraMesh (.am) file.
            output_dir (str): The output directory of .vts files.
    """
    try:
        # Open the file in binary read mode ('rb')
        with open(filename, "rb") as f:
            print(f"Reading {filename}")

            # Read the first 2048 bytes for the header.
            # Decoding with 'latin-1' treats each byte as a character,
            # preventing errors and preserving byte offsets for searching.
            header_bytes = f.read(2048)
            header_str = header_bytes.decode('latin-1')

            # --- 1. Parse Header ---
            if "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1" not in header_str:
                print("Error: Not a proper AmiraMesh file.", file=sys.stderr)
                return None

            # Extract grid dimensions using a regular expression
            match = re.search(r"define Lattice\s+(\d+)\s+(\d+)\s+(\d+)", header_str)
            if not match:
                print("Error: Could not find Lattice definition.", file=sys.stderr)
                return None
            x_dim, y_dim, z_dim = map(int, match.groups())
            print(f"\tGrid Dimensions: {x_dim} {y_dim} {z_dim}")

            # Extract BoundingBox
            match = re.search(
                r"BoundingBox\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)",
                header_str)
            if not match:
                print("Error: Could not find BoundingBox.", file=sys.stderr)
                return None
            x_min, x_max, y_min, y_max, z_min, z_max = map(float, match.groups())
            print(f"\tBoundingBox in x-Direction: [{x_min} ... {x_max}]")
            print(f"\tBoundingBox in y-Direction: [{y_min} ... {y_max}]")
            print(f"\tBoundingBox in z-Direction: [{z_min} ... {z_max}]")

            # Check for uniform coordinates
            is_uniform = "CoordType \"uniform\"" in header_str
            print(f"\tGridType: {'uniform' if is_uniform else 'UNKNOWN'}")

            # Determine the number of data components (scalar vs. vector)
            num_components = 0
            if "Lattice { float Data }" in header_str:
                num_components = 1
            else:
                match = re.search(r"Lattice { float\[(\d+)]", header_str)
                if match:
                    num_components = int(match.group(1))
            print(f"\tNumber of Components: {num_components}")

            # --- 2. Sanity Check ---
            if not all([x_dim > 0, y_dim > 0, z_dim > 0,
                        x_min <= x_max, y_min <= y_max, z_min <= z_max,
                        is_uniform, num_components > 0]):
                print("Error: Invalid header parameters.", file=sys.stderr)
                return None

            # --- 3. Find and Read Binary Data ---
            # The binary data section starts after the line "@1". We search for this
            # marker (including newlines) in the bytes we already read.
            data_marker = b'\r\n# Data section follows\r\n@1\r\n'
            try:
                header_offset = header_bytes.index(data_marker)
                data_start_pos = header_offset + len(data_marker)
            except ValueError:
                print("Error: Could not find data section marker '@1'.", file=sys.stderr)
                return None

            # Seek to the start of the binary data in the file
            f.seek(data_start_pos)

            # Calculate the total number of float values to read
            num_to_read = x_dim * y_dim * z_dim * num_components

            # Use NumPy to read the binary data directly from the file.
            # '<f4' specifies little-endian, 4-byte floats.
            data = np.fromfile(f, dtype=np.dtype('<f4'), count=num_to_read)

            if data.size != num_to_read:
                print("Error: Read incorrect amount of binary data.", file=sys.stderr)
                print(f"Expected {num_to_read} values, but got {data.size}.", file=sys.stderr)
                return None

            # --- 4. Process and Print Data ---
            # Reshape the flat array into a 3D grid. The order 'C' means the last
            # index (x) is the fastest-changing, matching the Amira format.
            if num_components > 1:
                shape = (z_dim, y_dim, x_dim, num_components)
            else:
                shape = (z_dim, y_dim, x_dim)

            data = data.reshape(shape)
            num_timesteps=z_dim
            height=y_dim
            width=x_dim

            if num_components != 2:
                raise ValueError("Input array must have shape (time, height, width, 2)")

            # Create the output path if it doesn't exist
            base_name = os.path.basename(filename)
            base_name_no_ext, extension = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, base_name_no_ext)
            os.makedirs(output_path, exist_ok=True)
            print(f"Saving {num_timesteps} VTK files to '{output_path}/'...")

            # Loop through each time step
            for t in tqdm(range(num_timesteps)):
                # 1. Create a 2D grid
                # The grid points correspond to the pixels/cells of your data.
                x_coords = np.arange(width)
                y_coords = np.arange(height)
                x, y = np.meshgrid(x_coords, y_coords)
                # Create a PyVista StructuredGrid object. Z-coordinates are all 0 for a 2D field.
                grid = pv.StructuredGrid(x.astype(np.float32), y.astype(np.float32), np.zeros_like(x).astype(np.float32))

                # 2. Prepare the vector data for the current time step
                # Get the (height, width, 2) slice for the current time step `t`
                vector_data_2d = data[t, :, :, :]

                # 3. Format vectors for VTK (which expects 3D vectors)
                # Create a zero-filled array for 3D vectors (vx, vy, vz)
                vector_data_3d = np.zeros((height, width, 3), dtype=np.float32)
                # Assign your 2D vectors to the x and y components
                vector_data_3d[:, :, 0:2] = vector_data_2d

                # 4. Assign the vector data to the grid's points
                # The data must be flattened in Fortran ('F') order to match the grid's point order.
                grid.point_data['vectors'] = vector_data_3d.reshape((-1, 3), order='F')

                # 5. Save the grid to a .vts file (XML format for structured grids)
                # Using zero-padded filenames ensures they are sorted correctly.
                filename = os.path.join(output_path, f'vector_field_{t:04d}.vts')
                grid.save(filename)

            print("Conversion complete. ✨")

    except FileNotFoundError:
        print(f"Error: Could not find file {filename}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None


def vts_to_tensor(filename: str, vector_array_name: str='vectors') -> torch.Tensor:
    """
    Reads a .vts file that defines a vector field on a uniform grid into a PyTorch tensor.

    Args:
        filename (str): The path to the VTK structured mesh (.vts) file.
        vector_array_name (str, optional): The name of the vector field to extract.

    Returns:
        torch.Tensor: The PyTorch tensor of the .vts vector field.
    """
    # 1. Read the .vts file, handling a potential file not found errors
    try:
        grid = pv.read(filename)
    except FileNotFoundError:
        print(f"❌ Error: The file '{filename}' was not found.")
        raise

    # 2. Check if the specified vector array name exists
    if vector_array_name not in grid.point_data:
        available_arrays = list(grid.point_data.keys())
        raise KeyError(
            f"❌ Error: Vector array '{vector_array_name}' not found. "
            f"Available arrays are: {available_arrays}"
        )

    # 2. Get the original grid dimensions from the object
    width, height, _ = grid.dimensions

    # 3. Access the flattened vector data array from the grid's point data
    flat_vectors = grid.point_data[vector_array_name]

    # 4. Reshape the data back into its original 2D grid structure
    vector_data_3d = flat_vectors.reshape((height, width, 3), order='F')

    # 5. Extract just the 2D components (vx, vy)
    vector_data_2d_np = vector_data_3d[:, :, :2]

    # 6. Convert the NumPy array to a PyTorch tensor
    vector_data_tensor = torch.from_numpy(vector_data_2d_np)

    return vector_data_tensor


def pt_to_vts(filename: str, output_dir: str):
    """
    Converts a 2D vector field from a PyTorch .pt file to a .vts file using PyVista.

    Args:
        filename (str): Path to the input .pt file (shape 256x256x2).
        output_dir (str): Path to the output .vts dir.
    """
    print(f"Loading tensor from '{filename}' and converting to NumPy...")
    # 1. Load the PyTorch tensor and convert to a NumPy array
    # Move tensor to CPU in case it was saved on a GPU
    tensor_data = torch.load(filename, map_location=torch.device('cpu'))

    tensor_data = normalize_vector_field(tensor_data)

    # Validate tensor shape
    if not (tensor_data.dim() == 3 and tensor_data.shape == (256, 256, 2)):
        raise ValueError(f"Expected tensor of shape (256, 256, 2), but got {tensor_data.shape}")

    numpy_data = tensor_data.numpy()
    height, width, _ = numpy_data.shape  # Assumes shape is (height, width, components)

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

    # 3. Prepare vector data and add it to the grid,
    # Create a 3D vector array by adding a zero Z-component
    vectors_3d = np.zeros((height, width, 3), dtype=np.float32)
    vectors_3d[..., :2] = numpy_data

    # Add the vectors to the grid's point data.
    # PyVista handles the correct ordering and flattening.
    # The name 'VectorField' will be used in visualization software like ParaView.
    grid.point_data['VectorField'] = vectors_3d.reshape(-1, 3)

    # 4. Save the grid to a .vts file
    base_name = os.path.basename(filename)
    base_name_no_ext, extension = os.path.splitext(base_name)
    vts_file_path = os.path.join(output_dir, f'{base_name_no_ext}.vts')
    print(f"Writing .vts file to '{vts_file_path}'...")
    # The .save() method is simple. binary=True is recommended for smaller files.
    grid.save(vts_file_path, binary=True)

    print("✅ Conversion successful!")


def pt_to_vti(filename: str, output_dir: str):
    """
    Converts a 2D vector field from a PyTorch .pt file to a .vti (VTK ImageData) file.

    Args:
        filename (str): Path to the input .pt file (e.g., shape 256x256x2).
        output_dir (str): Path to the output directory for the .vti file.
    """
    try:
        print(f"Loading tensor from '{filename}'...")
        # 1. Load the PyTorch tensor and convert to a NumPy array
        tensor_data = torch.load(filename, map_location=torch.device('cpu'))

        # NOTE: The function 'normalize_vector_field' was called in the original code but not provided.
        # It is commented out here. You may need to implement and call it if normalization is required.
        # tensor_data = normalize_vector_field(tensor_data)

        # Validate tensor shape
        if not (tensor_data.dim() == 3 and tensor_data.shape[2] == 2):
            raise ValueError(f"Expected a 3D tensor with 2 components (e.g., shape HxWx2), but got {tensor_data.shape}")

        numpy_data = tensor_data.numpy().astype(np.float32)
        height, width, _ = numpy_data.shape

        # 2. Create a vtkImageData object
        # This object represents a uniform grid, which is perfect for this kind of data.
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(width, height, 1)  # (nx, ny, nz)
        imageData.SetSpacing(1.0, 1.0, 1.0)  # Size of each voxel/pixel
        imageData.SetOrigin(0.0, 0.0, 0.0)  # World coordinates of the bottom-left corner

        # 3. Prepare the vector data and add it to the vtkImageData
        # The vector data must be 3D (vx, vy, vz). We'll add a zero z-component.
        vectors_3d = np.zeros((width * height, 3), dtype=np.float32)
        # Flatten the 2D numpy data and place it into the 3D array
        vectors_3d[:, :2] = numpy_data.reshape(-1, 2)

        # Convert the NumPy array to a VTK data array
        vtk_vectors = nps.numpy_to_vtk(vectors_3d, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_vectors.SetName("VectorField")  # This name will appear in ParaView/Visit

        # Attach the vector data to the points of the grid.
        # For ImageData, the number of points is width * height * depth.
        imageData.GetPointData().SetVectors(vtk_vectors)
        print("VTK ImageData created and vectors assigned.")

        # 4. Write the grid to a .vti file
        # Use the XML writer for the .vti format.
        writer = vtk.vtkXMLImageDataWriter()
        base_name = os.path.basename(filename)
        base_name_no_ext, _ = os.path.splitext(base_name)
        vti_file_path = os.path.join(output_dir, f'{base_name_no_ext}.vti')

        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        writer.SetFileName(vti_file_path)
        writer.SetInputData(imageData)
        writer.Write()  # Execute the write operation

        print(f"Writing .vti file to '{vti_file_path}'...")
        print("✅ Conversion successful!")

    except FileNotFoundError:
        print(f"❌ Error: Could not find file {filename}", file=sys.stderr)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    pt_to_vti("../Data/cropped_and_sampled_pt_data/vector_field_0000_0000.pt", "../Data")
