import vtk
import numpy as np
from vtkmodules.util import numpy_support

if __name__ == "__main__":
    # 1. DEFINE GRID PARAMETERS AND DATA
    # ------------------------------------
    grid_width = 10
    grid_height = 5
    file_path = '../Data/vector_field.vtu'

    # Total number of points
    num_points = grid_width * grid_height

    # Create point coordinates (X, Y, Z)
    # We create a regular grid of points. Z is always 0 for a 2D field.
    x = np.arange(0, grid_width, 1)
    y = np.arange(0, grid_height, 1)
    xx, yy = np.meshgrid(x, y)
    points_array = np.zeros((num_points, 3), dtype=np.float32)
    points_array[:, 0] = xx.flatten()
    points_array[:, 1] = yy.flatten()

    # Create vector data (Vx, Vy, Vz)
    # This is a sample rotational field. Vz is 0.
    center_x, center_y = (grid_width - 1) / 2.0, (grid_height - 1) / 2.0
    vectors_array = np.zeros((num_points, 3), dtype=np.float32)
    vectors_array[:, 0] = -(points_array[:, 1] - center_y)
    vectors_array[:, 1] = (points_array[:, 0] - center_x)

    print(f"Generated a {grid_height}x{grid_width} grid with {num_points} points.")



    # 2. CREATE VTK OBJECTS
    # ---------------------
    # Create a vtkPoints object and store the points in it
    vtk_points = vtk.vtkPoints()
    # Convert the NumPy array to a VTK data array
    vtk_data_array = numpy_support.numpy_to_vtk(points_array)
    vtk_points.SetData(vtk_data_array)

    # Create the unstructured grid
    grid = vtk.vtkUnstructuredGrid()
    # Set the points for the grid
    grid.SetPoints(vtk_points)

    # 3. DEFINE THE CELLS (TOPOLOGY)
    # ------------------------------
    # Iterate through the grid to create quadrilateral cells (vtkQuad)
    # A quad is defined by 4 point IDs.
    for j in range(grid_height - 1):
        for i in range(grid_width - 1):
            # Point IDs for the four corners of the cell
            p1 = j * grid_width + i
            p2 = j * grid_width + (i + 1)
            p3 = (j + 1) * grid_width + (i + 1)
            p4 = (j + 1) * grid_width + i

            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, p1)
            quad.GetPointIds().SetId(1, p2)
            quad.GetPointIds().SetId(2, p3)
            quad.GetPointIds().SetId(3, p4)

            # Add the cell to the grid
            grid.InsertNextCell(quad.GetCellType(), quad.GetPointIds())
    # 4. ADD VECTOR DATA TO THE POINTS
    # --------------------------------
    # Convert the NumPy vectors array to a VTK data array
    vtk_vectors = numpy_support.numpy_to_vtk(vectors_array, deep=True)
    vtk_vectors.SetName("Velocity")  # Set the name of the vector field

    # Add the vectors to the grid's point data
    grid.GetPointData().SetVectors(vtk_vectors)

    topology_filter = vtk.vtkVectorFieldTopology()
    topology_filter.SetInputData(grid)  # Use the converted grid here
    topology_filter.Update()

    # 3. Access the results (critical points)
    critical_points_polydata = topology_filter.GetOutput(0)

    points = critical_points_polydata.GetPoints()
    num_points = points.GetNumberOfPoints()

    classification_array = critical_points_polydata.GetPointData().GetArray("Classification")

    if num_points == 0:
        print("No critical points found in the domain.")

    print(f"\n--- Found {num_points} Critical Points ---")

    type_map = {
        2: "Saddle",
        3: "Repelling Node (Source)",
        4: "Attracting Node (Sink)",
        5: "Center",
        6: "Repelling Spiral (Source)",
        7: "Attracting Spiral (Sink)"
    }

    for i in range(num_points):
        pos = points.GetPoint(i)
        # point_type_code = classification_array.GetValue(i)
        # point_type_str = type_map.get(point_type_code, f"Unknown ({point_type_code})")
        pos_str = f"({pos[0]:.4f}, {pos[1]:.4f})"
        print(f"  - Point {i}: Position = {pos_str}")
        # print(f"  - Point {i}: Position = {pos_str}, Type = {point_type_str}")

    # 5. WRITE THE GRID TO A FILE
    # ---------------------------
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(grid)
    writer.Write()

    print(f"Successfully saved the vector field to '{file_path}'")