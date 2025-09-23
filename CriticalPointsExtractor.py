import vtk
import sys


def main(file_path):
    """
    Loads a 2D .vts file, converts it to a compatible grid type,
    finds critical points, and prints their position and type.
    """
    # 1. Read the .vts file (which creates a vtkStructuredGrid)
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    structured_grid = reader.GetOutput()

    if not structured_grid:
        print(f"Error: Could not read file {file_path}")
        return

    # --- THIS IS THE FIX ---
    # Convert the vtkStructuredGrid to a vtkUnstructuredGrid,
    # which is a compatible input for the topology filter.
    converter = vtk.vtkStructuredGridToUnstructuredGrid()
    converter.SetInputData(structured_grid)
    converter.Update()
    unstructured_grid = converter.GetOutput()
    # -----------------------

    # Set the active vector field by its name on the new grid.
    unstructured_grid.GetPointData().SetActiveVectors("VectorField")

    if not unstructured_grid.GetPointData().GetVectors():
        print("Error: Could not find a vector field named 'VectorField' in the input file.")
        return

    print(f"Successfully loaded and converted {file_path}.")

    # 2. Run the topology filter on the CONVERTED grid
    topology_filter = vtk.vtkVectorFieldTopology()
    topology_filter.SetInputData(unstructured_grid)  # Use the converted grid here
    topology_filter.Update()

    # 3. Access the results (critical points)
    critical_points_polydata = topology_filter.GetOutput(0)

    points = critical_points_polydata.GetPoints()
    num_points = points.GetNumberOfPoints()

    classification_array = critical_points_polydata.GetPointData().GetArray("Classification")

    if num_points == 0:
        print("No critical points found in the domain.")
        return

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
        point_type_code = classification_array.GetValue(i)
        point_type_str = type_map.get(point_type_code, f"Unknown ({point_type_code})")
        pos_str = f"({pos[0]:.4f}, {pos[1]:.4f})"

        print(f"  - Point {i}: Position = {pos_str}, Type = {point_type_str}")


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #    print("Usage: python extract_critical_points.py <path_to_your_file.vts>")
    #    sys.exit(1)

    #file_path = sys.argv[1]
    main("./data/out_1.vts")
