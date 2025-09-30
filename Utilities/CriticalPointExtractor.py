import vtk

if __name__ == "__main__":

    # --- Part 1: Load the .vti file ---

    # 1. Create a reader for the XML ImageData format (.vti)
    reader = vtk.vtkXMLImageDataReader()

    # 2. Specify the path to your input
    # Replace this with the actual path to your .vti file
    file_path = "../Data/vector_field_0000_0000.vti"
    reader.SetFileName(file_path)

    # 3. Execute the reader to load the data from the file
    reader.Update()
    print(f"Successfully loaded data from '{file_path}'")

    # 4. Get the grid data from the reader.
    # The output is a vtkImageData object, which is what the filter needs.
    grid = reader.GetOutput()

    # --- Part 2: Your existing analysis code ---

    # 1. Initialize the vector field topology filter
    topology_filter = vtk.vtkVectorFieldTopology()

    # 2. Set the loaded grid as the input for the filter
    topology_filter.SetInputData(grid)
    topology_filter.Update()  # Execute the topology analysis

    # 3. Access the results (critical points)
    critical_points_polydata = topology_filter.GetOutput(0)

    points = critical_points_polydata.GetPoints()
    num_cp = points.GetNumberOfPoints()

    # Get the array that classifies each point (e.g., as a saddle, sink, etc.)
    classification_array = critical_points_polydata.GetPointData().GetArray("typeDetailed")

    if num_cp == 0:
        print("No critical points found in the domain.")
    else:
        print(f"\n--- Found {num_cp} Critical Points ---")

    # Map the integer classification codes to human-readable names
    type_map = {
        -1: "Degenerate",
        0: "Attracting Node (Sink)",
        1: "Attracting Focus (Sink)",
        2: "Saddle",
        3: "Repelling Node (Source)",
        4: "Repelling Focus (Source)",
        5: "Center",
    }

    color_map = {
        -1: (0, 0, 0), # Black
        0: (255, 0, 0), # Red
        1: (255, 165, 0), # Orange
        2: (0, 255, 0), # Green
        3: (0, 0, 255), # Blue
        4: (128, 0, 128), # Purple
        5: (255, 255, 0), # Yellow
    }

    append_filter = vtk.vtkAppendPolyData()

    # Loop through all found points and print their details
    for i in range(num_cp):
        pos = points.GetPoint(i)
        point_type_code = int(classification_array.GetValue(i))
        point_type_str = type_map.get(point_type_code, f"Unknown code ({point_type_code})")
        pos_str = f"({pos[0]:.4f}, {pos[1]:.4f})"
        print(f"  - Point {i}: Position = {pos_str}, Type = {point_type_str}")

        # Create a sphere source
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(pos[0], pos[1], 0.0)
        sphere_source.SetRadius(2.0)
        sphere_source.SetThetaResolution(16)
        sphere_source.SetPhiResolution(16)
        sphere_source.Update()  # Execute the source to generate the geometry

        # Get the polydata object from the sphere source
        sphere_polydata = sphere_source.GetOutput()

        # --- START: COLOR MODIFICATION ---

        # 1. Create a vtkUnsignedCharArray to store color data
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetName("Colors")  # Give the data a name
        color_array.SetNumberOfComponents(3)  # For R, G, B

        # 2. Get the color for the current sphere
        current_color = color_map.get(point_type_code, (255, 255, 255))
        print(current_color)
        # 3. Assign the same color to every point on this sphere
        num_points = sphere_polydata.GetNumberOfPoints()
        for _ in range(num_points):
            color_array.InsertNextTuple3(current_color[0], current_color[1], current_color[2])

        # 4. Attach the color array to the sphere's point data
        sphere_polydata.GetPointData().SetScalars(color_array)

        # Add the modified sphere's polydata (now with color) to the appended filter
        append_filter.AddInputData(sphere_polydata)

    # Combine all the sphere data
    append_filter.Update()

    # Write the combined data to a .vtk file
    writer = vtk.vtkPolyDataWriter()
    output_file_path = "../Data/critical_points_spheres.vtk"
    writer.SetFileName(output_file_path)
    writer.SetInputData(append_filter.GetOutput())
    writer.SetFileTypeToASCII()  # Use ASCII for a human-readable file
    writer.Write()

    print(f"Successfully generated '{output_file_path}' with {num_cp} colored spheres.")
