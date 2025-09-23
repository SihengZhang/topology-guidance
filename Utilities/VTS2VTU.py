import vtk

if __name__ == '__main__':
    # --- Configuration ---
    # Specify the path to your input .vts file
    input_filename = 'input_vector_field.vts'

    # Specify the desired name for the output .vtu file
    output_filename = 'output_vector_field.vtu'
    # ---------------------

    # 1. Reader: Create a reader for the .vts file
    print(f"Reading data from '{input_filename}'...")
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(input_filename)

    # 2. Filter: Cast the structured grid to an unstructured grid.
    # This is the corrected step. vtkCastToUnstructuredGrid preserves the
    # volumetric cells, unlike vtkGeometryFilter which only extracts the surface.
    print("Casting from structured to unstructured grid...")
    cast_filter = vtk.vtkCastToUnstructuredGrid()
    cast_filter.SetInputConnection(reader.GetOutputPort())

    # 3. Writer: Create a writer for the .vtu file
    print(f"Writing data to '{output_filename}'...")
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_filename)
    writer.SetInputConnection(cast_filter.GetOutputPort())  # Connect to the new filter

    # Set the data mode to binary for smaller file sizes
    writer.SetDataModeToBinary()

    # Execute the entire pipeline by calling Write() on the final stage
    writer.Write()

    print("\n✅ Conversion complete!")