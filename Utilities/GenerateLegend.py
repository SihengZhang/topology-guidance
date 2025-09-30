from PIL import Image, ImageDraw, ImageFont

def generate_legend_image():
    """
    Generates a PNG image file representing the legend for critical point types.
    """
    # --- 1. Define Data Mappings ---
    # These dictionaries map the critical point type codes to their names and colors.
    type_map = {
        0: "Attracting Node (Sink)",
        1: "Attracting Focus (Sink)",
        2: "Saddle",
        3: "Repelling Node (Source)",
        4: "Repelling Focus (Source)",
        5: "Center",
        -1: "Degenerate",
    }

    color_map = {
        0: (255, 0, 0),  # Red
        1: (255, 165, 0),  # Orange
        2: (0, 255, 0),  # Green
        3: (0, 0, 255),  # Blue
        4: (128, 0, 128),  # Purple
        5: (255, 255, 0),  # Yellow (Corrected from the original script)
        -1: (0, 0, 0),  # Black
    }

    # --- 2. Set Up Image and Drawing Parameters ---
    padding = 20
    swatch_size = 18
    line_height = 30
    title_height = 40
    font_size = 15

    # Calculate image dimensions dynamically
    num_items = len(type_map)
    img_width = 320
    img_height = title_height + (num_items * line_height) + padding

    # Create a blank white image
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    # --- 3. Load Font ---
    # Tries to load a common system font; falls back to a default if not found.
    try:
        title_font = ImageFont.truetype("arialbd.ttf", font_size + 2)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Arial font not found. Using default font.")
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()

    # --- 4. Draw Title ---
    draw.text(
        (padding, padding // 2),
        "Critical Point Legend",
        fill="black",
        font=title_font
    )

    # --- 5. Draw Legend Items ---
    # Loop through the map and draw a color swatch and text for each item.
    current_y = title_height
    for type_code, type_name in type_map.items():
        color = color_map.get(type_code, (255, 255, 255))  # Default to white if not found

        # Position for the color swatch (rectangle)
        swatch_x1 = padding
        swatch_y1 = current_y
        swatch_x2 = swatch_x1 + swatch_size
        swatch_y2 = swatch_y1 + swatch_size

        # Draw the color swatch
        draw.ellipse(
            [swatch_x1, swatch_y1, swatch_x2, swatch_y2],
            fill=color,
            outline=None
        )

        # Position for the text label
        text_x = swatch_x2 + 10
        text_y = current_y + (swatch_size - font.getbbox(type_name)[3]) // 2

        # Draw the text label
        draw.text((text_x, text_y), type_name, fill="black", font=font)

        current_y += line_height

    # --- 6. Save the Image ---
    output_filename = "../Data/critical_point_legend.png"
    image.save(output_filename)
    print(f"✅ Successfully created legend: '{output_filename}'")


if __name__ == "__main__":
    generate_legend_image()