import os
from typing import Tuple
from tqdm import tqdm
from PIL import Image, ImageOps
import math

def transform(img):
    #return img
    # Identity transform for now
    return border(crop(img, (650,1275), (400, 1600)))

def crop(img: Image.Image, x: Tuple[int, int], y: Tuple[int, int]) -> Image.Image:
    """
    Crop an image using x and y coordinate ranges.

    Parameters:
        img: PIL.Image.Image — the input image
        x: Tuple[int, int] — horizontal pixel bounds (x_min, x_max)
        y: Tuple[int, int] — vertical pixel bounds (y_min, y_max)

    Returns:
        A cropped Image.
    """
    x_min, x_max = x
    y_min, y_max = y
    return img.crop((x_min, y_min, x_max, y_max))

def border(img: Image.Image, border_width: int=5) -> Image.Image:
    """
    Add a black border of given width to the image.

    Parameters:
        img: PIL.Image.Image — the input image
        border_width: int — thickness of the border in pixels

    Returns:
        A new image with the border added.
    """
    return ImageOps.expand(img, border=border_width, fill='black')

def find_topdown_images(root_dir):
    topdown_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'topdown.png':
                topdown_paths.append(os.path.join(dirpath, filename))
    return topdown_paths

def load_and_transform_images(paths):
    images = []
    for path in tqdm(paths):
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
    return images

def create_image_grid(images):
    if not images:
        return None

    # Determine grid size (closest to square)
    count = len(images)
    grid_cols = math.ceil(math.sqrt(count))
    grid_rows = math.ceil(count / grid_cols)


    grid_cols = grid_cols # // 2
    grid_rows = math.ceil(count / grid_cols)

    grid_cols = int(grid_cols)
    grid_rows = int(grid_rows)

    # Assume all images are the same size
    width, height = images[0].size
    grid_image = Image.new('RGB', (grid_cols * width, grid_rows * height))

    for idx, img in enumerate(tqdm(images)):
        row = idx // grid_cols
        col = idx % grid_cols
        grid_image.paste(img, (col * width, row * height))

    # Check and resize if necessary
    max_size = 3000
    if grid_image.width > max_size or grid_image.height > max_size:
        print("Final image was too large for LaTeX, resizing it.")
        scale_factor = max_size / max(grid_image.width, grid_image.height)
        print("Resize factor:", scale_factor)
        new_size = (
            int(grid_image.width * scale_factor),
            int(grid_image.height * scale_factor)
        )
        grid_image = grid_image.resize(new_size, Image.LANCZOS)

    grid_image = grid_image.rotate(90, expand=True)

    return grid_image

# Example usage
if __name__ == "__main__":
    root_directory = "/Users/charlie/Projects/Holodeck/hippo/sampled_scenes/RANDOM_SACHA_KITCHEN/filtered"
    paths = find_topdown_images(root_directory)
    images = load_and_transform_images(paths)
    grid = create_image_grid(images)
    if grid:
        grid.save("scenegen.jpeg")
        print("DONE")
