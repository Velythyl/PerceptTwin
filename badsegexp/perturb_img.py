import os
from PIL import Image, ImageDraw
import random

def process_images(path, radius, NUM_CIRCLES):
    # Supported image extensions
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(extensions):
                try:
                    filepath = os.path.join(root, file)

                    if "/mask/" in filepath:
                        continue

                    with Image.open(filepath) as img:
                        # Convert to RGBA if not already to support transparency
                        img = img.convert("RGBA")
                        pixels = img.load()  # Create pixel access object
                        
                        radius = int(min([img.width, img.height]) * 0.15)

                        # Draw 3 random circles
                        for _ in range(NUM_CIRCLES):
                            # Random position ensuring circle stays within image bounds
                            x = random.randint(radius, img.width - radius - 1)
                            y = random.randint(radius, img.height - radius - 1)
                            
                            # Iterate through all pixels in the image
                            for i in range(img.width):
                                for j in range(img.height):
                                    # Check if pixel is within the circle
                                    if (i - x)**2 + (j - y)**2 <= radius**2:
                                        # Make pixel transparent
                                        pixels[i, j] = (0, 0, 0, 0)
                        
                        # Save the modified image
                        temp_path = filepath + "_temp.png"
                        img.save(temp_path)
                        
                        # Replace original file (more atomic than direct overwrite)
                        os.remove(filepath)
                        os.rename(temp_path, filepath)
                        print(f"Processed: {filepath}")
                        
                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")

if __name__ == "__main__":
    # Configuration - change these values as needed
    PATH = "/home/mila/c/charlie.gauthier/Holodeck/hippo/datasets/badseg/replica_room0_banner/segments"
    RADIUS = 25        # Radius of circles in pixels
    NUM_CIRCLES = 10    # Number of circles to draw per image
    
    # Run the processing
    process_images(PATH, RADIUS, NUM_CIRCLES)
    print("Processing complete!")