###
# Author: Tobias
# Example code of how the crops for the Chicks4FreeID dataset has been done.
# If you want to do the preprocessing by yourself, this code additionally requires opencv (Apache2.0)

from pathlib import Path
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm.rich import tqdm

# Adjust the path as needed
data_dir = Path('~/Downloads/chicks')
# Code will look for images like this:
# files = list(data_dir.glob('image_*.png'))

mask_dir = data_dir / "240509_masks"
# Code will look for instance mask files like this:
# files = list(mask_dir.glob('mask_*_*.png'))


# Patches will be outputted to the same directory as the masks 

assert data_dir.is_dir()
assert mask_dir.is_dir()




def make_square_image(image, size, fill_color=(0, 0, 0)):
    """
    Create a square image from any rectangular image by specifying the square size.

    Parameters:
    - image: A numpy array representing the image.
    - size: The desired side length of the square image.
    - fill_color: A tuple specifying the color used for padding. Default is black.

    Returns:
    - A numpy array representing the square image.
    """
    # Create a new square canvas filled with the fill_color
    image = np.array(image)

    square_image = np.full((size, size, image.shape[2]), fill_color, dtype=np.uint8)

    # Calculate coordinates to center the original image on the square canvas
    y_offset = (size - image.shape[0]) // 2
    x_offset = (size - image.shape[1]) // 2

    # Place the original image in the center of the square canvas
    square_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

    return Image.fromarray(square_image)


def find_bounding_box(mask):
    """
    Finds the bounding box for a given numpy boolean mask.

    Parameters:
    - mask: A numpy array of shape (height, width) where the mask is boolean.

    Returns:
    - A tuple (x, y, width, height) representing the bounding box of the non-zero area.
    """
    # Ensure the mask is boolean
    mask = mask.astype(bool)

    # Find the indices of the rows and columns that contain True values
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]

    if rows.size and cols.size:
        # Compute the bounding box corners
        y_min, y_max = rows[[0, -1]]
        x_min, x_max = cols[[0, -1]]

        # Calculate width and height
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        return x_min, y_min, width, height
    else:
        # Return a default or an error if the mask is empty
        return 0, 0, 0, 0


def extract_patches_with_no_background():
    # List all the image and mask files
    files = list(mask_dir.glob('mask_*_*.png'))

    # Filter out the individual mask files
    individual_masks = [f for f in files]

    for mask_path in tqdm(individual_masks):
        # Construct the image file name corresponding to the mask
        image_name = mask_path.name.split('_')[1]
        image_path = data_dir / f'image_{image_name}.png'

        if not image_path.exists():
            # print(f"Image file {image_path} does not exist.")
            continue
        print(f"Processing {image_path.stem}")
        # Load the mask image
        mask = np.array(Image.open(mask_path))

        # Convert the mask to grayscale and find contours
        gray_mask = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        x, y, w, h = find_bounding_box(gray_mask > 0)

        # Crop the mask using the bounding box
        cropped_mask = gray_mask[y:y+h, x:x+w]
        # Convert cropped mask to a boolean mask
        bool_mask = cropped_mask > 0

        # Load the corresponding image
        image = np.array(Image.open(image_path))

        # Crop the image using the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Apply the boolean mask to the cropped image (set background to zero)
        for i in range(3):  # Assuming RGB
            cropped_image[:,:,i] = cropped_image[:,:,i] * bool_mask

        # Convert the array back to an image
        final_image = Image.fromarray(cropped_image)

        final_image  = make_square_image(final_image, max(w, h))

        # Save the final image
        patch_path = mask_dir / f'no_bg_{mask_path.stem}.png'
        final_image.save(patch_path)
        print(f"Saved image with no background to {patch_path}")


if __name__ == '__main__':
    extract_patches_with_no_background()