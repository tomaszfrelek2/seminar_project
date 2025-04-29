import cv2
import numpy as np
import random
import argparse
import os
from seminar_project.data_modification.make_crack import *

def generate_random_mud_splatter(image, num_splatters=15, opaque = False):
    h, w = image.shape[:2]
    mud_overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    for _ in range(num_splatters):
        center_x = np.random.randint(0, w)
        center_y = np.random.randint(0, h)
        num_points = np.random.randint(5, 12)
        radius = np.random.randint(20, 100)

        # Generate irregular splatter shape
        angles = np.linspace(0, 2 * np.pi, num_points)
        pts = []
        for angle in angles:
            r = radius + np.random.randint(-10, 20)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            pts.append([x, y])

        pts = np.array([pts], dtype=np.int32)
        
        # Draw filled brown shape with some transparency
        mud_color = (np.random.randint(20, 40), np.random.randint(90, 110), np.random.randint(130, 150), max(255 *int(opaque),np.random.randint(180, 255)))  # brown RGBA
        cv2.fillPoly(mud_overlay, pts, mud_color)

    # Slight blur to make it look more realistic
    mud_overlay = cv2.GaussianBlur(mud_overlay, (21, 21), 0)

    # Convert original image to BGRA
    if image.shape[2] == 3:
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        image_bgra = image.copy()

    # Alpha blend the mud overlay
    alpha_mud = mud_overlay[:, :, 3:] / 255.0
    blended = image_bgra[:, :, :3] * (1 - alpha_mud) + mud_overlay[:, :, :3] * alpha_mud
    result = blended.astype(np.uint8)

    return result


def overlay_random_crack(image, crack_img):
    # Load base and crack images
    base = image
    crack = crack_img

    if base is None or crack is None:
        raise ValueError("Could not load base or crack image.")
    print(crack)
    crack = cv2.resize(crack, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)

    

    # Convert base to BGRA if not already
    if base.shape[2] == 3:
        base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)

    # Extract crack RGB and alpha
    crack_rgb = crack[:, :, :3]
    crack_alpha = crack[:, :, 3] / 255.0

   

    # Alpha blend
    for c in range(3):
        base[:, :, c] = (base[:, :, c] * (1 - crack_alpha) + crack_rgb[:, :, c] * crack_alpha).astype(np.uint8)


    return base

def tint_image(image, tint=0, intensity=0.15):
    """
    Tint an image toward red, green, or blue.

    Parameters:
        image (np.ndarray): Input image (BGR or BGRA).
        tint (str): 'red', 'green', or 'blue'.
        intensity (float): Strength of the tint, between 0.0 and 1.0.

    Returns:
        np.ndarray: Tinted image.
    """
    
    # Clone image to avoid modifying original
    image = image.copy()

    

    # Add tint by blending toward full-color channel
    #print(tint)
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[:, :, int(tint)] = 255

    tinted = cv2.addWeighted(image, 1.0, overlay, intensity, 0)

    return tinted

def blackout_scanline(image, num = 5):
    """
    Black out a horizontal row (or multiple rows) in an image to simulate a broken scanline.

    Parameters:
        image (np.ndarray): Input image.
        row (int): Row index to black out. If None, a random row is chosen.
        thickness (int): Number of rows to black out (default is 1).

    Returns:
        np.ndarray: Image with blacked-out scanline.
    """
    img = image.copy()
    h = img.shape[0]
    rows = []
    # Random row if not specified
    for i in range(num):
            rows.append(np.random.randint(0, h - 1))

  

    # Black out the scanline(s)
    for i in range(num):
        img[rows[i]:min(h, rows[i] + 1), :] = 0

    return img

def simulate_chromatic_aberration(image, shift_pixels=5):
    """
    Simulates chromatic aberration by shifting red and blue channels in opposite directions.

    Parameters:
        image (np.ndarray): Input image in BGR format.
        shift_pixels (int): Number of pixels to shift red and blue channels. Default is 2.

    Returns:
        np.ndarray: Image with chromatic aberration effect.
    """
    # Split channels (OpenCV uses BGR)
    image_channels = cv2.split(image)
    if len(image_channels) == 3:
        b, g, r= image_channels
    elif len(image_channels) == 4:
        b, g, r, a= image_channels
    else:
        print("error, incompatable channel format")

    # Define affine transform matrix for shifting
    def shift_channel(channel, dx, dy):
        rows, cols = channel.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(channel, M, (cols, rows), borderMode=cv2.BORDER_REFLECT101)
        return shifted

    # Shift red right-down, blue left-up
    r_shifted = shift_channel(r, shift_pixels, shift_pixels)
    b_shifted = shift_channel(b, -shift_pixels, -shift_pixels)

    # Green stays the same
    result = cv2.merge([b_shifted, g, r_shifted])
    return result

def simulate_sensor_blooming(image, threshold=200, intensity=0.99, blur_size=25):
    """
    Simulates sensor blooming by bleeding bright areas into surrounding pixels.

    Parameters:
        image (np.ndarray): Input image (BGR).
        threshold (int): Brightness threshold (0-255) above which blooming is applied.
        intensity (float): Bloom blending intensity (0.0 to 1.0).
        blur_size (int): Size of the Gaussian blur kernel.

    Returns:
        np.ndarray: Image with sensor blooming effect.
    """
    image = image.copy()

    # Convert to grayscale to detect brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create bloom mask where brightness exceeds threshold
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Ensure mask has the same number of channels as the original image
    if len(image.shape) == 3 and image.shape[2] == 3:  # 3-channel image
        mask_color = cv2.merge([mask, mask, mask])  # Merge mask into 3 channels
    elif len(image.shape) == 3 and image.shape[2] == 4:  # 4-channel image (BGRA)
        mask_color = cv2.merge([mask, mask, mask, mask])  # Merge mask into 4 channels
    else:
        raise ValueError("Unsupported image format")

    # Extract bright regions
    bright_regions = cv2.bitwise_and(image, mask_color)

    # Blur bright regions to create blooming effect
    bloom = cv2.GaussianBlur(bright_regions, (blur_size, blur_size), 0)

    # Add the bloom back into the original image
    bloomed = cv2.addWeighted(image, 1.0, bloom, intensity, 0)

    return bloomed

def main():
    parser = argparse.ArgumentParser(description="Apply multiple camera defect effects to an image.")
    
    # Required
    parser.add_argument("--image_path", default = "/Users/tomaszfrelek/seminar_project/test_folder/1478019952686311006_jpg.rf.54e2d12dbabc46be3c78995b6eaf3fee.jpg", help="Path to the base image.")
    
    
    # Optional crack overlay
    parser.add_argument("--crack_path", help="Path to the crack PNG (with alpha).")
    
    # Mud
    parser.add_argument("--mud", action="store_true", help="Apply mud splatters.")
    parser.add_argument("--mud_splatters", type=int, default=15, help="Number of mud splatters.")
    parser.add_argument("--mud_opaque", type=bool, default=False, help="Is the mud opaque?")

    
    # Tint
    parser.add_argument("--tint", help="Apply color tint.")
    parser.add_argument("--tint_intensity", type=float, default=0.1, help="Tint intensity (0.0 - 1.0).")
    
    # Scanline
    parser.add_argument("--scanline", action="store_true", help="Apply a black scanline.")
    parser.add_argument("--number_scanlines", type=int, default=1, help="number of broken scanlines.")

    
    # Chromatic aberration
    parser.add_argument("--aberration", action="store_true", help="Apply chromatic aberration.")
    parser.add_argument("--aberration_shift", type=int, default=2, help="Shift amount for chromatic aberration.")
    
    # Output
    parser.add_argument("--output", default="output_defected.png", help="Filename for the output image.")
    
    # Sensor blooming
    parser.add_argument("--bloom", action="store_true", help="Apply sensor blooming effect.")
    parser.add_argument("--bloom_threshold", type=int, default=200, help="Brightness threshold for blooming (0–255).")
    parser.add_argument("--bloom_intensity", type=float, default=0.99, help="Intensity of bloom effect.")
    parser.add_argument("--bloom_blur", type=int, default=25, help="Blur kernel size for bloom effect (odd number).")

    args = parser.parse_args()

    # Load base image
    if not os.path.exists(args.image_path):
        print(f"Image file '{args.image_path}' not found.")
        return
    base = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if base is None:
        print("Could not load base image.")
        return

    # Crack overlay
    if args.crack_path:
        if not os.path.exists(args.crack_path):
            print(f"Crack file '{args.crack_path}' not found.")
            return
        crack = save_crack(args.crack_path, num_cracks=np.random.randint(7,16), thickness_range=(1, 2), branch_chance=random.uniform(.5, .75))

        base = overlay_random_crack(base, crack)

    # Mud splatters
    if args.mud:
        base = generate_random_mud_splatter(base, num_splatters=args.mud_splatters, opaque = args.mud_opaque)

    # Tint
    if args.tint:
        base = tint_image(base, tint=args.tint, intensity=args.tint_intensity)

    # Chromatic aberration
    if args.aberration:
        base = simulate_chromatic_aberration(base, shift_pixels=args.aberration_shift)

        # Sensor blooming
    if args.bloom:
        base = simulate_sensor_blooming(base, threshold=args.bloom_threshold, intensity=args.bloom_intensity, blur_size=args.bloom_blur)

    # Scanline
    if args.scanline:
        base = blackout_scanline(base, num = args.number_scanlines)
    

    # Save output
    cv2.imwrite(args.output, base)
    print(f"Saved result to '{args.output}'")


if __name__ == "__main__":
    main()