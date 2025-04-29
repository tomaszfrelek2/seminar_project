import cv2
import numpy as np
import random
import argparse
def generate_crack_texture(width=512, height=512, num_cracks=5, thickness_range=(1, 2), branch_chance=0.3, blur_ksize=3):
    """
    Generates a transparent PNG image with random crack-like lines and optional blur.

    Parameters:
        width (int): Width of the output image.
        height (int): Height of the output image.
        num_cracks (int): Number of main cracks.
        thickness_range (tuple): Min and max line thickness.
        branch_chance (float): Probability of spawning a branch from a crack segment.
        blur_ksize (int): Kernel size for Gaussian blur (must be odd, >= 3).

    Returns:
        np.ndarray: RGBA image with generated cracks.
    """
    # Create transparent background
    crack_img = np.zeros((height, width, 4), dtype=np.uint8)
    branching_chance = branch_chance
    def draw_jagged_line(start, direction, length, thickness,branching_chance):
        x, y = start
        for _ in range(length):
            dx = direction[0] + random.randint(-1, 1)
            dy = direction[1] + random.randint(-1, 1)
            next_x = int(np.clip(x + dx, 0, width - 1))
            next_y = int(np.clip(y + dy, 0, height - 1))
            cv2.line(crack_img, (x, y), (next_x, next_y), (255, 255, 255, 255), thickness)
            
            if random.random() < branching_chance:
                # Create a small branch crack
                branch_dir = (random.choice([-1, 1]) * dy, random.choice([-1, 1]) * dx)
                branching_chance -= .1
                draw_jagged_line((x, y), branch_dir, length // 4, max(1, thickness - 1),branching_chance)
                


            x, y = next_x, next_y

    for _ in range(num_cracks):
        branching_chance = branch_chance
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        direction = (random.choice([-1, 1]) * random.randint(1, 3), random.choice([-1, 1]) * random.randint(1, 3))
        length = random.randint(40, 100)
        thickness = random.randint(*thickness_range)
        draw_jagged_line((start_x, start_y), direction, length, thickness,branching_chance)

    # Apply Gaussian blur to soften the cracks (only on RGB channels)
    rgb = crack_img[:, :, :3]
    alpha = crack_img[:, :, 3]
    blurred_rgb = cv2.GaussianBlur(rgb, (blur_ksize, blur_ksize), 0)
    crack_img = cv2.merge([blurred_rgb, alpha])

    return crack_img

def save_crack(output_file = "/Users/tomaszfrelek/seminar_project/my_crack.png", width=512, height=512, num_cracks=5, thickness_range=(1, 2), branch_chance=0.3, blur_ksize=3):
    crack = generate_crack_texture(width=512, height=512, num_cracks=num_cracks, thickness_range=thickness_range, branch_chance=branch_chance, blur_ksize=3)
    cv2.imwrite(output_file, crack)
    return crack




def main():
    parser = argparse.ArgumentParser(description="Generate a crack texture with transparent background.")
    parser.add_argument("--width", type=int, default=512, help="Width of the output image.")
    parser.add_argument("--height", type=int, default=512, help="Height of the output image.")
    parser.add_argument("--num_cracks", type=int, default=5, help="Number of main cracks.")
    parser.add_argument("--thickness_min", type=int, default=1, help="Minimum crack thickness.")
    parser.add_argument("--thickness_max", type=int, default=2, help="Maximum crack thickness.")
    parser.add_argument("--branch_chance", type=float, default=0.3, help="Chance for cracks to branch.")
    parser.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel size (must be odd).")
    parser.add_argument("--output", type=str, default="my_crack.png", help="Output PNG filename.")
    args = parser.parse_args()

    # Ensure blur size is odd
    if args.blur % 2 == 0:
        args.blur += 1

    crack = generate_crack_texture(
        width=args.width,
        height=args.height,
        num_cracks=args.num_cracks,
        thickness_range=(args.thickness_min, args.thickness_max),
        branch_chance=args.branch_chance,
        blur_ksize=args.blur
    )

    cv2.imwrite(args.output, crack)
    print(f"Crack texture saved as '{args.output}'")

if __name__ == "__main__":
    main()
