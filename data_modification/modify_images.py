import cv2
import numpy as np
import random
import argparse
import os
from modify_image import *
from make_crack import *




def process_images_in_folder(input_folder, output_folder, crack_image = "/Users/tomaszfrelek/seminar_project/my_crack.png"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Only process image files (You can add more file formats)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff','avif')):
            print(f"Processing {filename}...")

            # Read the image
            image = cv2.imread(file_path)
            #rand_tuple = (np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2))
            
            rand_tuple = (np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2))
            if image is not None:
                # Apply the effect 
                if rand_tuple[0]:
                    image = generate_random_mud_splatter(image,np.random.randint(5,16))
                if rand_tuple[1]:
                    crack = save_crack(crack_image, num_cracks=np.random.randint(7,16), thickness_range=(1, 2), branch_chance=random.uniform(.5, .75))
                    image = overlay_random_crack(image,crack)
                if rand_tuple[2]:
                    image = tint_image(image,np.random.randint(0,3) , random.uniform(.05,.20))
                if rand_tuple[3]:
                    image = simulate_chromatic_aberration(image,np.random.randint(1,7))
                if rand_tuple[4]:
                    image = simulate_sensor_blooming(image, threshold=np.random.randint(180,240), intensity=random.uniform(.5, .99))
                if rand_tuple[5]:
                    image = blackout_scanline(image,np.random.randint(1,21))

                # Save the modified image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image)
                print(f"Saved modified image to {output_path}")
            else:
                print(f"Failed to read image {filename}")
        else:
            print(f"Skipping non-image file {filename}")

def main():
    parser = argparse.ArgumentParser(description="Modify all images in a folder")
    parser.add_argument("--input_folder",default = "/Users/tomaszfrelek/Downloads/seminar_project/partitioned_data/train/images",  help="Path to the folder containing input images")
    parser.add_argument("--output_folder",default = "/Users/tomaszfrelek/Downloads/seminar_project/partitioned_data/mud_train/images", help="Path to the folder to save modified images")
    args = parser.parse_args()


    # Call the function to process all images in the folder
    process_images_in_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()