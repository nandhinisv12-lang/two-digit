import os
import random
import cv2
import numpy as np

# ----------------------------
# Paths
    # ----------------------------
input_dataset = r"C:\Users\nandh\OneDrive\Desktop\two digit\double\merge_dataset\merged_train"
           # Your single-digit dataset
output_dataset = "double_digit_dataset"

os.makedirs(output_dataset, exist_ok=True)

# Create folders 00 to 99
for i in range(100):
    os.makedirs(os.path.join(output_dataset, f"{i:02d}"), exist_ok=True)

# ----------------------------
# Generate Images
# ----------------------------

images_per_class = 100   # Number of images to generate for each double-digit class

for first_digit in range(10):
    for second_digit in range(10):

        folder_name = f"{first_digit}{second_digit}"

        first_folder = os.path.join(input_dataset, str(first_digit))
        second_folder = os.path.join(input_dataset, str(second_digit))
        print("Input dataset:", input_dataset)
        print("Trying to open:", first_folder)
        print("Exists:", os.path.exists(first_folder))

        first_images = os.listdir(first_folder)
        first_images = os.listdir(first_folder)
        second_images = os.listdir(second_folder)

        for count in range(images_per_class):

            img1_name = random.choice(first_images)
            img2_name = random.choice(second_images)

            img1 = cv2.imread(os.path.join(first_folder, img1_name),
                              cv2.IMREAD_GRAYSCALE)

            img2 = cv2.imread(os.path.join(second_folder, img2_name),
                              cv2.IMREAD_GRAYSCALE)

            # Resize if necessary
            img1 = cv2.resize(img1, (64,64))
            img2 = cv2.resize(img2, (64,64))

            # Combine horizontally
            combined = np.hstack((img1, img2))

            save_path = os.path.join(
                output_dataset,
                folder_name,
                f"{folder_name}_{count}.png"
            )

            cv2.imwrite(save_path, combined)

print("Double-digit dataset created successfully!")