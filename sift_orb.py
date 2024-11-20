import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def detect_faces_using_sift_orb(image_path):
    # Load and resize the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (800, 600))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT and ORB detectors
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors using SIFT and ORB
    keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)

    # Draw keypoints on the image
    img_sift = cv2.drawKeypoints(img_resized, keypoints_sift, None)
    img_orb = cv2.drawKeypoints(img_resized, keypoints_orb, None)

    # Extract keypoint coordinates
    sift_points = np.array([kp.pt for kp in keypoints_sift])
    orb_points = np.array([kp.pt for kp in keypoints_orb])

    # Estimate the approximate location and size of face parts (eyes, mouth)
    face_center = np.mean(sift_points, axis=0)
    face_radius = np.ptp(sift_points, axis=0)  # Estimate size based on keypoint spread

    # Approximate eyes region (using keypoint density)
    eyes_region = np.array([
        [face_center[0] - face_radius[0] / 4, face_center[1] - face_radius[1] / 4],
        [face_center[0] + face_radius[0] / 4, face_center[1] - face_radius[1] / 4],
        [face_center[0] + face_radius[0] / 4, face_center[1] + face_radius[1] / 4],
        [face_center[0] - face_radius[0] / 4, face_center[1] + face_radius[1] / 4]
    ])

    # Draw the eyes region (approximation)
    cv2.polylines(img_resized, [np.int32(eyes_region)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Approximate mouth region
    mouth_region = np.array([
        [face_center[0] - face_radius[0] / 2, face_center[1] + face_radius[1] / 2],
        [face_center[0] + face_radius[0] / 2, face_center[1] + face_radius[1] / 2],
        [face_center[0] + face_radius[0] / 2, face_center[1] + face_radius[1]],
        [face_center[0] - face_radius[0] / 2, face_center[1] + face_radius[1]]
    ])

    # Draw the mouth region (approximation)
    cv2.polylines(img_resized, [np.int32(mouth_region)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Return the processed image
    return img_resized

# Get list of images from the 'images' folder
image_folder = "../images"  # Adjust this path to your 'images' folder
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if there are any images
if not image_paths:
    print("No images found in the folder!")
else:
    # Create a figure to display images
    fig, axes = plt.subplots(1, len(image_paths), figsize=(12, 6))

    # Process each image and display the results
    for i, image_path in enumerate(image_paths):
        detected_image = detect_faces_using_sift_orb(image_path)
        
        # Display the image with detected regions
        axes[i].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Detected face - {image_path}")
        axes[i].axis('off')

    # Adjust layout and show the images
    plt.tight_layout()
    plt.show()
