import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def detect_faces_using_sift_orb(image_path):
    """Detect faces using SIFT and ORB and draw regions on the image."""
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
    sift_points = np.array([kp.pt for kp in keypoints_sift]) if keypoints_sift else np.array([])
    orb_points = np.array([kp.pt for kp in keypoints_orb]) if keypoints_orb else np.array([])

    if len(sift_points) > 0:
        # Estimate the approximate location and size of face parts (eyes, mouth)
        face_center = np.mean(sift_points, axis=0)
        face_radius = np.ptp(sift_points, axis=0)  # Estimate size based on keypoint spread

        # Approximate eyes region
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

    return img_resized

def measure_performance(images, batch_sizes=[1, 8, 16, 32]):
    performance_data = {'SIFT_ORB': {}}

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        runtimes = []

        # Get subset of images for current batch size
        batch_images = images[:batch_size]

        # Measure multiple predictions
        for _ in range(10):  # Reduced to 10 iterations for speed
            start_time = time.time()
            for image_path in batch_images:
                detect_faces_using_sift_orb(image_path)
            end_time = time.time()
            runtimes.append(end_time - start_time)

        performance_data['SIFT_ORB'][batch_size] = runtimes

    return performance_data

def plot_performance(performance_data):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot performance
    ax.boxplot(performance_data['SIFT_ORB'].values())
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('SIFT + ORB Runtime by Batch Size')
    ax.set_xticks(range(1, len(performance_data['SIFT_ORB']) + 1))
    ax.set_xticklabels(performance_data['SIFT_ORB'].keys())
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('sift_orb_combined_runtime_comparison.png')
    plt.show()

    # Print statistics
    print("\nSIFT + ORB Statistics:")
    for batch_size, runtimes in performance_data['SIFT_ORB'].items():
        print(f"\nBatch size {batch_size}:")
        print(f"Average runtime: {np.mean(runtimes):.4f} seconds")
        print(f"Standard deviation: {np.std(runtimes):.4f} seconds")

def main():
    # Load your data
    base_path = "archive"
    image_folder = os.path.join(base_path, "images", "train")

    if not os.path.exists(image_folder):
        print(f"Directory not found: {image_folder}")
        return

    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(image_paths)} images")

    if not image_paths:
        print("No images found! Please check the image directory.")
        return

    # Measure and plot performance
    performance_data = measure_performance(image_paths)
    plot_performance(performance_data)

if __name__ == "__main__":
    main()
