import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def load_images(base_path, num_images=50):
    """Load and preprocess images for testing"""
    image_folder = os.path.join(base_path, "images", "train")
    
    if not os.path.exists(image_folder):
        print(f"Directory not found: {image_folder}")
        return []
    
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_images]
    
    print(f"Found {len(image_paths)} images")
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (800, 600))
            images.append(img)
        else:
            print(f"Failed to load image: {path}")
    
    return images

def process_image_sift_orb(img, sift, orb):
    """Process a single image with SIFT and ORB"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and descriptors
    keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
    
    # Extract keypoint coordinates
    sift_points = np.array([kp.pt for kp in keypoints_sift]) if keypoints_sift else np.array([])
    
    if len(sift_points) > 0:
        # Estimate face regions (simplified for performance measurement)
        face_center = np.mean(sift_points, axis=0)
        face_radius = np.ptp(sift_points, axis=0)
    
    return len(keypoints_sift), len(keypoints_orb)

def measure_performance(images, batch_sizes=[1, 8, 16, 32]):
    performance_data = {'SIFT': {}, 'ORB': {}}
    
    # Initialize SIFT and ORB detectors
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        sift_runtimes = []
        orb_runtimes = []
        
        # Get subset of images for current batch size
        batch_images = images[:batch_size]
        
        # Warm-up run
        for img in batch_images:
            process_image_sift_orb(img, sift, orb)
        
        # Measure multiple predictions
        for _ in range(50):  # 50 iterations for each batch size
            # Measure SIFT
            start_time = time.time()
            for img in batch_images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
            end_time = time.time()
            sift_runtimes.append(end_time - start_time)
            
            # Measure ORB
            start_time = time.time()
            for img in batch_images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
            end_time = time.time()
            orb_runtimes.append(end_time - start_time)
            
        performance_data['SIFT'][batch_size] = sift_runtimes
        performance_data['ORB'][batch_size] = orb_runtimes
    
    return performance_data

def plot_performance(performance_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot SIFT performance
    ax1.boxplot(performance_data['SIFT'].values())
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('SIFT Runtime by Batch Size')
    ax1.set_xticks(range(1, len(performance_data['SIFT']) + 1))
    ax1.set_xticklabels(performance_data['SIFT'].keys())
    ax1.grid(True)
    
    # Plot ORB performance
    ax2.boxplot(performance_data['ORB'].values())
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('ORB Runtime by Batch Size')
    ax2.set_xticks(range(1, len(performance_data['ORB']) + 1))
    ax2.set_xticklabels(performance_data['ORB'].keys())
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('sift_orb_runtime_comparison.png')
    plt.show()
    
    # Print statistics
    for method in ['SIFT', 'ORB']:
        print(f"\n{method} Statistics:")
        for batch_size, runtimes in performance_data[method].items():
            print(f"\nBatch size {batch_size}:")
            print(f"Average runtime: {np.mean(runtimes):.4f} seconds")
            print(f"Standard deviation: {np.std(runtimes):.4f} seconds")

def main():
    # Load your data
    base_path = "/Users/avnipatel/Documents/Machine Vision/Final Project/archive-2"
    images = load_images(base_path)
    
    if not images:
        print("No images found! Please check the image directory.")
        return
    
    # Measure and plot performance
    performance_data = measure_performance(images)
    plot_performance(performance_data)

if __name__ == "__main__":
    main() 