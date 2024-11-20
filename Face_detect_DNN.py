import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def load_and_preprocess_data(base_path, img_size=(64, 64)):
    images = []
    labels = []
    
    # Convert string path to Path object
    base_path = Path(base_path)
    
    # Process both train and val directories
    for split in ['train', 'val']:
        images_path = base_path / 'images' / split
        labels_path = base_path / 'labels' / split
        
        print(f"Processing {split} data...")
        
        # Iterate through all images
        for img_path in images_path.glob('*.jpg'):  # Adjust extension if needed
            try:
                # Load and preprocess image
                img = Image.open(img_path)
                
                # Convert grayscale to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(img_size)
                
                # Convert to numpy array and ensure shape is (64, 64, 3)
                img_array = np.array(img, dtype=np.float32)
                
                # Normalize to [0,1]
                img_array = img_array / 255.0
                
                # Verify shape
                if img_array.shape != (img_size[0], img_size[1], 3):
                    print(f"Skipping {img_path} due to incorrect shape: {img_array.shape}")
                    continue
                
                # Load corresponding label file
                label_file = labels_path / (img_path.stem + '.txt')
                
                if label_file.exists():
                    images.append(img_array)
                    labels.append(1)
                    
                    # Add negative samples
                    with open(label_file, 'r') as f:
                        yolo_boxes = [list(map(float, line.strip().split()[1:])) for line in f]
                    
                    neg_img = generate_negative_sample(img_array, yolo_boxes)
                    if neg_img is not None:
                        images.append(neg_img)
                        labels.append(0)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
    
    # Convert to numpy arrays with explicit shapes
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    print(f"Final dataset shapes - X: {X.shape}, y: {y.shape}")
    return X, y

def generate_negative_sample(img_array, yolo_boxes, iou_threshold=0.3):
    """Generate a negative sample by taking a random crop that doesn't overlap significantly with faces"""
    h, w = img_array.shape[:2]
    crop_size = min(h, w) // 2
    
    for _ in range(10):  # Try 10 times to find a suitable crop
        x = np.random.randint(0, w - crop_size)
        y = np.random.randint(0, h - crop_size)
        
        # Convert crop coordinates to YOLO format
        crop_box = [(x + crop_size/2)/w, (y + crop_size/2)/h, crop_size/w, crop_size/h]
        
        # Check overlap with all face boxes
        overlap = False
        for box in yolo_boxes:
            if calculate_iou(crop_box, box) > iou_threshold:
                overlap = True
                break
        
        if not overlap:
            # Crop and resize
            crop = img_array[y:y+crop_size, x:x+crop_size]
            return cv2.resize(crop, (64, 64))
    
    return None

def calculate_iou(box1, box2):
    """Calculate IoU between two YOLO format boxes"""
    # Convert YOLO format to corners
    def yolo_to_corners(box):
        cx, cy, w, h = box
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]
    
    box1 = yolo_to_corners(box1)
    box2 = yolo_to_corners(box2)
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def create_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def main():
    # Specify the base path to your dataset
    base_path = "/Users/avnipatel/Documents/Machine Vision/Final Project/archive-2"
    
    print("Loading and preprocessing data...")
    X_train, y_train = load_and_preprocess_data(base_path)
    
    if len(X_train) == 0:
        print("No valid images were loaded. Please check the dataset.")
        return
        
    print(f"Dataset loaded. Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Create and train model
    model = create_model(input_shape=(64, 64, 3))
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save the model
    model.save('face_detection_model.keras')
    print("Model saved as 'face_detection_model.keras'")

if __name__ == "__main__":
    main()
