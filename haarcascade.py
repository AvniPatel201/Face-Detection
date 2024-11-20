import cv2
import matplotlib.pyplot as plt
import os

def detect_faces_haar(image_path):
    # Load the Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (800, 600))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    num_faces = len(faces)
    if num_faces == 0:
        print(f"No faces detected in {image_path}!")
    else:
        print(f"Detected {num_faces} face(s) in {image_path}")

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img, num_faces

# Get list of images from the 'images' folder
image_folder = "../images"  
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if there are any images
if not image_paths:
    print("No images found in the folder!")
else:
    # Create a figure to display the images
    fig, axes = plt.subplots(1, len(image_paths), figsize=(12, 6))

    # Process each image and display results
    for i, image_path in enumerate(image_paths):
        detected_image, num_faces = detect_faces_haar(image_path)
        axes[i].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{num_faces} face(s) detected - {image_path}")
        axes[i].axis('off')

    # Adjust layout and show the images
    plt.tight_layout()
    plt.show()
