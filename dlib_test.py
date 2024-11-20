import cv2
import matplotlib.pyplot as plt
import os
import dlib

# Initialize the dlib face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Path to the shape predictor file

def detect_faces_dlib(image_path):
    # Load and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (800, 600))
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_detector(gray)

    # Report the number of faces detected
    num_faces = len(faces)
    if num_faces == 0:
        print(f"No faces detected in {image_path}!")
    else:
        print(f"Detected {num_faces} face(s) in {image_path}")
    
    # Loop through all detected faces
    for face in faces:
        # Draw a rectangle around the detected face
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Predict facial landmarks for the detected face
        landmarks = landmark_predictor(gray, face)
        
        # Draw circles around the landmarks (eyes, nose, mouth)
        for n in range(36, 48):  # Landmarks for eyes, nose, and mouth (index 36 to 47)
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # Return the image with detected faces and landmarks, along with the number of faces detected
    return img, num_faces

# Get the list of images from the 'images' folder
image_folder = "../images"  # Adjust this path to your 'images' folder
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if there are any images in the folder
if not image_paths:
    print("No images found in the folder!")
else:
    # Create a figure to display the images
    fig, axes = plt.subplots(1, len(image_paths), figsize=(12, 6))

    # Process each image and display the results
    for i, image_path in enumerate(image_paths):
        detected_image, num_faces = detect_faces_dlib(image_path)
        axes[i].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{num_faces} face(s) detected - {image_path}")
        axes[i].axis('off')

    # Adjust layout and display the images
    plt.tight_layout()
    plt.show()
