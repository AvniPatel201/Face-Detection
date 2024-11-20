import cv2
import matplotlib.pyplot as plt
import os
import dlib
from deepface import DeepFace

# Initialize dlib face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # dlib predictor

# Emotion detection function using DeepFace
def detect_emotion(image, face_coordinates):
    try:
        face = image[face_coordinates[1]:face_coordinates[3], face_coordinates[0]:face_coordinates[2]]
        analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        return None

# Function to detect faces using Haar Cascades
def detect_faces_haar(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    return img_resized, faces

# Function to detect faces using dlib
def detect_faces_dlib(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    return img_resized, faces

# Function to detect features using SIFT/ORB (combined visualization)
def detect_faces_using_sift_orb(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()
    keypoints_sift, _ = sift.detectAndCompute(gray, None)
    keypoints_orb, _ = orb.detectAndCompute(gray, None)
    img_combined = cv2.drawKeypoints(img_resized, keypoints_sift, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_combined = cv2.drawKeypoints(img_combined, keypoints_orb, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_combined

# Combine all face detection methods
image_folder = "../images"  # Adjust this path to your 'images' folder
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_paths:
    print("No images found in the folder!")
else:
    for image_path in image_paths:
        img_haar, faces_haar = detect_faces_haar(image_path)
        img_dlib, faces_dlib = detect_faces_dlib(image_path)
        img_sift_orb = detect_faces_using_sift_orb(image_path)

        # Annotate Haar results
        for (x, y, w, h) in faces_haar:
            emotion = detect_emotion(img_haar, (x, y, x + w, y + h))
            cv2.rectangle(img_haar, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img_haar, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Annotate Dlib results
        for face in faces_dlib:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            emotion = detect_emotion(img_dlib, (x1, y1, x2, y2))
            cv2.rectangle(img_dlib, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_dlib, emotion, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Plot results for comparison
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))
        axs[0].imshow(cv2.cvtColor(img_haar, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Haar Cascade")
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(img_dlib, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Dlib HOG Detector")
        axs[1].axis('off')

        axs[2].imshow(cv2.cvtColor(img_sift_orb, cv2.COLOR_BGR2RGB))
        axs[2].set_title("SIFT & ORB Keypoints")
        axs[2].axis('off')

        plt.suptitle(f"Face Detection Comparisons - {os.path.basename(image_path)}", fontsize=16)
        plt.tight_layout()
        plt.show()
