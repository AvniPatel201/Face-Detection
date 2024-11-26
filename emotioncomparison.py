import cv2
import numpy as np
from keras.models import load_model
from deepface import DeepFace
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load a custom trained DNN model for face detection (replace with your own model loading method)
def load_custom_dnn_model(model_path='face_detection_model.keras'):
    model = load_model(model_path)
    return model

# Perform face detection using the custom DNN model (you will need to adapt this method)
def dnn_face_detection(image, model):
    input_image = cv2.resize(image, (64, 64))  # Assuming your DNN works with 64x64 input
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    input_image = input_image.astype('float32') / 255.0  # Normalize input

    prediction = model.predict(input_image)
    
    # Simulate bounding box (for demonstration purposes, we'll assume face detected if prediction > 0.5)
    if prediction > 0.5:
        h, w = image.shape[:2]
        return [(int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))]  # Example bounding box (feel free to adjust)
    return []

# Perform face detection using Haar Cascade
def haar_face_detection(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Perform face detection using DLib
def dlib_face_detection(image):
    import dlib
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

# Perform face detection using SIFT and ORB
def sift_orb_face_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_sift, des_sift = sift.detectAndCompute(gray, None)

    orb = cv2.ORB_create()
    kp_orb, des_orb = orb.detectAndCompute(gray, None)

    if len(kp_sift) > 10 or len(kp_orb) > 10:
        h, w = image.shape[:2]
        return [(int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))]  # Example bounding box
    return []

# Perform emotion detection using DeepFace
def deepface_emotion(image, x1, y1, x2, y2):
    face_img = image[y1:y2, x1:x2]  # Crop the detected face
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "IDK"

# Main function to run face detection and display comparison
def main():
    model = load_custom_dnn_model('face_detection_model.keras')

    image_paths = ["clearpotrait.jpg", "groupphoto.jpg", "multiple.jpg", "shadowface.jpg", "sideportrait.jpg"]
    face_detection_methods = ['DNN', 'Haar', 'DLib', 'SIFT/ORB']
    
    fig = plt.figure(figsize=(20, 25))
    gs = GridSpec(5, 5, figure=fig)  # 5 rows, 5 columns (methods in first column, images in remaining columns)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        # Perform face detection
        dnn_faces = dnn_face_detection(image, model)
        haar_faces = haar_face_detection(image)
        dlib_faces = dlib_face_detection(image)
        sift_orb_faces = sift_orb_face_detection(image)

        # Create subplots for each method with better spacing
        ax_dnn = fig.add_subplot(gs[i, 1])  # Row i, Column 1 (DNN)
        ax_haar = fig.add_subplot(gs[i, 2])  # Row i, Column 2 (Haar)
        ax_dlib = fig.add_subplot(gs[i, 3])  # Row i, Column 3 (DLib)
        ax_sift_orb = fig.add_subplot(gs[i, 4])  # Row i, Column 4 (SIFT/ORB)

        # Process DNN detection
        dnn_image = image.copy()
        dnn_face_count = 0
        for (x1, y1, x2, y2) in dnn_faces:
            cv2.rectangle(dnn_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            emotion = deepface_emotion(image, x1, y1, x2, y2)
            cv2.putText(dnn_image, f"Emotion: {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            dnn_face_count += 1
        ax_dnn.imshow(cv2.cvtColor(dnn_image, cv2.COLOR_BGR2RGB))
        ax_dnn.set_title(f"DNN\nFaces: {dnn_face_count}", fontsize=10, pad=10)
        ax_dnn.axis('off')

        # Process Haar detection
        haar_image = image.copy()
        haar_face_count = 0
        for (x, y, w, h) in haar_faces:
            cv2.rectangle(haar_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            emotion = deepface_emotion(image, x, y, x + w, y + h)
            cv2.putText(haar_image, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            haar_face_count += 1
        ax_haar.imshow(cv2.cvtColor(haar_image, cv2.COLOR_BGR2RGB))
        ax_haar.set_title(f"Haar\nFaces: {haar_face_count}", fontsize=10, pad=10)
        ax_haar.axis('off')

        # Process DLib detection
        dlib_image = image.copy()
        dlib_face_count = 0
        for face in dlib_faces:
            x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
            cv2.rectangle(dlib_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            emotion = deepface_emotion(image, x1, y1, x2, y2)
            cv2.putText(dlib_image, f"Emotion: {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            dlib_face_count += 1
        ax_dlib.imshow(cv2.cvtColor(dlib_image, cv2.COLOR_BGR2RGB))
        ax_dlib.set_title(f"DLib\nFaces: {dlib_face_count}", fontsize=10, pad=10)
        ax_dlib.axis('off')

        # Process SIFT/ORB detection
        sift_orb_image = image.copy()
        sift_orb_face_count = 0
        for (x1, y1, x2, y2) in sift_orb_faces:
            cv2.rectangle(sift_orb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            emotion = deepface_emotion(image, x1, y1, x2, y2)
            cv2.putText(sift_orb_image, f"Emotion: {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            sift_orb_face_count += 1
        ax_sift_orb.imshow(cv2.cvtColor(sift_orb_image, cv2.COLOR_BGR2RGB))
        ax_sift_orb.set_title(f"SIFT/ORB\nFaces: {sift_orb_face_count}", fontsize=10, pad=10)
        ax_sift_orb.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
