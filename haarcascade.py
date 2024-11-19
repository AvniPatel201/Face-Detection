import cv2
import matplotlib.pyplot as plt

def detect_faces_haar(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    num_faces = len(faces)
    if num_faces == 0:
        print(f"No faces detected in {image_path}!")
    else:
        print(f"Detected {num_faces} face(s) in {image_path}")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img, num_faces

image_paths = ['imageA.jpg', 'imgA.jpg', 'imgB.jpg']

fig, axes = plt.subplots(1, len(image_paths), figsize=(12, 6))

for i, image_path in enumerate(image_paths):
    detected_image, num_faces = detect_faces_haar(image_path)
    axes[i].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"{num_faces} face(s) detected - {image_path}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
