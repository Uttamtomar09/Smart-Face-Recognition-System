
import cv2
import os

photo_path = "known_faces/Uttam.jpg"

if not os.path.exists(photo_path):
    print("Photo nahi mili! known_faces folder check karo.")
    exit()

face_cascade = cv2.CascadeClassifier(
    r"C:\Users\tomar\Desktop\Image_processing_project\haarcascade_frontalface_default.xml"
)

image = cv2.imread(photo_path)

if image is None:
    print("Photo read nahi ho paayi!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
)

if len(faces) == 0:
    print("Photo mein face detect nahi hua. Clear photo lo.")
    exit()

x, y, w, h = faces[0]
face = gray[y:y+h, x:x+w]
face = cv2.resize(face, (200, 200))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train([face], __import__("numpy").array([1]))
recognizer.write("face_model.yml")

print("Training complete! face_model.yml save ho gaya.")
