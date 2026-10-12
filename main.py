

import cv2
import pyttsx3
import time

# Voice setup
voice = pyttsx3.init()
voice.setProperty("rate", 150)

def speak(message):
    voice.say(message)
    voice.runAndWait()

# Face detection model
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\tomar\Desktop\Image_processing_project\haarcascade_frontalface_default.xml"
)

# Face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened!")
    exit()

mode = 1
last_voice = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    display = frame.copy()

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(face)

        if label == 1 and confidence < 45:
            name = "Uttam"
            color = (0, 255, 0)

            if time.time() - last_voice > 10:
                speak("Welcome Uttam")
                last_voice = time.time()
        else:
            name = "Unknown Person"
            color = (0, 0, 255)

        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        cv2.putText(display, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Image modes
    if mode == 2:
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == 3:
        display = cv2.GaussianBlur(display, (15, 15), 0)
    elif mode == 4:
        display = cv2.Canny(gray, 100, 200)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    cv2.putText(display, "1: Original  2: Gray  3: Blur",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, "4: Edges  5: Face Recognition  S: Save  Q: Exit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("VisionX", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3
    elif key == ord('4'):
        mode = 4
    elif key == ord('5'):
        mode = 5
    elif key == ord('s'):
        cv2.imwrite("captured_image.jpg", frame)
        print("Image saved!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program closed!")
