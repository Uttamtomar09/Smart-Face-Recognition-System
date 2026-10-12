
import cv2
import os

os.makedirs("known_faces", exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera open nahi hua!")
    exit()

print("SPACE = Photo save | Q = Exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera se frame nahi mila!")
        break

    cv2.imshow("Uttam - Capture Photo", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        path = os.path.abspath("known_faces/Uttam.jpg")
        cv2.imwrite(path, frame)
        print("Photo saved:", path)
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
