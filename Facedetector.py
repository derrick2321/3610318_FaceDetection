# Importing OpenCV
import cv2
import os
# Importing HARR CASCADE XML file
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# Capture video from webcam hence (0) or else add your own media file
face_detection = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frames = face_detection.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frames)

    # Waiting for q key for image to close, adding the break statement to end the face detection screen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
face_detection.release()
cv2.destroyAllWindows()