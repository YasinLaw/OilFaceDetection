import cv2
import sys
import datetime
import uuid

faceCascade = cv2.CascadeClassifier("./env/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")

video_capture = cv2.VideoCapture(0)

cameraNumber = 0

count = 0
countNew = 0
captureSpan = datetime.timedelta(seconds=3)
countChangeTime = datetime.datetime.now()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if countNew != len(faces):
        countChangeTime = datetime.datetime.now()
        countNew = len(faces)
    else:
        if datetime.datetime.now() - countChangeTime > captureSpan:
            if count > countNew:
                count = countNew
                print("target exit: " + str(datetime.datetime.now()))
            elif count < countNew:
                count = countNew
                cv2.imwrite("./capture/camera" + str(cameraNumber) + "-" + str(datetime.datetime.now()) + ".png", frame)
                print("capture!")
        else:
            if count != countNew:
                print("target insight:" + str(datetime.datetime.now() - countChangeTime))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
