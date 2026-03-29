import cv2

# Step 1: Load the pre-trained 'Haar Cascade' model for frontal faces
# This XML file contains the mathematical patterns that represent a human face
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Step 2: Initialize the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Step 3: Capture the current frame from the webcam
    ret, img = webcam.read()
    
    # Safety check: if the camera fails to send an image, stop the loop
    if not ret:
        print("failed to grab image")
        break

    # Step 4: Convert the image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 5: Detect faces in the image
    # 1.5 is the Scale Factor (compensates for faces being closer/further away)
    # 4 is the minNeighbors (how many 'detections' must overlap to count as a face)
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)

    # Step 6: Draw a rectangle around every face found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Step 7: Display the live video feed with the rectangles drawn on it
    cv2.imshow("Face detection", img)

    # Step 8: Wait for 1 millisecond for the 'q' key (ASCII 113) to quit
    key = cv2.waitKey(1)
    if key == 113:
        break

# Step 9: Release the hardware and close the display window
webcam.release()
cv2.destroyAllWindows()