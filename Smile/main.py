import cv2

# Load the face and smile classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Access the webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Function to detect faces and smiles
def detect_faces_and_smiles(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(35, 35))
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(vid, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Focus on the face region to detect smiles
        face_region_gray = gray_image[y:y+h, x:x+w]
        face_region_color = vid[y:y+h, x:x+w]

        # Detect smiles within the face region
        smiles = smile_classifier.detectMultiScale(face_region_gray, scaleFactor=1.8, minNeighbors=30, minSize=(20, 20))
        for (sx, sy, sw, sh) in smiles:
            # Draw a rectangle around each detected smile
            cv2.rectangle(face_region_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

# Main loop for real-time face and smile detection
while True:
    # Read frames from the video
    result, video_frame = video_capture.read()  
    if not result:
        break

    # Apply face and smile detection
    detect_faces_and_smiles(video_frame)

    # Display the processed frame
    cv2.imshow("Smile Detection", video_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
