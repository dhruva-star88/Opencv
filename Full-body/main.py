import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)
# Access the webcam
video_capture = cv2.VideoCapture(0)
# Set the resolution higher for better full body detection
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Function identify faces in Cam and draw the bounding box around it
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.05, 3, minSize=(100, 100))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

# Creating a Loop for Real-Time Face Detection
while True:
    # read frames from the video
    result, video_frame = video_capture.read()  
    if result is False:
        # terminate the loop if the frame is not read successfully
        break  
# apply the function we created to the video frame
    faces = detect_bounding_box(
        video_frame
    )  
# display the processed frame in a window named "My Face Detection Project"
    cv2.imshow(
        "My Face Detection Project", video_frame
    )  

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()