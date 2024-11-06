import cv2

stopsign_classifier = cv2.CascadeClassifier(
 "D:\Opencv\object\stop_data.xml"
)
# Access the webcam
video_capture = cv2.VideoCapture(0)

# Function identify faces in Cam and draw the bounding box around it
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    sign = stopsign_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in sign:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return sign

# Creating a Loop for Real-Time Face Detection
while True:
    # read frames from the video
    result, video_frame = video_capture.read()  
    if result is False:
        # terminate the loop if the frame is not read successfully
        break  
# apply the function we created to the video frame
    sign = detect_bounding_box(
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