import cv2
import matplotlib.pyplot as plt

imagePath = r'D:\Opencv\Facedetection\face.jpg'
img = cv2.imread(imagePath)
print(img.shape)

# convert to gray scale 
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

#Load the Classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Perform the Face Detection:
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

#Drawing a Bounding Box
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

#Displaying Image- Converting back to normal image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Use the Matplotlib library to display the image

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
