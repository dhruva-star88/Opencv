import cv2
from matplotlib import pyplot as plt
  
# Opening image
img = cv2.imread("D:\Opencv\object\stop.jpg")
  
# OpenCV opens images as BRG but we want it as RGB We'll also need a grayscale version
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  
# Use minSize because for not bothering with extra-small dots that would look like STOP signs
stop_data = cv2.CascadeClassifier('D:\Opencv\object\stop_data.xml')
  
found = stop_data.detectMultiScale(img_gray, 
                                   minSize =(20, 20))
  
# Don't do anything if there's no sign (no. of objects(stop sign-boards) found)
amount_found = len(found)
  
if amount_found != 0:
            plt.figure(figsize=(10, 10))
            for i, (x, y, width, height) in enumerate(found, start=1):
                # Draw a rectangle around each detected sign
                cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 5)

                # Crop the detected region
                cropped_img = img_rgb[y:y + height, x:x + width]

                # Display the cropped region
                plt.subplot(1, amount_found, i)
                plt.imshow(cropped_img)
                plt.axis("off")
                plt.show()

            # Display the original image with rectangles around detected areas
            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.show()
