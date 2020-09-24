# Code to extract all the faces recognised from a photo
# and store it in a folder with date and time of photo.

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from google.colab.patches import cv2_imshow
import cv2

image = Image.open("Image path")
image = image.convert('RGB')
pixels = asarray(image)
detector = MTCNN()
results = detector.detect_faces(pixels)
count = 0

for i in results:
    count += 1
    x1,y1,widht,height = i['box']
    # print(x1,y1,x1+widht,y1+height)
    image1 = pixels[y1:y1+height, x1:x1+widht]
    image1 = Image.fromarray(image1)
    image1 = image1.resize((160,160))
    image1 = asarray(image1)
    cv2.imwrite("Image writing path"+ str(count)+".jpg",image1)
    # image1 = Image.fromarray(image1)
    cv2_imshow(image1)