import matplotlib as plt
import cv2

def image_resize(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC) 

for i in range(6):
    file_name = "image" + str(i+1) + ".jpg"
    img = image_resize(cv2.imread(file_name), 200, 200)
    cv2.imwrite(file_name, img)