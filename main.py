from moviepy.editor import VideoFileClip
import numpy as np
import os
import matplotlib.pyplot as plt
import pytesseract
import cv2
import easyocr

SAVING_FRAMES_PER_SECOND = 10

pytesseract.pytesseract.tesseract_cmd = "c:/Program Files/Tesseract-OCR/tesseract.exe"    #'C:/OCR/Tesseract-OCR/tesseract.exe'
def textrecon(file_path):
    reader = easyocr.Reader()
    result = reader.readtext(file_path)

    return result
def idk():
    file_path = input("Enter a file path")
    textrecon(file_path=file_path)

def carplate_extract(image, carplate_haar_cascade):

    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y+15:y+h-10, x+15:x+w-20]

    return carplate_img

def open_img(img_path):
    carplate_img = cv2.imread(img_path)
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    #print('fff')
    #plt.imshow(carplate_img)
    #print('444')
    #plt.show()
    #print("555")
    return carplate_img


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    plt.axis('off')
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized_image


cap = cv2.VideoCapture(0)

def detect(img):
    idk()
    '''
    carplate_img_rgb = img #open_img(img_path='D:/git/python/carplate2/img2.png')
    #print('0')
    carplate_haar_cascade = cv2.CascadeClassifier('/home/danil/PycharmProjects/pythonProject14/venv/carplates/haarcascade_russian_plate_number.xml')
    # 'd:/git/python/carplate2/venv/Lib/site-packages/cv2/data/haarcascade_russian_plate_number.xml')
    #print('1')
    try:
        carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)
        carplate_extract_img = enlarge_img(carplate_extract_img, 150)
        plt.imshow(carplate_extract_img)
        # plt.show()
        #print('2')

        carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
        plt.axis('off')

        print('Номер авто: ', pytesseract.image_to_string(
            carplate_extract_img_gray,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') )
    except:
        pass
    '''
#detect(0)
#exit(0)
i = 0
while True:
    i = i + 1
    #print(i)
    if i % 500000 == 0:
        #print(">>>>>>>")
        ret, img = cap.read()
        cv2.imshow("camera", img)

        detect(img)

        if cv2.waitKey(10) == 27:
            break

cap.release()
cv2.destroyAllWindows()