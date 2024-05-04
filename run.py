import numpy as np
from PIL import ImageGrab
import cv2
import matplotlib.pyplot as plt


def draw_lines(img, lines):
    try:
        for line in lines:
            coordinates = line[0]
            cv2.line(img, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), [255,0,0], 5)
    except:
        pass

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=150, threshold2=350)
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    vertices = np.array([[0, 630], [0, 450], [400,250], [800,450], [600, 630]])
    processed_img = roi(processed_img, vertices)


    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 180)
    draw_lines(processed_img, lines)

    return processed_img

while(True):
    screen = np.array(ImageGrab.grab(bbox=(0, 0, 800, 630)))
    new_screen = process_img(screen)
    cv2.imshow('game', new_screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break