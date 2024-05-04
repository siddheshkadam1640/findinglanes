import numpy as np
from PIL import ImageGrab
import cv2
import matplotlib.pyplot as plt
import time


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    vertices = np.array([[0, 630], [0, 450], [400,250], [800,450], [600, 630]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        try:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2,y2), (255, 0, 0), 10)

        except:
            pass
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]


            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    except:
        pass

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

    y= time.time() + 5

    if time.time() > y:
        left_fit.clear()
        right_fit.clear()

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1*(2/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    except:
        pass

# while(True):
#     screen = np.array(ImageGrab.grab(bbox=(0, 0, 800, 630)))
#     new_screen = process_img(screen)
#     cv2.imshow('game', new_screen)
#     lines = cv2.HoughLinesP(new_screen, 1, np.pi / 180, 180, np.array([]), 100, 180)
#     averaged_lines = average_slope_intercept(new_screen, lines)
#     line_image = display_lines(new_screen, averaged_lines)
#     display_lines(new_screen, lines)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
#


while True:
    screen = np.array(ImageGrab.grab(bbox=(0, 0, 800, 630)))
    canny_image = canny(screen)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 180, np.array([]), 100, 400)
    averaged_lines = average_slope_intercept(cropped_image, lines)
    line_image = display_lines(screen, averaged_lines)
    final_image = cv2.addWeighted(screen, 0.8, line_image, 1, 1)
    cv2.imshow('lanes', final_image)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()

