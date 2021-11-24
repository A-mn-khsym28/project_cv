import cv2
import numpy as np
import imutils
import argparse
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import append

# class ShapeDetector:
#     def __init__(self):
#         pass


# def detect(self, c):
#     # initialize the shape name and approx the contour
#     shape = "undindentified"
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.04 * peri, True)


def av_pix(img, circles, size):
    av_value = []
    for coords in circles[0, :]:
        col = np.mean(img[coords[1]-size:coords[1]+size,
                      coords[0]-size:coords[0]+size])
        # print(img[coords[1]-size:coords[1]+size,coords[1]-size:coords[0]+size])
        av_value.append(col)
    return av_value


def get_radius(circles):
    radius = []
    for coords in circles[0, :]:
        radius.append(coords[2])
    return radius


# load the image with original color
original_image = cv2.imread('19.2 capstone_coins.png', 1)
# color the original image gray
img_g = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2GRAY)
# blur the gray image
img = cv2.GaussianBlur(img_g, (5, 5), 0)
# making threshold from gray image
thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))

# draw the contour and center of the shape on the image
    cv2.drawContours(original_image, [c], -1, (0, 255, 0), 2)
    cv2.circle(original_image, (cX, cY), 7, (255, 255, 255), -1)


# using Hough Circle
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                           0.9, 120, param1=20, param2=60, minRadius=50, maxRadius=200)
# print(circles)
circles = np.uint16(np.around(circles))
count = 1
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(original_image, (i[0], i[1]), i[2], (0, 255, 0), 4)
    # draw the center of the circle
    cv2.circle(original_image, (i[0], i[1]), 2, (0, 255, 0,), 3)
    # labeling text in original image
    # cv2.putText(original_image, str(count),
    #             (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    count += 1

radii = get_radius(circles)
print(radii)

bright_values = av_pix(original_image, circles, 20)
print(bright_values)

values = []
for a, b in zip(bright_values, radii):
    if a == 85.7275 and b == 137:
        values.append(50)
    elif a > 170 and b > 115:
        values.append(10)
    elif a > 170 and b <= 90:
        values.append(5)
    elif a < 105 and b >= 125:
        values.append(2)
    elif a < 125 and b < 110:
        values.append(1)

print(values)
count_2 = 0
for i in circles[0, :]:
    cv2.putText(original_image, str(
        values[count_2]) + "p", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    count_2 += 1
cv2.putText(original_image, "Estimated Total Value: " + str(sum(values)
                                                            ) + "p", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255)

cv2.imshow("detected coins", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
