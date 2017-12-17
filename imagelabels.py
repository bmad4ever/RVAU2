import cv2
import numpy as np


def add_label(image, x, y, text):
    size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=10, thickness=1)
    fill_color = (255, 255, 255)
    border_color = (0, 0, 0)
    margin = 50
    pointer = margin * 2
    cv2.rectangle(image,
                  (int(x - size[0][0] / 2 - margin * 2), int(y - size[0][1] - margin * 2 - pointer)),
                  (int(x + size[0][0] / 2 + margin * 2), y - pointer),
                  fill_color,
                  -1)
    cv2.rectangle(image,
                  (int(x - size[0][0] / 2 - margin * 2), int(y - size[0][1] - margin * 2 - pointer)),
                  (int(x + size[0][0] / 2 + margin * 2), int(y - pointer)),
                  border_color,
                  10)
    cv2.line(image, (x + pointer, y - pointer), (x - pointer, y - pointer), fill_color, 10)
    triangle = np.array([[x + pointer, y - pointer], [x - pointer, y - pointer], [x, y]])
    cv2.fillConvexPoly(image, triangle, fill_color)
    cv2.line(image, (x + pointer, y - pointer), (x, y), border_color, 10)
    cv2.line(image, (x - pointer, y - pointer), (x, y), border_color, 10)
    cv2.circle(image, (x, y), 10, (0, 0, 0))

    cv2.putText(img=image,
                text=text,
                org=(int(x - size[0][0] / 2), y - margin - pointer),
                color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=10,
                thickness=10)
