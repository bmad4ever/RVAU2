import cv2
import openfile as of

filename = of.get_file()
image = cv2.imread(filename)


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 63, (0, 0, 255), -1)


window_name = "window name"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_circle)

while (1):
    cv2.imshow(window_name, image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

