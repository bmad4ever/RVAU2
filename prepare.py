# TODO LIST
# save drawn image and annotations to files
# write text on image, need an algorithm to properly scale !!!
# binary semi automatic selection -> GRAB_CUT
#   https://www.youtube.com/watch?v=mUpk3zgDAR0
#   https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html
# gather keyPoints/descriptors and remove ones outside selection
# save/load keyPoints/descriptors
# select a file with a GUI
#   https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
# revert action [+Z]
# draw commands list

import cv2
import numpy as np

# region MAIN VARIABLES

# annotations are position,text tuples that are saved to a file after editing
annotations = []

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1

text_tag_background_color = (255, 255, 255, 255)
text_tag_text_color = (0, 0, 0, 255)
grab_cut_color_foreground = (255, 0, 0, 0)
grab_cut_color_background = (0, 0, 255, 0)

# reference image
src_img = cv2.imread('images/poster_test.jpg', cv2.IMREAD_UNCHANGED)
src_width, src_height, channels = tuple(src_img.shape)
if channels is not None and channels < 4:
    b, g, r = cv2.split(src_img)
    src_img = cv2.merge((b, g, r, np.ones((src_width, src_height, 1), np.uint8) * 255))
# src_img = src_img.asType(float)
# draw image
draw_img = np.zeros((src_width, src_height, 4), np.uint8)
# draw_img = draw_img.asType(float)
# text image
annotations_img = np.zeros((src_width, src_height, 4), np.uint8)
# annotations_img = annotations_img.asType(float)

# endregion MAIN VARIABLES

# region AUX METHODS


def overlay_type_additive():
    r = cv2.add(src_img, draw_img)
    return cv2.add(r, annotations_img)

# NOT WORKING  def OVERLAY_TYPE_MUL_IT():
#    r = src_img.copy()
#    overlayImages(r,draw_img)
#    overlayImages(r,annotations_img)


def overlay_type_cv_blend():
    r = alpha_blend(src_img, draw_img)
    return alpha_blend(r, annotations_img)

# NOT WORKING def overlayImages(src,overlay):
#    for i in range(src_width):
#        for j in range(src_height):
#            alpha = float(overlay[i][j][3]/255)
#            src[i][j] = alpha*overlay[i][j]+(1-alpha)*src[i][j]
#            src[i][j][3] = 255


def alpha_blend(background, foreground):
    alpha = cv2.split(foreground)[3]
    alpha = cv2.merge((alpha, alpha, alpha, alpha)).astype(float)/255
    background = background/255
    foreground = foreground/255
    background = cv2.multiply(1.0 - alpha, background)
    return cv2.add(foreground, background)


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, annotations

    if event == cv2.EVENT_RBUTTONDOWN:
        print('please add string')
        text = input()
        new_annotation = [(x, y), text]
        print("debug - annotation added" + str(new_annotation))
        annotations = annotations.append(new_annotation)
        # TODO
        # cv2.rectangle(img, (x-, y-), (x+, y+), (0, 255, 0), -1)
        # cv2.putText(annotations_img, time_string, (0, sizes[1] + black_margin - 5), cv2.FONT_HERSHEY_TRIPLEX, 2.5,
        #            (255, 255, 255), 2, cv2.LINE_AA)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(draw_img, (ix, iy), (x, y), (0, 255, 0, 255), -1)
            else:
                cv2.circle(draw_img, (x, y), 5, (0, 0, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(draw_img, (ix, iy), (x, y), (0, 255, 0, 255), -1)
        else:
            cv2.circle(draw_img, (x, y), 5, (0, 0, 255, 255), -1)


def save():
    global annotations, draw_img
    # TODO implement

# endregion AUX METHODS

# region PROGRAM DEFINITIONS/CONFIG

# assign the method. name should start with OVERLAY_TYPE.
# OVERLAY_TYPE_CV_BLEND needs that images are cast to floats. it will occlude layers below and maker program slow =(
# OVERLAY_TYPE_ADDITIVE uses the uInt8 default format. it will not occlude layers below


LAYERS_OVERLAY_METHOD = overlay_type_additive

MAIN_WINDOW_NAME = 'Prepare Program'

# endregion PROGRAM DEFINITIONS/CONFIG

# region Main Program Code

cv2.namedWindow(MAIN_WINDOW_NAME)
cv2.setMouseCallback(MAIN_WINDOW_NAME, draw_circle)

while(1):
    img2show = LAYERS_OVERLAY_METHOD()
    cv2.imshow(MAIN_WINDOW_NAME, img2show)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):  # CHANGES DRAW MODE
        mode = not mode
    if k == ord('s'):  # TODO SAVE
        break
    elif k == 27:
        break

cv2.destroyAllWindows()

# endregion Main Program Code
