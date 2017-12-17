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
import math
import pickle
import auxfuncs

# region MAIN VARIABLES

# annotations are position,text tuples that are saved to a file after editing
annotations = [((12, 0), 'example1'), ((12, 0), 'example2')]

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1

#colors from 0 to 1 (1->255)
text_tag_background_color = (1, 1, 1, 1)
text_tag_text_color = (0, 0, 0, 1)
grab_cut_color_foreground = (1, 0, 0, 0)
grab_cut_color_background = (0, 0, 1, 0)
draw_rectangle_color = (0, 1, 0 , 1)
draw_circle_color = (1, 0, 0 , 1)

# reference image
src_img = cv2.imread('images/poster_test.jpg', cv2.IMREAD_UNCHANGED)
#src_width, src_height, channels = tuple(src_img.shape)

#reference image scaled down
#this will be used to avoid slow interaction and to save 'small' info images
TARGET_LENGTH = 400
MIN_SIZE = min(tuple(src_img.shape[1::-1]))
scale = TARGET_LENGTH / MIN_SIZE
src_img_scaled = cv2.resize(src_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
src_width_scaled, src_height_scaled, channels = tuple(src_img_scaled.shape)

if channels is not None and channels < 4:
    b, g, r = cv2.split(src_img_scaled)
    src_img_scaled = cv2.merge((b, g, r, np.ones((src_width_scaled, src_height_scaled, 1), np.uint8) * 255))
src_img_scaled = src_img_scaled.astype(float) / 255
#src_img_debug = src_img.copy().astype(float) / 255
# draw image
draw_img = np.zeros((src_width_scaled, src_height_scaled, 4), np.uint8)
draw_img = draw_img.astype(float)
# text image
annotations_img = np.zeros((src_width_scaled, src_height_scaled, 4), np.uint8)
annotations_img = annotations_img.astype(float)

# endregion MAIN VARIABLES

# region AUX METHODS


def overlay_type_additive():
    r = cv2.add(src_img_scaled, draw_img)
    return cv2.add(r, annotations_img)

'''
uses CPU ONLY. is 2 SLOW!!!
def OVERLAY_TYPE_MUL_IT():
    r = src_img.copy()
    overlayImages(r,draw_img)
    overlayImages(r,annotations_img)
    return r
'''

def overlay_type_cv_blend():
    r = alpha_blend(src_img_scaled, draw_img)
    return alpha_blend(r, annotations_img)

'''
uses CPU ONLY. is 2 SLOW!!!
def overlayImages(src,overlay):
    for i in range(src_width):
        for j in range(src_height):
            alpha = float(overlay[i][j][3]/255)
            src[i][j] = alpha*overlay[i][j]+(1-alpha)*src[i][j]
            src[i][j][3] = 255
'''

def alpha_blend(background, foreground):
    alpha = cv2.split(foreground)[3]#.astype(float)
    alpha = cv2.merge((alpha, alpha, alpha,alpha))
    #background = background/255
    #foreground = foreground/255
    background = cv2.multiply(1.0 - alpha, background)
    return cv2.add(foreground, background)


# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, mode, annotations

    if event == cv2.EVENT_RBUTTONDOWN:
        print('please add string')
        text = input()
        new_annotation = [(math.floor(x*scale), math.floor(y*scale)), text]
        print("debug - annotation added" + str(new_annotation))
        annotations = annotations.append(new_annotation)
        #DEBUG cv2.circle(src_img_debug, (math.floor(x*scale), math.floor(y*scale)), 5, draw_circle_color, -1)
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
                cv2.rectangle(draw_img, (ix, iy), (x, y), draw_rectangle_color, -1)
            else:
                cv2.circle(draw_img, (x, y), 5, draw_circle_color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(draw_img, (ix, iy), (x, y), draw_rectangle_color, -1)
        else:
            cv2.circle(draw_img, (x, y), 5, draw_circle_color, -1)


def save(img):
    #src_img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)

    with open('testdata.pkl', 'wb') as output:
        pickle.dump(auxfuncs.pickle_keypoints(kp1, des1), output)
        pickle.dump(annotations, output, pickle.HIGHEST_PROTOCOL)  # annotations position are relative to src_img size
        pickle.dump(draw_img, output, pickle.HIGHEST_PROTOCOL)  # visual annotations as user draws in a scaled down img
        pickle.dump(src_img, output, pickle.HIGHEST_PROTOCOL)  # only need size but saving the entire image can be useful for debuging

# endregion AUX METHODS

# region PROGRAM DEFINITIONS/CONFIG

# assign the method. name should start with OVERLAY_TYPE.
# OVERLAY_TYPE_CV_BLEND needs that images are cast to floats. it will occlude layers below and maker program slow =(
# OVERLAY_TYPE_ADDITIVE uses the uInt8 default format. it will not occlude layers below


LAYERS_OVERLAY_METHOD = overlay_type_cv_blend

MAIN_WINDOW_NAME = 'Prepare Program'

# endregion PROGRAM DEFINITIONS/CONFIG

# region Main Program Code

cv2.namedWindow(MAIN_WINDOW_NAME)
cv2.setMouseCallback(MAIN_WINDOW_NAME, mouse_callback)

while(1):
    img2show = LAYERS_OVERLAY_METHOD()
    cv2.imshow(MAIN_WINDOW_NAME, img2show)
    #cv2.imshow('debug scale', src_img_debug)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):  # CHANGES DRAW MODE
        mode = not mode
    if k == ord('s'):  # TODO SAVE
        save(src_img)
        break
    elif k == 27:
        break

cv2.destroyAllWindows()

# endregion Main Program Code
