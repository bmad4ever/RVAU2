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
import openfile as of
from tkinter import *

# region MAIN VARIABLES

root = Tk()

kp1 = None  # SIFT keypoints
des1 = None  # SIFT descriptors

# annotations are position,text tuples that are saved to a file after editing
annotations = [((12, 0), 'example1'), ((12, 0), 'example2')]
scale = 0
img2show = None
img2show_mask = None

# Mask Variables
mask_drawing = None
mask_image = None
mask = None
mask_x, mask_y = -1, -1
mask_window_name = None
circle_radius = IntVar()
circle_radius.set(10)

# Drawing Variables
drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
font_scale = 1

#colors from 0 to 1 (1->255)
text_tag_background_color = (1, 1, 1, 1)
text_tag_text_color = (0, 0, 0, 1)
grab_cut_color_foreground = (1, 0, 0, 0)
grab_cut_color_background = (0, 0, 1, 0)
draw_rectangle_color = (0, 1, 0, 0.5)
draw_circle_color = (1, 0, 0, 0.5)
draw_circle_color_mask = (1, 1, 1, 1)

# reference image
src_img = cv2.imread('images/poster_test.jpg', cv2.IMREAD_UNCHANGED)
# src_width, src_height, channels = tuple(src_img.shape)


# reference image scaled down
# this will be used to avoid slow interaction and to save 'small' info images
def scale_image():
    global src_img_scaled, draw_img, annotations_img, scale, mask_image
    target_length = 400
    min_size = min(tuple(src_img.shape[1::-1]))
    scale = target_length / min_size
    src_img_scaled = cv2.resize(src_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    src_width_scaled, src_height_scaled, channels = tuple(src_img_scaled.shape)

    if channels is not None and channels < 4:
        b, g, r = cv2.split(src_img_scaled)
        src_img_scaled = cv2.merge((b, g, r, np.ones((src_width_scaled, src_height_scaled, 1), np.uint8) * 255))
    src_img_scaled = src_img_scaled.astype(float) / 255
    # draw image
    draw_img = np.zeros((src_width_scaled, src_height_scaled, 4), np.uint8)
    draw_img = draw_img.astype(float)
    # mask image
    mask_image = np.zeros((src_width_scaled, src_height_scaled, 4), np.uint8)
    mask_image = mask_image.astype(float)
    # text image
    annotations_img = np.zeros((src_width_scaled, src_height_scaled, 4), np.uint8)
    annotations_img = annotations_img.astype(float)

# endregion MAIN VARIABLES

# region AUX METHODS

# NOT USED
def overlay_type_additive():
    r = cv2.add(src_img_scaled, draw_img)
    return cv2.add(r, annotations_img)


def overlay_type_cv_blend():
    r = auxfuncs.alpha_blend(src_img_scaled, draw_img)
    return auxfuncs.alpha_blend(r, annotations_img)


def overlay_type_cv_blend_mask():
    return auxfuncs.alpha_blend(src_img_scaled, mask_image)


def create_annotation(x, y, text, window):
    global annotations, img2show, annotations_img
    new_annotation = [(math.floor(x * scale), math.floor(y * scale)), text]
    print("debug - annotation added" + str(new_annotation))
    annotations.append(new_annotation)
    auxfuncs.paint_label(annotations_img, x, y, text, font_scale=font_scale)
    window.withdraw()

    img2show = overlay_type_cv_blend()
    cv2.imshow(main_window_name, img2show)

    cv2.setMouseCallback(main_window_name, mouse_callback)


def mouse_callback_none(event, x, y, flags, param):
    return None


# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, mode, scale, img2show, draw_img

    if mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                #cv2.rectangle(draw_img, (ix, iy), (x, y), draw_rectangle_color, -1)
                cv2.circle(draw_img, (x, y), 5, draw_circle_color, -1)
                img2show = overlay_type_cv_blend()
                cv2.imshow(main_window_name, img2show)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # cv2.rectangle(draw_img, (ix, iy), (x, y), draw_rectangle_color, -1)
            cv2.circle(draw_img, (x, y), 5, draw_circle_color, -1)
    else:
        if event == cv2.EVENT_LBUTTONDBLCLK:
            text_window = Toplevel(root)
            text_frame = Frame(text_window)
            text_frame.pack(side="top", fill="both")
            Label(text_frame, text="Annotation Text", justify=LEFT).pack(side=LEFT)
            text_var = StringVar()
            Entry(text_frame, textvariable=text_var).pack(side=LEFT)

            def com():
                create_annotation(x, y, text_var.get(), text_window)
            Button(text_frame, text="Add", command=com).pack(side="top")


def save(img):
    if kp1 is None or des1 is None:
        print("Missing keypoints or descriptors")
        return

    # src_img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if len(kp1) < 4:
        print("There must be at least 4 key points")
        return

    with open('testdata.pkl', 'wb') as output:
        pickle.dump(auxfuncs.pickle_keypoints(kp1, des1), output)
        pickle.dump(annotations, output, pickle.HIGHEST_PROTOCOL)  # annotations position are relative to src_img size
        pickle.dump(draw_img, output, pickle.HIGHEST_PROTOCOL)  # visual annotations as user draws in a scaled down img
        pickle.dump(src_img, output, pickle.HIGHEST_PROTOCOL)  # only need size but saving the entire image can be useful for debuging

# endregion AUX METHODS


# region Main Program Code


def center(top_level):
    top_level.update_idletasks()
    w = top_level.winfo_screenwidth()
    h = top_level.winfo_screenheight()
    size = tuple(int(_) for _ in top_level.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    top_level.geometry("%dx%d+%d+%d" % (size + (x, y)))


def load_image():
    global src_img, img2show, main_window_name, mode
    filename = of.get_file()
    src_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    scale_image()
    main_window_name = 'Prepare Program'
    cv2.namedWindow(main_window_name)
    cv2.setMouseCallback(main_window_name, mouse_callback)
    img2show = overlay_type_cv_blend()
    cv2.imshow(main_window_name, img2show)
    control = Toplevel(root)
    control_frame = Frame(control)
    control_frame.pack(side="top", fill="both")


def compute_sift(img):
    global kp1, des1, mask, mask_image, scale
    img_test = img.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    mask = mask_image.copy() * 255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(255 - mask, (0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
    kp1, des1 = sift.detectAndCompute(img, mask)
    auxfuncs.cv_showWindowWithMaxDim('\'DB\' features', cv2.drawKeypoints(img_test, kp1, img_test), maxdim=500)


def mask_mouse_callback(event, x, y, flags, param):
    global mask_drawing, mask_image, circle_radius, mask_y, mask_x, img2show_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        mask_drawing = True
        mask_x, mask_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if mask_drawing:
            cv2.circle(mask_image, (x, y), circle_radius.get(), draw_circle_color_mask, -1)
            img2show_mask = overlay_type_cv_blend_mask()
            cv2.imshow(mask_window_name, img2show_mask)

    elif event == cv2.EVENT_LBUTTONUP:
        mask_drawing = False


def open_mask_creation():
    global mask_window_name, img2show_mask
    mask_window_name = 'Mask Creation'
    cv2.namedWindow(mask_window_name)
    cv2.setMouseCallback(mask_window_name, mask_mouse_callback)
    img2show_mask = overlay_type_cv_blend_mask()
    cv2.imshow(mask_window_name, img2show_mask)


def mode_command():
    global mode
    mode = not mode
    if mode:
        mode_text.set('mode: shape')
    else:
        mode_text.set('mode: label')


root.title('Prepare')
center(root)
frame = Frame(root)
frame.pack(side="top", fill="both")
Button(frame, text="Load Image", command=load_image).pack(side="top")
mode_text = StringVar()
mode_text.set('mode: shape')
Label(frame, textvariable=mode_text).pack()
Button(frame, text="Change", command=mode_command).pack()
Button(frame, text="Load Mask").pack(side="top")
Button(frame, text="Create Mask", command=open_mask_creation).pack(side="top")
Label(frame, text="Circle Radius").pack()
Scale(frame, from_=0, to=100, orient=HORIZONTAL, variable=circle_radius).pack()
Button(frame, text="Compute KeyPoint", command=lambda: compute_sift(src_img)).pack(side="top")
Button(frame, text="Save to File", command=lambda: save(img2show)).pack(side="top")
Button(frame, text="Close OpenCV Windows", command=cv2.destroyAllWindows).pack(side="top")
root.mainloop()
cv2.destroyAllWindows()

# endregion Main Program Code
