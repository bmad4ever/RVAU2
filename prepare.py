# TODO LIST
# criar menu explicativo (como funciona adição de de anotações - duplo clique, painting, mask creation, eraser, etc...)
# document functions, example, what is the purpose of center()? the name does not tell anything (ninguém faz isso...)
# remover variaveis/funções/comentarios não usados (CAGA)
# melhorar GUI (nomes de janelas, e outros) (JÁ TÁ)
# implementar load Mask ROMANO
# permitir q o utilizador defina as cores usadas no draw ROMANO

import cv2
import numpy as np
import math
import pickle
import auxfuncs
import openfile as of
from tkinter import *
from tkinter.colorchooser import *
import threading

class CvWindowRefresher(threading.Thread):
    """
    use an async opencv window that refreshes its content
    """
    def __init__(self, windowname,imagebuilderfunc):
        super(CvWindowRefresher, self).__init__()
        self.windowname = windowname
        self.imagebuilderfunc = imagebuilderfunc
        self.update = True
        self.final_img = None
    def run(self):
        global updateDrawThreadRunning
        updateDrawThreadRunning = True
        while self.update:
            try:
                self.final_img = self.imagebuilderfunc()
            except:
                return
            cv2.waitKey(1)
            if cv2.getWindowProperty(self.windowname, 0) >= 0:
                cv2.imshow(self.windowname, self.final_img)
            else:
                self.update = False
        cv2.destroyWindow(self.windowname)
        return

    def quit(self):
        self.update = False

# region MAIN VARIABLES

src_img = None  #global variable that stores loaded image

root = None                     #tkinter main window
cvAsyncPrepareWindow = None     #CvWindowRefresher used for annotations, created on load_image()
cvAsyncMaskWindow = None        #CvWindowRefresher used for drawing a mask, created on TODO

kp1 = None   # SIFT keypoints obtained from src_img
des1 = None  # SIFT descriptors obtained from src_img

# annotations are position,text tuples that are saved to a file after editing
annotations = []        # stores annotations with the following format -> ((100, 100), 'example1')
scale = 0               # scale factor applied on the original image. the scaled version is used in prepare and mask windows

# Mask Variables
mask_drawing = None
mask_image = None
mask_x, mask_y = -1, -1
mask_window_name = None
brush_radius = None

# Drawing Variables
MODE_BRUSH = 0      # value correspondent to this mode
MODE_LABEL = 1      # value correspondent to this mode
MODE_RECTANGLE = 2  # value correspondent to this mode
prepare_mode_ivar = None         # the current mode used in the prepare window
drawing = False     # true if mouse is pressed
ix, iy = -1, -1     # mouse position when button was pressed down (not held)
font_scale = 1      # the size of the labels/annotations font
#colors from 0 to 1 (1->255)
text_tag_background_color = (1, 1, 1, 1)
text_tag_text_color = (0, 0, 0, 1)
draw_rectangle_color = (0, 1, 0, 0.5)
draw_circle_color = (1, 0, 0, 0.5)
draw_circle_color_mask = (1, 1, 1, 1)
brush_circle_color_mask = (1, 1, 1, 1)
eraser_circle_color_mask = (0, 0, 0, 0)


def scale_image() -> None:
    """
    What does this scale_image() do?
        1. created a scaled version of src_img -> src_img_scaled
        2. creates blank draw, annotation and mask layers with the same dimensions of src_img_scaled
    Why is this needed?
        Big images can cause a low window refresh rate. Scaling such images down solves the problem.
    :return:
    """
    global src_img_scaled, draw_img, annotations_img, scale, mask_image
    target_length = 500     # the bigger dimension should have ~500 pixels max
    min_size = max(tuple(src_img.shape[1::-1]))
    scale = target_length / min_size
    src_img_scaled = cv2.resize(src_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if len(src_img_scaled.shape) < 3:
        src_img_scaled = cv2.cvtColor(src_img_scaled, cv2.COLOR_GRAY2RGB)
    src_width_scaled, src_height_scaled, channels = tuple(src_img_scaled.shape)

    if channels == 4:
        src_img_scaled = cv2.cvtColor(src_img_scaled, cv2.COLOR_RGBA2RGB)

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


def create_annotation(x, y, text, window):
    global annotations, annotations_img
    new_annotation = [(math.floor(x * scale), math.floor(y * scale)), text]
    print("debug - annotation added" + str(new_annotation))
    annotations.append(new_annotation)
    auxfuncs.paint_label(annotations_img, x, y, text, font_scale=font_scale)
    window.withdraw()


# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, prepare_mode_ivar, scale, draw_img, brush_radius, annotations, annotations_img

    if prepare_mode_ivar.get() == MODE_BRUSH:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            #ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(draw_img, (x, y), brush_radius.get(), draw_circle_color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(draw_img, (x, y), brush_radius.get(), draw_circle_color, -1)

    elif prepare_mode_ivar.get() == MODE_RECTANGLE:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.rectangle(draw_img, (ix, iy), (x, y), draw_rectangle_color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(draw_img, (ix, iy), (x, y), draw_rectangle_color, -1)

    elif prepare_mode_ivar.get() == MODE_LABEL:
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
        elif event == cv2.EVENT_LBUTTONDOWN:
            annotations_img = auxfuncs.detect_label_collision(annotations_img, x, y, annotations, font_scale,
                                                              scale=scale)


def save():
    if kp1 is None or des1 is None:
        print("Missing keypoints or descriptors")
        return

    if len(kp1) < 4:
        print("There must be at least 4 key points")
        return

    file = of.save_file()

    with file as output:
        pickle.dump(auxfuncs.pickle_keypoints(kp1, des1), output)
        pickle.dump(annotations, output, pickle.HIGHEST_PROTOCOL)  # annotations position are relative to src_img size
        pickle.dump(draw_img, output, pickle.HIGHEST_PROTOCOL)  # visual annotations as user draws in a scaled down img
        pickle.dump(src_img, output, pickle.HIGHEST_PROTOCOL)  # only need size but saving the entire image can be useful for debuging


def center(top_level):
    top_level.update_idletasks()
    w = top_level.winfo_screenwidth()
    h = top_level.winfo_screenheight()
    size = tuple(int(_) for _ in top_level.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    top_level.geometry("%dx%d+%d+%d" % (size + (x, y)))


def close_all_windows():
    global cvAsyncPrepareWindow, cvAsyncMaskWindow
    if cvAsyncPrepareWindow is not None:
        cvAsyncPrepareWindow.quit()
        cvAsyncPrepareWindow = None
    if cvAsyncMaskWindow is not None:
        cvAsyncMaskWindow.quit()
        cvAsyncMaskWindow = None
    cv2.destroyAllWindows()


def load_image():
    global annotations, src_img #, img2show, main_window_name, mode,cvAsyncPrepareWindow

    annotations = []    # clear annotations from previous images

    #Close active windows
    if cvAsyncPrepareWindow is not None:
        cvAsyncPrepareWindow.quit()
    if cvAsyncMaskWindow is not None:
        cvAsyncMaskWindow.quit()

    filename = of.get_file()
    if filename is None or filename == '':
        return

    src_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    scale_image()

    prepare_image()


def prepare_image():
    global cvAsyncPrepareWindow, src_img

    if src_img is None:
        warning_window = Toplevel(root)
        Label(warning_window, text="No Image Loaded!", justify=LEFT).pack(side=LEFT)
        Button(warning_window, text=" OK ", command=warning_window.destroy).pack(side=LEFT)
        return

    def overlay_type_cv_blend():
        global src_img_scaled, draw_img, annotations_img
        r = auxfuncs.alpha_blend(src_img_scaled, draw_img, channels=3)
        return auxfuncs.alpha_blend(r, annotations_img, channels=3)

    if cvAsyncPrepareWindow is not None:
        cvAsyncPrepareWindow.quit()
        cvAsyncPrepareWindow = None
        return

    main_window_name = 'Prepare Program'
    cv2.namedWindow(main_window_name)
    cv2.setMouseCallback(main_window_name, mouse_callback)
    cv2.imshow(main_window_name, overlay_type_cv_blend() )

    cvAsyncPrepareWindow = CvWindowRefresher(main_window_name,overlay_type_cv_blend)
    cvAsyncPrepareWindow.start()

def compute_sift(img):
    global kp1, des1, mask_image, scale
    img_test = img.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    mask = mask_image.copy() * 255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(255 - mask, (0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
    kp1, des1 = sift.detectAndCompute(img, mask)
    auxfuncs.cv_showWindowWithMaxDim('\'DB\' features', cv2.drawKeypoints(img_test, kp1, img_test), maxdim=500)


def mask_mouse_callback(event, x, y, flags, param):
    global mask_drawing, mask_image, brush_radius, mask_y, mask_x#, img2show_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        mask_drawing = True
        mask_x, mask_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if mask_drawing:
            cv2.circle(mask_image, (x, y), brush_radius.get(), draw_circle_color_mask, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        mask_drawing = False


def open_mask_creation():
    global cvAsyncMaskWindow

    if src_img is None:
        warning_window = Toplevel(root)
        Label(warning_window, text="No Image Loaded!", justify=LEFT).pack(side=LEFT)
        Button(warning_window, text=" OK ", command=warning_window.destroy).pack(side=LEFT)
        return

    def overlay_type_cv_blend_mask():
        global src_img_scaled,mask_image
        return auxfuncs.alpha_blend(src_img_scaled, mask_image, channels=3)

    if cvAsyncMaskWindow is not None:
        cvAsyncMaskWindow.quit()
        cvAsyncMaskWindow = None
        return

    mask_window_name = 'Mask Creation'
    cv2.namedWindow(mask_window_name)
    cv2.setMouseCallback(mask_window_name, mask_mouse_callback)
    img2show_mask = overlay_type_cv_blend_mask()
    cv2.imshow(mask_window_name, img2show_mask)

    cvAsyncMaskWindow = CvWindowRefresher(mask_window_name, overlay_type_cv_blend_mask)
    cvAsyncMaskWindow.start()


def change_brush():
    global draw_circle_color_mask, draw_circle_color
    if erase_brush.get():
        draw_circle_color_mask = eraser_circle_color_mask
        draw_circle_color = eraser_circle_color_mask
    else:
        draw_circle_color_mask = brush_circle_color_mask
        draw_circle_color = (float(brush_color[0][0]) / 255,
                             float(brush_color[0][1]) / 255,
                             float(brush_color[0][2]) / 255,
                             0.5)


# endregion AUX METHODS

#TODO add main here


root = Tk()
root.title('Prepare')
#get screen dimensions
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen

header_frame = Frame(root)
header_frame.pack(side="top", fill="both")
Button(header_frame, text=" Load Image ", command=load_image).pack(side=LEFT,fill="both", expand="yes")
Button(header_frame, text=" Load Mask ").pack(side=LEFT, fill="both", expand="yes")
Button(header_frame, text="Compute KeyPoint", command=lambda: compute_sift(src_img)).pack(side=LEFT, fill="both", expand="yes")
Button(header_frame, text="Save to File", command=save).pack(side=LEFT, fill="both", expand="yes")
Button(header_frame, text="Close OpenCV Windows", command=close_all_windows).pack(side="top")

#Label(root, text='_'*100).pack(side=TOP, fill="both", expand="yes")

prepare_frame = Frame(root)
prepare_frame.pack(side="top", fill="both")
Button(prepare_frame, text="Open/Close Prepare Image", command=prepare_image).pack(side=LEFT,fill="both", expand="yes")
Button(prepare_frame, text="Open/Close Custom Mask Editor", command=open_mask_creation).pack(side=LEFT, fill="both", expand="yes")
Button(prepare_frame, text="HELP", command=None).pack(side=LEFT, fill="both", expand="yes")

#Prepare image Options
#mode_text = StringVar()
#mode_text.set('mode: shape')
#Label(root, textvariable=mode_text).pack()
#Button(root, text="     Change     ", command=mode_command).pack()

prepare_mode_ivar = IntVar()
prepare_mode_ivar.set(1)
prepare_mode_frame = Frame(root)
prepare_mode_frame.pack(side="top", fill="both")
Label(prepare_mode_frame, text='mode').pack(side="left", anchor=W)
Radiobutton(prepare_mode_frame, text="brush", variable=prepare_mode_ivar, value=0).pack(side="left", anchor=W)
Radiobutton(prepare_mode_frame, text="rectangles", variable=prepare_mode_ivar, value=2).pack(side="left", anchor=W)
Radiobutton(prepare_mode_frame, text="label", variable=prepare_mode_ivar, value=1).pack(side="left", anchor=W)

#Brush Options
brush_radius = IntVar()
brush_radius.set(5)
preview_brush_bvar = BooleanVar()
erase_brush = BooleanVar()
erase_brush.set(False)


def preview_paintbrush_size(from_toggle):
    if preview_brush_bvar.get():
        preview_img = np.zeros((200,200,1),np.float32)
        cv2.circle(preview_img, (100, 100), brush_radius.get(), 1, -1)
        cv2.imshow('brush preview', preview_img)
        if from_toggle:
            cv2.moveWindow('brush preview', 0, 25*5+30);
    else:
        cv2.destroyWindow('brush preview')


circle_frame = Frame(root)
circle_frame.pack(side="top", fill="both")
Label(circle_frame, text="\n Circle Radius  ").pack(side=LEFT, fill="both")
Scale(circle_frame, from_=0, to=100, orient=HORIZONTAL, variable=brush_radius, command=lambda v: preview_paintbrush_size(False)).pack(side=LEFT, fill="both", expand="yes")
Label(circle_frame, text="\nErase Brush").pack(side=LEFT, fill="both")
Checkbutton(circle_frame, variable=erase_brush, command=change_brush).pack(side=LEFT, anchor=S)
Label(circle_frame, text="\nPreview Brush").pack(side=LEFT, fill="both")
Checkbutton(circle_frame, variable=preview_brush_bvar, command=lambda: preview_paintbrush_size(True)).pack(side=LEFT, anchor=S)


def change_rectangle_color():
    global rectangle_color, rectangle_color_label, draw_rectangle_color
    rectangle_color = askcolor()
    rectangle_color_label.config(background=rectangle_color[1])
    draw_rectangle_color = (float(rectangle_color[0][2]) / 255,
                            float(rectangle_color[0][1]) / 255,
                            float(rectangle_color[0][0]) / 255,
                            0.5)


def change_brush_color():
    global brush_color, brush_color_label, draw_circle_color
    brush_color = askcolor()
    brush_color_label.config(background=brush_color[1])
    draw_circle_color = (float(brush_color[0][2]) / 255,
                         float(brush_color[0][1]) / 255,
                         float(brush_color[0][0]) / 255,
                         0.5)


# Color picker Frame
color_frame = Frame(root)
color_frame.pack(side="top", fill="both")
rectangle_color = '#%02x%02x%02x' % (int(draw_rectangle_color[0] * 255),
                                     int(draw_rectangle_color[1] * 255),
                                     int(draw_rectangle_color[2] * 255))
brush_color = '#%02x%02x%02x' % (int(draw_circle_color[2] * 255),
                                 int(draw_circle_color[1] * 255),
                                 int(draw_circle_color[0] * 255))
rectangle_color_label = Label(color_frame, text="   ", background=rectangle_color)
rectangle_color_label.pack()
Label(color_frame, text="\n Rectangle Color").pack()
Button(color_frame, text="Pick Color", command=change_rectangle_color).pack()
brush_color_label = Label(color_frame, text="   ", background=brush_color)
brush_color_label.pack()
Label(color_frame, text="\n Brush Color").pack()
Button(color_frame, text="Pick Color", command=change_brush_color).pack()


center(root)   #center window content
root.geometry('%dx%d+%d+%d' % (ws, 25*5, -8, 0))    #place window on this screen position

root.mainloop()
cv2.destroyAllWindows()

# endregion Main Program Code
