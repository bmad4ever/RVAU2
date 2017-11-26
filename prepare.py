#TODO LIST
#save drawn image and annotations to files
#write text on image, need an algorithm to properly scale !!!
#binary semi automatic selection -> GRABCUT
#   https://www.youtube.com/watch?v=mUpk3zgDAR0
#   https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html
#gather keypoints/descriptors and remove ones outside selection
#save/load keypoints/descriptors
#select a file with a GUI
    #https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
#revert action [+Z]
#draw commands list

import cv2
import numpy as np

#region MAIN VARIABLES

#annotations are position,text tuples that are saved to a file after editing
annotations = []

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

text_tag_background_color = (255,255,255,255)
text_tag_text_color = (0,0,0,255)
grabcut_color_foreround = (255,0,0,0)
grabcut_color_background = (0,0,255,0)

#reference image
src_img = cv2.imread('../posters/poster_test.jpg',cv2.IMREAD_UNCHANGED)
src_width , src_height, channels = tuple(src_img.shape)
if channels is not None and channels<4:
    b, g, r = cv2.split(src_img)
    src_img = cv2.merge((b, g, r, np.ones((src_width,src_height,1), np.uint8)*255))
#src_img = src_img.astype(float)
#draw image
draw_img = np.zeros((src_width,src_height,4), np.uint8)
#draw_img = draw_img.astype(float)
#text image
annotations_img = np.zeros((src_width,src_height,4), np.uint8)
#annotations_img = annotations_img.astype(float)

#endregion MAIN VARIABLES

#region AUX METHODS

def OVERLAY_TYPE_ADDITIVE():
    r = cv2.add(src_img,draw_img)
    return cv2.add(r,annotations_img)

##NOT WORKING  def OVERLAY_TYPE_MUL_IT():
#    r = src_img.copy()
#    overlayImages(r,draw_img)
#    overlayImages(r,annotations_img)

def OVERLAY_TYPE_CVBLEND():
    r = alphaBlend(src_img,draw_img)
    return alphaBlend(r,annotations_img)

#NOT WORKING def overlayImages(src,overlay):
#    for i in range(src_width):
#        for j in range(src_height):
#            alpha = float(overlay[i][j][3]/255)
#            src[i][j] = alpha*overlay[i][j]+(1-alpha)*src[i][j]
#            src[i][j][3] = 255

def alphaBlend(background,foreground):
    alpha = cv2.split(foreground)[3]
    alpha = cv2.merge((alpha,alpha,alpha,alpha)).astype(float)/255
    background = background/255
    foreground = foreground/255
    background = cv2.multiply(1.0 - alpha, background)
    return cv2.add(foreground, background)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,annotations

    if event == cv2.EVENT_RBUTTONDOWN:
        print('please add string')
        text = input()
        new_annotation = [(x,y),text]
        print("debug - annotation added" + str(new_annotation))
        annotations = annotations.append(new_annotation)
        #TODO
        #cv2.rectangle(img, (x-, y-), (x+, y+), (0, 255, 0), -1)
        #cv2.putText(annotations_img, time_string, (0, sizes[1] + black_margin - 5), cv2.FONT_HERSHEY_TRIPLEX, 2.5,
        #            (255, 255, 255), 2, cv2.LINE_AA)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(draw_img,(ix,iy),(x,y),(0,255,0,255),-1)
            else:
                cv2.circle(draw_img,(x,y),5,(0,0,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(draw_img,(ix,iy),(x,y),(0,255,0,255),-1)
        else:
            cv2.circle(draw_img,(x,y),5,(0,0,255,255),-1)

def SAVE():
    global annotations,draw_img
    # TODO implement

#endregion AUX METHODS

#region PROGRAM DEFINITIONS/CONFIG

#assign the method. name should start with OVERLAY_TYPE.
#OVERLAY_TYPE_CVBLEND needs that images are cast to floats. it will oclude layers below and maker program slow =(
#OVERLAY_TYPE_ADDITIVE uses the uint8 default format. it will not oclude layers below
LAYERS_OVERLAY_METHOD = OVERLAY_TYPE_ADDITIVE

MAIN_WINDOW_NAME = 'Prepare Program'

#endregion PROGRAM DEFINITIONS/CONFIG

#region Main Program Code

cv2.namedWindow(MAIN_WINDOW_NAME)
cv2.setMouseCallback(MAIN_WINDOW_NAME,draw_circle)

while(1):
    img2show = LAYERS_OVERLAY_METHOD()
    cv2.imshow(MAIN_WINDOW_NAME, img2show)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):#CHANGES DRAW MODE
        mode = not mode
    if k == ord('s'): #TODO SAVE
        break
    elif k == 27:
        break

cv2.destroyAllWindows()

#endregion Main Program Code