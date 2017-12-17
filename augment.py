import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import pickle
#from PIL import Image
#from PIL import ImageTk
#import auxfuncs
from auxfuncs import *
from tkinter import *
from tkinter import filedialog

#region NORMAl & TEST MODE RELATED & other info output VARIABLES & METHODS
TEST = True
NORMAL = False
mode_var = None
explain_label_svar = None #initialized in menu with StringVar()
test_images_descriptions = """Window name - Description
\n'DB' features - database image & keypoints
\n'aug' features - keypoints found in the image to be augmented
\nAll matches - Draws all homography matches even if they are outliers (using RANSAC)
\nRANSAC Inliers Matches - Draws homography valid matches only (using RANSAC)
\n'raw' augmented - the final image 'extended', without being cropped to the original source size. may contain annotations that are lost after the crop operation.
\nAUGMENTED - the final image that is presented to the user
"""

def resetExplainLabel():
    global explain_label_svar,explain_label
    explain_label_svar.set('Steps description:')
def PrintTestStep(text,level=1):
    if mode_var.get()==TEST:
        global explain_label_svar, explain_label
        print('Test Mode : ' + 'Sub'*level + ' Step >> ' + text)
        explain_label_svar.set(explain_label_svar.get() + '\n' + '~ ' * level + '| '+ text)
def PrintTestNote(msg):
    if mode_var.get()==TEST:
        global explain_label_svar, explain_label
        print('Test Mode : explain:' + msg)
        explain_label_svar.set(explain_label_svar.get() + '\n' + msg)
def PrintInfoAndCleanPrevText(msg):
    global explain_label_svar, explain_label
    explain_label_svar.set('Info:')
    print('Info : explain:' + msg)
    explain_label_svar.set(explain_label_svar.get() + '\n' + msg)
def PrintInfo(msg):
    global explain_label_svar, explain_label
    print('Info : explain:' + msg)
    explain_label_svar.set(explain_label_svar.get() + '\n\nInfo:' + msg)
#endregion

#region DB images data & image to augment variables & METHODS
db_images_names = []
db_image_i = None #current DB image loaded in iteration
user_img = None #image to be augmented

def clearDB():
    global db_images_names
    db_images_names = []
    PrintInfoAndCleanPrevText('pkl files list emptied')
def loadDB():
    global db_images_names
    filename = filedialog.askopenfilename(filetypes=(("pkl files", "*.pkl"), ("all files", "*.*")),
                                          title="Choose an Image File")
    if filename is not None and filename != '':
        db_images_names.append(filename)
        PrintInfoAndCleanPrevText(filename + ' added to plk list')
def previewDB():
    global db_images_names
    if db_images_names is not None and len(db_images_names)>0:
        for db_item in db_images_names:
            with open(db_item, 'rb') as input:
                kp1, des1 = unpickle_keypoints(pickle.load(input))
                annotations = pickle.load(input)
                drawn_image = pickle.load(input)
                src_img = pickle.load(input)#not really needed but useful for debug purposes.
                dw, dh = drawn_image.shape[1::-1]
                sw, sh = src_img.shape[1::-1]  # dimensions of the source image from which the loaded keypoints were extracted
                drawn_image = cv2.resize(drawn_image, (0, 0), fx=sw / dw, fy=sh / dh, interpolation=cv2.INTER_CUBIC)
                temp_img = np.concatenate((alpha_blend(background=src_img.astype(float) / 255,
                                                       foreground=drawn_image.astype(float), channels=3),
                                          cv2.drawKeypoints(src_img, kp1, src_img).astype(float)/255
                                          ), axis=1)
                cv_showWindowWithMaxDim('DB:' + db_item + ' (preview)',temp_img ,  maxdim=500)
    else: PrintInfoAndCleanPrevText('pkl files list is empty!')

def loadImage():
    global user_img
    filename = filedialog.askopenfilename(filetypes=(("jpeg, png files", "*.jpg *.png"), ("all files", "*.*")),
                               title="Choose an Image File")
    if filename is not None and filename != '':
        user_img = cv2.imread(filename,cv2.IMREAD_COLOR)
        PrintInfoAndCleanPrevText(filename + ' loaded')#TODO add algorithm info
def previewLoadImage():
    if user_img is not None:
        cv_showWindowWithMaxDim('Source to Augment Preview', user_img, maxdim=300)
    else: PrintInfoAndCleanPrevText('no image loaded!')

#endregion

def FindGoodMatchesLowes(matches):
    PrintTestStep('store all the good matches as per Lowe\'s ratio test.')
    r = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            r.append(m)
    return r

#TODO select user image (that will be augmented)
#TODO loop to add stored files with the descriptors and annotations
#TODO




def augment(BFlimit=-1000,featuresType='sift',norm=cv2.NORM_L2,crossCheck=False):
    resetExplainLabel()
    #norm=cv2.HAMMING_NORM_TYPE
    #crossCheck=True
    #LOAD REFERENCE IMAGE DATA
    with open('testdata.pkl', 'rb') as input:
        kp1, des1 = unpickle_keypoints(pickle.load(input))
        annotations = pickle.load(input)
        drawn_image = pickle.load(input)
        src_img = pickle.load(input)#not really needed but useful for debug purposes.


    if mode_var.get() == TEST:
        test_images_description_window = Tk()
        test_images_description_window.title("Test Images Descriptions")
        Label(test_images_description_window, text=test_images_descriptions
              ,width=60,height=20,anchor=NW,justify="left",wraplength=400).pack(side="top",fill="both",expand="yes")
        temp_img = src_img.copy()
        cv_showWindowWithMaxDim('\'DB\' features', cv2.drawKeypoints(temp_img,kp1,temp_img),maxdim=500)

    #DEBUG STUFF
    #print(str(kp1) + str(des1))
    #print(annotations)
    #cv2.imshow('drawn',drawn_image)

    if user_img is None:
        PrintInfo('No image to augment was yet loaded!')
        return False

    try:
        PrintTestStep('extract keypoints and descriptors from user image with SIFT')
        user_img_grey = cv2.cvtColor(user_img,cv2.COLOR_RGB2GRAY)
        feature_detector = cv2.xfeatures2d.SIFT_create()  # cv2.ORB_create(nfeatures=5000)#cv2.xfeatures2d.SURF_create()#cv2.xfeatures2d.SIFT_create()
        kp2, des2 = feature_detector.detectAndCompute(user_img_grey, None)
        if mode_var.get() == TEST:
            temp_img = user_img.copy()
            cv_showWindowWithMaxDim('\'aug\' features', cv2.drawKeypoints(temp_img, kp2,temp_img),maxdim=500)
    except:
        print('failed to get features from image to augment')
        print("Unexpected error:" + str(sys.exc_info()[0]))
        return False

    PrintTestStep('select matcher based on number of descriptors')
    if len(des1)+len(des2) < BFlimit or BFlimit<0:
        PrintTestStep('create Brute Force matcher (prioritize precision)')
        matcher = cv2.BFMatcher(normType=norm,crossCheck=crossCheck)
    else:
        PrintTestStep('create FLANN based matcher (prioritize speed)')
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        matcher = cv2.FlannBasedMatcher(index_params,search_params)


    if crossCheck:
        PrintTestStep('find matches with K-Nearest Neighbors (knn)')
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:10]
    else:
        PrintTestStep('find matches with K-Nearest Neighbors (knn)')
        matches = matcher.knnMatch(des1,des2,k=2)
        PrintTestStep('store all the good matches as per Lowe\'s ratio test.')
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

    # if there very few matches do nothing
    MIN_MATCH_COUNT = 10
    if len(good)<MIN_MATCH_COUNT:
    #ACCEPT_RATIO = .1
    #good_ratio = float(len(good)) / float(min(len(kp1),len(kp2)))
    #if good_ratio < ACCEPT_RATIO:
        PrintInfo('Not enough matches found : ' + str(len(good)) + '/' + str(MIN_MATCH_COUNT))
        return False

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    PrintTestStep('find homography between images using RANSAC (RANdom SAmple Consensus)')
    transformMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #TODO considerar mudar o threshold
    matchesMask = mask.ravel().tolist() #only used in debug stuff


    PrintTestStep('format draw and user images')
    try:
        dw, dh = drawn_image.shape[1::-1] #dimensions of the draw image
        uw,uh = user_img.shape[1::-1]     #dimensions of the picture given by the user
        #resize draw image to original trainning mage size
        sw,sh = src_img.shape[1::-1]      #dimensions of the source image from which the loaded keypoints were extracted
        PrintTestStep('resize draw image to fit the dimensions of the original source image from which the descriptors and keypoints were extracted',level=2)
        draw_img_ext = cv2.resize(drawn_image, (0, 0), fx=sw/dw, fy=sh/dh, interpolation=cv2.INTER_CUBIC)
        dw, dh = draw_img_ext.shape[1::-1] #update dimensions after scaling

        PrintTestStep(
        """extend images borders so they have the same dimensions. This is needed to:
     - avoid loosing draw image parts that should be visible in the final output;
     - allow to apply alphaBlend function using the given images"""
            , level=2
        )

        #get the maximum sizes between the 2 images on each dimension
        maxw , maxh = (max(dw, uw), max(dh, uh))
        #print('%d,%d,%d,%d,%d,%d,%d,%d,' % (dw, dh, uw, uh, maxw, maxh, maxh - dh, maxw - dw)) #DEBUG PRINT, SIZES
        draw_img_ext = cv2.copyMakeBorder(draw_img_ext, 0, maxh - dh, 0, maxw - dw, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        user_img_ext = cv2.copyMakeBorder(user_img,0,maxh-uh,0,maxw-uw,borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
    except:
        print('failed to extend images')
        print("Unexpected error:" + str(sys.exc_info()[0]))
        return False

    PrintTestStep('build final image')
    try:
        PrintTestStep('warp draw image', level=2)
        im_temp = cv2.warpPerspective(draw_img_ext, transformMatrix, draw_img_ext.shape[1::-1])#TODO review this

        #This step is commented because is only needed when not using alph blending to find the position of  corners of te warped image on the user image
        #note: dw & dh are the extended draw image dimensions obtained on previous step
        #extended_drawn_image_points = np.array([[[0, 0]], [[dw - 1, 0]], [[dw - 1, dh - 1]], [[0, dh - 1]]], dtype=np.float32)
        #projpoints= cv2.perspectiveTransform(extended_drawn_image_points,transformMatrix)

        PrintTestStep('alpha blend draw image over user image', level=2)
        outimage = alpha_blend(user_img_ext.astype(float)/255,im_temp.astype(float),channels=3)

        #TODO add annotations, maybe add dynamicly on a callback with mouse over
        #TODO resize out image to target scale

        if mode_var.get() == TEST:
            cv_showWindowWithMaxDim('\'raw\' augmented', outimage,maxdim=500)
    except:
        print('failed to build annotated image')
        print("Unexpected error:" + str(sys.exc_info()[0]))
        return False

    if mode_var.get()==TEST:
        h, w, dims = src_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, transformMatrix)

        img3 = user_img.copy()
        user_img_matches = cv2.polylines(img3, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor = (0,0,255), # draw all in red
                    singlePointColor = None,
                    matchesMask = None,
                    flags = 2)
        img3 = cv2.drawMatches(src_img, kp1, user_img_matches, kp2, good, None, **draw_params)
        cv_showWindowWithMaxDim('All Matches', img3, maxdim=500)

        draw_params = dict(matchColor = (0,255,0), # draw only 'inliners' in green color
                    singlePointColor = None,
                    matchesMask = matchesMask,
                    flags = 2)
        img3 = cv2.drawMatches(src_img, kp1, user_img_matches, kp2, good, None, **draw_params)
        cv_showWindowWithMaxDim('RANSAC Inliers Matches', img3, maxdim=500)
    try:
        PrintTestStep('Crop extended image and display it')
        outimage = outimage[0:uh,0:uw]
        cv_showWindowWithMaxDim('AUGMENTED', outimage, maxdim=500, sw=uw, sh=uh)
    except:
        PrintInfo('Failed to \'reframe\' final image!')
        return False
    #cv2.waitKey(0)
    return True



if __name__ == "__main__":
    tkRoot = Tk()
    tkRoot.title('augment')

    frameTestMode = Frame(tkRoot)
    frameTestMode.pack(side="top",fill="both")
    mode_var = BooleanVar()
    Label(frameTestMode,text="TEST MODE",anchor=W,justify="left").pack(side="left")
    Radiobutton(frameTestMode, text="On", variable=mode_var, value=True).pack(side="left" , anchor=W)
    Radiobutton(frameTestMode, text="Off", variable=mode_var, value=False).pack(side="left" ,anchor=W)

    frameDB = Frame(tkRoot)
    frameDB.pack(side="top", fill="both")
    Button(frameDB, text="Add file to DB files list", command=loadDB).pack(side="left", fill="both", expand="yes")
    Button(frameDB, text="Clear files DB list", command=clearDB).pack(side="left", fill="both", expand="yes")
    Button(frameDB, text="Preview DB", command=previewDB).pack(side="left", fill="both", expand="yes")

    frameLoadImage = Frame(tkRoot)
    frameLoadImage.pack(side="top", fill="both")
    Button(frameLoadImage, text="Load Image", command=loadImage).pack(side="left", fill="both", expand="yes")
    Button(frameLoadImage, text="Preview", command=previewLoadImage).pack(side="left", fill="both", expand="yes")

    Button(tkRoot, text="Augment", command=augment).pack(side="top", fill="both", expand="yes")
    Button(tkRoot, text="Close All Cv Windows", command=cv2.destroyAllWindows).pack(side="top", fill="both", expand="yes")

    explain_label_svar = StringVar(tkRoot)
    resetExplainLabel()
    explain_label = Label(tkRoot,textvariable=explain_label_svar,width=70,height=25,anchor=NW,justify="left",wraplength=500)
    explain_label.pack(side="top", fill="both", expand="yes")

    tkRoot.mainloop()
    cv2.destroyAllWindows()
    #tkRoot.destroy()
