import pickle
import time
from auxfuncs import *
from tkinter import *
from tkinter import filedialog

# region MODE SELECTION & other info output VARIABLES & METHODS
TEST = True
NORMAL = False
mode_var = None
explain_label_svar = None  # initialized in menu with StringVar()
test_images_descriptions = """Window name - Description
\n'DB' features - database image & keypoints
\n'aug' features - keypoints found in the image to be augmented
\nAll matches - Draws all homography matches even if they are outliers (using RANSAC)
\nRANSAC Inliers Matches - Draws homography valid matches only (using RANSAC)
\n'raw' augmented - the final image 'extended', without being cropped to the original source size. may contain annotations that are lost after the crop operation.
\nAUGMENTED - the final image that is presented to the user
"""


def resetExplainLabel(text):
    global explain_label_svar, explain_label
    explain_label_svar.set(text)


def PrintTestStep(text, level=1):
    if mode_var.get() == TEST:
        global explain_label_svar, explain_label
        print('Test Mode : ' + 'Sub' * level + ' Step >> ' + text)
        explain_label_svar.set(explain_label_svar.get() + '\n' + '~ ' * level + '| ' + text)


def PrintTestNote(msg):
    if mode_var.get() == TEST:
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
    explain_label_svar.set(explain_label_svar.get() + '\nInfo:' + msg)


# endregion

# region DB images data & image to augment variables & METHODS
db_images_names = []
#db_image_i = None  # current DB image loaded in iteration
user_img = None  # image to be augmented

HIGH_RESOLUTION = 1400*1400       #if image resolution is higher than this warn user that may take too much time to compute

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
    if db_images_names is not None and len(db_images_names) > 0:
        for db_item in db_images_names:
            with open(db_item, 'rb') as input:
                kp1, des1 = unpickle_keypoints(pickle.load(input))
                annotations = pickle.load(input)
                drawn_image = pickle.load(input)
                src_img = pickle.load(input)  # not really needed but useful for debug purposes.
                dw, dh = drawn_image.shape[1::-1]
                sw, sh = src_img.shape[
                         1::-1]  # dimensions of the source image from which the loaded keypoints were extracted
                drawn_image = cv2.resize(drawn_image, (0, 0), fx=sw / dw, fy=sh / dh, interpolation=cv2.INTER_CUBIC)
                temp_img = np.concatenate((alpha_blend(background=src_img.astype(float) / 255,
                                                       foreground=drawn_image.astype(float), channels=3),
                                           cv2.drawKeypoints(src_img, kp1, src_img).astype(float) / 255
                                           ), axis=1)

                for annotation in annotations:
                    x, y = annotation[0]#(200,200)
                    text = annotation[1]
                    m = (sw + sh) / 2 * (10/max(len(text),7)) #adjust label size depending on image size and its text size
                    paint_label(temp_img, x=int(x), y=int(y), text=text, font_scale=max(1,m/400), text_thickness=max(1,int(m/400)))
                cv_showWindowWithMaxDim('DB:' + db_item + ' (preview)', temp_img, maxdim=500)
    else:
        PrintInfoAndCleanPrevText('pkl files list is empty!')


def loadImage():
    global user_img,HIGH_RESOLUTION
    filename = filedialog.askopenfilename(filetypes=(("jpeg, png files", "*.jpg *.png"), ("all files", "*.*")),
                                          title="Choose an Image File")
    if filename is not None and filename != '':
        user_img = cv2.imread(filename, cv2.IMREAD_COLOR)
        PrintInfoAndCleanPrevText(filename + ' loaded')  # TODO add algorithm info
        if user_img.shape[0]*user_img.shape[1] > HIGH_RESOLUTION:
            PrintInfo('-  WARNING! - ' * 12)
            PrintInfo('LOADED IMAGE IS VERY LARGE AND MAY TAKE TOO MUCH TIME TO COMPUTE ! ! !')
            PrintInfo('Consider resizing the image before augmenting.')

def previewLoadImage():
    if user_img is not None:
        cv_showWindowWithMaxDim('Source to Augment Preview', user_img, maxdim=300)
    else:
        PrintInfoAndCleanPrevText('no image loaded!')


def loadTestCase1():
    db_images_names = ['']  # TODO
    user_img = cv2.imread('', cv2.IMREAD_COLOR)

def loadTestCase2():
    db_images_names = ['', '']  # TODO
    user_img = cv2.imread('', cv2.IMREAD_COLOR)

# endregion

#region Augment Options VARS & METHODS

homo_method_ivar = None#IntVar()
ransac_thresh_dvar = None#DoubleVar()
min_good_points_ivar = None
min2speedup_ivar = None
explain_speedup_popup = None
norm_ivar = None
output2file_bvar = None
output2file_path = None
output_file_name = None
#elapsed_time_svar = None
#NOT USED! float_regex = re.compile(r"^d+\.\d+$")

def getHomoMethodName(method):
    if method == 0: return 'Normal'
    if method == cv2.RANSAC: return 'RANSAC (RANdom SAmple Consensus)'
    if method == cv2.LMEDS: return 'LMEDS (Least MEDian of Squares)'
    else: return '???'

def explain_speedup():
    global explain_speedup_popup
    if explain_speedup_popup is not None:
        explain_speedup_popup.destroy()
        explain_speedup_popup = None
        return
    explain_speedup_popup = Tk()
    explain_speedup_popup.title('Explain Speedup')
    Label(explain_speedup_popup, text=
    'If number of keyponts of both images is higher than the given value '
    + 'then FLANN based matcher is used with default params. (should be faster than brute force)'
    + '\nOtherwise, or if the given number is lesser than zero, a Brute Force Matcher '
    + 'is used with the assigned norm. CrossCheck is false and D.Lowes ratio test is used always.'
          , width=55, anchor=NW, justify="left", wraplength=400).pack()

def set_save_destination():
    global output2file_path
    dest = filedialog.askdirectory(title="Choose a destination")
    if dest is not None and dest != '': output2file_path.set(dest)

#endregion

#region augment method auxiliar methods

def findBestMatch(imgdesc):
    """
    Uses FLANN base matcher to find matches in all the DB sources.
    Select the one with most good matches.
    :param imgdesc: should be the image to augment
    :return: nothing
    """
    global db_images_names
    if len(db_images_names) == 1: return db_images_names[0]

    matcher = cv2.FlannBasedMatcher()

    best_found = db_images_names[0]
    best_ratio = 0; best_numbergood =0

    for db_item in db_images_names:
        with open(db_item, 'rb') as input:
            kp1, des1 = unpickle_keypoints(pickle.load(input))
            #no need to read rest of data... -> pickle.load(input); pickle.load(input); pickle.load(input)
            matches = matcher.knnMatch(des1, imgdesc, k=2)
            number_matches = len(matches)
            number_good_matches = len(findGoodMatchesLowes(matches))
            if number_good_matches>best_numbergood or number_good_matches == best_numbergood and number_good_matches/number_matches>best_ratio:
                best_found=db_item
                best_numbergood=number_good_matches
                best_ratio=number_good_matches/number_matches
    return best_found

def findGoodMatchesLowes(matches):
    r = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            r.append(m)
    return r

#endregion


def augment():
    """
    Augments the user image
    :return: False if could not augment the image, True otherwise
    """
    global homo_method_ivar, ransac_thresh_dvar,min_good_points_ivar,min2speedup_ivar,norm_ivar#,elapsed_time_svar
    resetExplainLabel('Steps description:')
    start_time = time.time()

#region prepare/validate variables used by the augment method
    PrintTestStep('validating input args and settingup aux varibles')
    ransac_threshold = 0.0
    if homo_method_ivar.get() == cv2.RANSAC:
        try:
            ransac_threshold = float(ransac_thresh_dvar.get())
        except:
            PrintInfo('RANSAC threshold not valid!')
            return False

    if user_img is None:
        PrintInfo('No image to augment was yet loaded!')
        return False

    if db_images_names is None or len(db_images_names)==0:
        PrintInfo('No pkl files loaded')
        return False

    try:
        min_good_points_extra = int(min_good_points_ivar.get())
    except:
        PrintInfo('Invalid Minimum Number of Good Points!')
        return False

    try:
        min2speedup = int(min2speedup_ivar.get())
    except:
        PrintInfo('Could not parse Minimum Number of Keypoints to Speedup!')
        return False

    # need at least 4
    MIN_MATCH_COUNT = 4 + min_good_points_extra
#endregion

    #compute SIFT
    try:
        PrintTestStep('extract keypoints and descriptors from user image with SIFT')
        user_img_grey = cv2.cvtColor(user_img, cv2.COLOR_RGB2GRAY)
        feature_detector = cv2.xfeatures2d.SIFT_create()  # cv2.ORB_create(nfeatures=5000)#cv2.xfeatures2d.SURF_create()#cv2.xfeatures2d.SIFT_create()
        kp2, des2 = feature_detector.detectAndCompute(user_img_grey, None)
        if mode_var.get() == TEST:
            temp_img = user_img.copy()
            cv_showWindowWithMaxDim('\'aug\' features', cv2.drawKeypoints(temp_img, kp2, temp_img), maxdim=500)
    except:
        PrintInfo('failed to get features from image to augment')
        PrintInfo("Unexpected error:" + str(sys.exc_info()[0]))
        return False

    # LOAD REFERENCE IMAGE DATA
    task_start_time = time.time()
    try:
        with open(findBestMatch(des2), 'rb') as input:
            kp1, des1 = unpickle_keypoints(pickle.load(input))
            annotations = pickle.load(input)
            drawn_image = pickle.load(input)
            src_img = pickle.load(input)  # not really needed but useful for debug purposes.
    except:
        PrintInfo("Unexpected error:" + str(sys.exc_info()[0]))
        PrintInfo('Failed to read pkl file!')
        return False
    PrintInfo("Found best image in > " + str(int((time.time() - task_start_time)*1000)) + ' milliseconds')

    # 'DEBUG LOADED DATA'; print(str(kp1) + str(des1)); print(annotations); cv2.imshow('drawn',drawn_image)

    # open a form with explanations about the different test windows
    if mode_var.get() == TEST:
        test_images_description_window = Tk()
        test_images_description_window.title("Test Images Descriptions")
        Label(test_images_description_window, text=test_images_descriptions
              , width=60, height=20, anchor=NW, justify="left", wraplength=400).pack(side="top", fill="both",
                                                                                     expand="yes")
        temp_img = src_img.copy()
        cv_showWindowWithMaxDim('\'DB\' features', cv2.drawKeypoints(temp_img, kp1, temp_img), maxdim=500)

    PrintTestStep('select matcher based on number of descriptors')
    task_start_time = time.time()
    if len(des1) + len(des2) < min2speedup or min2speedup < 0:
        PrintTestStep('create Brute Force matcher (prioritize precision)')
        matcher = cv2.BFMatcher(normType=int(norm_ivar.get()), crossCheck=False)#crossCheck)
    else:
        PrintTestStep('create FLANN based matcher (prioritize speed)')
        #'FLANN PARAMS'; FLANN_INDEX_KDTREE = 0; index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5); search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher()  # cv2.FlannBasedMatcher(index_params,search_params)

    #removed crossCheck variation
    # if crossCheck:
    #    PrintTestStep('find ' + MIN_MATCH_COUNT + ' best matches after crosscheck')
    #    matches = matcher.match(des1, des2)
    #    good = sorted(matches, key=lambda x: x.distance)[:MIN_MATCH_COUNT]
    #else:
    PrintTestStep('find matches with K-Nearest Neighbors (knn)')
    matches = matcher.knnMatch(des1, des2, k=2)
    PrintTestStep('store all the good matches as per Lowe\'s ratio test.')
    good = findGoodMatchesLowes(matches)
    PrintInfo("Found good matches in > " + str(int((time.time() - task_start_time) * 1000)) + ' milliseconds')

    # if there are very few matches do nothing
    if len(good) < MIN_MATCH_COUNT:
        PrintInfo('Not enough matches found : ' + str(len(good)) + '/(4+' + str(MIN_MATCH_COUNT-4) + ')')
        return False

    PrintTestStep('find homography between images using ' + getHomoMethodName(homo_method_ivar.get()))
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    task_start_time = time.time()
    transformMatrix, mask = cv2.findHomography(src_pts, dst_pts, homo_method_ivar.get(), ransac_threshold)
    matchesMask = mask.ravel().tolist()  # only used in test mode
    PrintInfo("Found homography in > " + str(int((time.time() - task_start_time) * 1000)) + ' milliseconds')

    PrintTestStep('format draw and user images')
    try:
        dw, dh = drawn_image.shape[1::-1]  # dimensions of the draw image
        uw, uh = user_img.shape[1::-1]  # dimensions of the picture given by the user
        # resize draw image to original trainning mage size
        sw, sh = src_img.shape[1::-1]  # dimensions of the source image from which the loaded keypoints were extracted
        PrintTestStep(
            'resize draw image to fit the dimensions of the original source image from which the descriptors and keypoints were extracted',
            level=2)
        draw_img_ext = cv2.resize(drawn_image, (0, 0), fx=sw / dw, fy=sh / dh, interpolation=cv2.INTER_CUBIC)
        dw, dh = draw_img_ext.shape[1::-1]  # update dimensions after scaling

        PrintTestStep(
            """extend images borders so they have the same dimensions. This is needed to:
     - avoid loosing draw image parts that should be visible in the final output;
     - allow to apply alphaBlend function using the given images"""
            , level=2
        )

        # get the maximum sizes between the 2 images on each dimension
        maxw, maxh = (max(dw, uw), max(dh, uh))#; print('%d,%d,%d,%d,%d,%d,%d,%d,' % (dw, dh, uw, uh, maxw, maxh, maxh - dh, maxw - dw)) #DEBUG PRINT, SIZES
        draw_img_ext = cv2.copyMakeBorder(draw_img_ext, 0, maxh - dh, 0, maxw - dw, borderType=cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0, 0))
        user_img_ext = cv2.copyMakeBorder(user_img, 0, maxh - uh, 0, maxw - uw, borderType=cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))
    except:
        PrintInfo('failed to extend images')
        PrintInfo("Unexpected error:" + str(sys.exc_info()[0]))
        return False

    PrintTestStep('build final image')
    try:
        PrintTestStep('warp draw image', level=2)
        im_temp = cv2.warpPerspective(draw_img_ext, transformMatrix, draw_img_ext.shape[1::-1])  # TODO review this

        # This step is commented because is only needed when not using alph blending to find the position of  corners of te warped image on the user image
        # note: dw & dh are the extended draw image dimensions obtained on previous step
        # extended_drawn_image_points = np.array([[[0, 0]], [[dw - 1, 0]], [[dw - 1, dh - 1]], [[0, dh - 1]]], dtype=np.float32)
        # projpoints= cv2.perspectiveTransform(extended_drawn_image_points,transformMatrix)

        PrintTestStep('alpha blend draw image over user image', level=2)
        outimage = alpha_blend(user_img_ext.astype(float) / 255, im_temp.astype(float), channels=3)
        for annotation in annotations:
            print(str(annotation))
            x,y = annotation[0] #(200,200)
            #apply homography to annotation points
            x,y = cv2.perspectiveTransform(np.array([[[x,y]]], dtype=np.float32),transformMatrix)[0][0]
            paint_label(outimage,x=int(x),y=int(y),text=annotation[1],font_scale=1.2,text_thickness=1)

        if mode_var.get() == TEST:
            cv_showWindowWithMaxDim('\'raw\' augmented', outimage, maxdim=500)
    except:
        PrintInfo('failed to build annotated image')
        PrintInfo("Unexpected error:" + str(sys.exc_info()[0]))
        return False

    # draw matches
    if mode_var.get() == TEST:
        h, w, dims = src_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, transformMatrix)

        img3 = user_img.copy()
        user_img_matches = cv2.polylines(img3, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 0, 255),  # draw all in red
                           singlePointColor=None,
                           matchesMask=None,
                           flags=2)
        img3 = cv2.drawMatches(src_img, kp1, user_img_matches, kp2, good, None, **draw_params)
        cv_showWindowWithMaxDim('All Matches', img3, maxdim=500)

        draw_params = dict(matchColor=(0, 255, 0),  # draw only 'inliners' in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        img3 = cv2.drawMatches(src_img, kp1, user_img_matches, kp2, good, None, **draw_params)
        cv_showWindowWithMaxDim( getHomoMethodName(homo_method_ivar.get()) +' Method Inliers', img3, maxdim=500)

    PrintTestStep('Crop extended image and display it')
    try:
        outimage = outimage[0:uh, 0:uw]
        cv_showWindowWithMaxDim('AUGMENTED', outimage, maxdim=500, sw=uw, sh=uh)
    except:
        PrintInfo('Failed to \'reframe\' final image!')
        PrintInfo("Unexpected error:" + str(sys.exc_info()[0]))
        return False

    #try to save to file
    try:
        if output2file_bvar.get():
            if output_file_name.get() == '':
                PrintInfo('file name not set!')
                raise Exception()
            if output2file_path.get() == '':
                PrintInfo('dest path not set!')
                raise Exception()
            cv2.imwrite(output2file_path.get()+'/'+output_file_name.get()+'.png',outimage*255)
    except:
        PrintInfo('Failed to output to file!')

    PrintInfo("Completed in > " + str(int((time.time() - start_time) * 1000)) + ' milliseconds')
    return True


if __name__ == "__main__":
    tkRoot = Tk()
    tkRoot.title('Augment')

    frameTestMode = Frame(tkRoot)
    frameTestMode.pack(side="top", fill="both")
    mode_var = BooleanVar()
    Label(frameTestMode, text="TEST MODE", anchor=W, justify="left").pack(side="left")
    Radiobutton(frameTestMode, text="On", variable=mode_var, value=True).pack(side="left", anchor=W)
    Radiobutton(frameTestMode, text="Off", variable=mode_var, value=False).pack(side="left", anchor=W)
    Button(frameTestMode, text="Close All Cv Windows", command=cv2.destroyAllWindows).pack(side="right")
    Button(frameTestMode, text="Load Sample 2", command=None).pack(side="right")
    Button(frameTestMode, text="Load Sample 1", command=None).pack(side="right")

    Label().pack()
    frameDB = Frame(tkRoot)
    frameDB.pack(side="top", fill="both")
    Label(frameDB, text="DB files references", width = 15, anchor=W, justify="left").pack(side=LEFT)
    Button(frameDB, text="Add file to DB files list", command=loadDB).pack(side="left", fill="both", expand="yes")
    Button(frameDB, text="Clear files DB list", command=clearDB).pack(side="left", fill="both", expand="yes")
    Button(frameDB, text="Preview DB", command=previewDB).pack(side="left", fill="both", expand="yes")

    frameLoadImage = Frame(tkRoot)
    Label(frameLoadImage, text="Image to Augment", width = 15, anchor=W, justify="left").pack(side=LEFT)
    frameLoadImage.pack(side="top", fill="both")
    Button(frameLoadImage, text="Load Image", command=loadImage).pack(side="left", fill="both", expand="yes")
    Button(frameLoadImage, text="Preview", command=previewLoadImage).pack(side="left", fill="both", expand="yes")

#region AUGMENT OPTIONS

    #init variables
    homo_method_ivar = IntVar(); homo_method_ivar.set(cv2.RANSAC)
    ransac_thresh_dvar = DoubleVar(); ransac_thresh_dvar.set(5.0)
    min_good_points_ivar = IntVar(); min_good_points_ivar.set(6)
    min2speedup_ivar = IntVar(); min2speedup_ivar.set(1000)
    norm_ivar = IntVar(); norm_ivar.set(cv2.NORM_L2)
    output2file_bvar = BooleanVar(); output2file_bvar.set(False)
    output2file_path = StringVar(); output2file_path.set('')
    output_file_name = StringVar(); output2file_path.set('')

    #add components to form
    Label(tkRoot, text="\nAugment Options", anchor=W, justify="left").pack()

    frameMatcher2Options = Frame(tkRoot)
    frameMatcher2Options.pack(side="top", fill="both")
    Label(frameMatcher2Options, text="Speedup matching when number keypoints is higher than:", anchor=W, justify="left").pack(side="left")
    Entry(frameMatcher2Options, textvariable=min2speedup_ivar).pack(side="left", anchor=W)
    Button(frameMatcher2Options, text="explain", command=explain_speedup).pack(side="top", fill="both", expand="yes")

    frameBFNormOptions = Frame(tkRoot)
    frameBFNormOptions.pack(side="top", fill="both")
    Label(frameBFNormOptions, text="Brute Force Norm:", anchor=W, justify="left").pack(side="left")
    Radiobutton(frameBFNormOptions, text="L1", variable=norm_ivar, value=cv2.NORM_L1).pack(side="left", anchor=W)
    Radiobutton(frameBFNormOptions, text="L2", variable=norm_ivar, value=cv2.NORM_L2).pack(side="left", anchor=W)

    frameMatcher1Options = Frame(tkRoot)
    frameMatcher1Options.pack(side="top", fill="both")
    Label(frameMatcher1Options, text="Minimum 'good' points to match 4 + ", anchor=W, justify="left").pack(side="left")
    Entry(frameMatcher1Options, textvariable=min_good_points_ivar).pack(side="left", anchor=W)

    frameHomoOptions = Frame(tkRoot)
    frameHomoOptions.pack(side="top", fill="both")
    Label(frameHomoOptions, text="find homography method:", anchor=W, justify="left").pack(side="left")
    Radiobutton(frameHomoOptions, text="Normal (all points)", variable=homo_method_ivar, value=0).pack(side="left", anchor=W)
    Radiobutton(frameHomoOptions, text="LMEDS", variable=homo_method_ivar, value=cv2.LMEDS).pack(side="left", anchor=W)
    Radiobutton(frameHomoOptions, text="RANSAC", variable=homo_method_ivar, value=cv2.RANSAC).pack(side="left", anchor=W)

    frameRANSACOptions = Frame(tkRoot)
    frameRANSACOptions.pack(side="top", fill="both")
    Label(frameRANSACOptions, text="RANSAC threshold (used only when RANSAC is selected):", anchor=W, justify="left").pack(side="left")
    Entry(frameRANSACOptions,textvariable=ransac_thresh_dvar).pack(side="left", anchor=W)

    frameOutputFileOptions = Frame(tkRoot)
    frameOutputFileOptions.pack(side="top", fill="both")
    Label(frameOutputFileOptions, text="Save Augment to file?", anchor=W, justify="left").pack(side="left")
    Radiobutton(frameOutputFileOptions, text="Yes", variable=output2file_bvar, value=True).pack(side="left", anchor=W)
    Radiobutton(frameOutputFileOptions, text="No", variable=output2file_bvar, value=False).pack(side="left", anchor=W)
    Button(frameOutputFileOptions, text="set destination", command=set_save_destination).pack(side="right")
    Entry(frameOutputFileOptions, textvariable=output_file_name).pack(side="right", anchor=W)
    Label(frameOutputFileOptions, text="Filename", anchor=W, justify="right").pack(side="right")

    frameDisplayPath = Frame(tkRoot)
    frameDisplayPath.pack(side="top", fill="both", expand="yes")
    Label(frameDisplayPath, text='Output Path > ', anchor=W, justify="left").pack(side=LEFT)
    Label(frameDisplayPath, textvariable=output2file_path, anchor=W, justify="left").pack(side=LEFT)

#endregion

    Label().pack()
    Button(tkRoot, text="Augment", command=augment).pack(side="top", fill="both", expand="yes")

    Label(tkRoot,text='_'*100).pack()#draw a line to separate from the rest of the form
    explain_label_svar = StringVar(tkRoot)
    resetExplainLabel('Info messages Area')
    explain_label = Label(tkRoot, textvariable=explain_label_svar, width=70, height=25, anchor=NW, justify="left",wraplength=500)
    explain_label.pack(side="top", fill="both", expand="yes")

    tkRoot.mainloop()
    cv2.destroyAllWindows()
    # tkRoot.destroy()
