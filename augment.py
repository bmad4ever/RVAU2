import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import pickle
import auxfuncs

TEST = True
NORMAL = False
MODE = TEST

def PrintTestStep(text,level=1):
    if MODE==TEST: print('Test Mode : ' + 'Sub'*level + ' Step >> ' + text)
def PrintTestNote(msg):
    if MODE==TEST: print('Test Mode : explain:' + msg)


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

#LOAD REFERENCE IMAGE DATA
with open('testdata.pkl', 'rb') as input:
    kp1, des1 = auxfuncs.unpickle_keypoints(pickle.load(input))
    annotations = pickle.load(input)
    drawn_image = pickle.load(input)
    src_img = pickle.load(input)#not really needed but useful for debug purposes.

#DEBUG STUFF
print(str(kp1) + str(des1))
#print(annotations)
#cv2.imshow('drawn',drawn_image)

#LOAD USER GIVEN IMAGE
user_img = cv2.imread('./images/poster_test.jpg',cv2.IMREAD_COLOR)
user_img_grey = cv2.cvtColor(user_img,cv2.COLOR_RGB2GRAY)
#src_img = cv2.imread('./images/poster4.jpg',0) # trainImage

PrintTestStep('extract keypoints and descriptors from user image with SIFT')
sift = cv2.xfeatures2d.SIFT_create()
kp2, des2 = sift.detectAndCompute(user_img_grey, None)

PrintTestStep('select matcher based on number of descriptors')
if len(des1)+len(des2) < 1000:
    PrintTestStep('create Brute Force matcher (prioritize precision)')
    matcher = cv2.BFMatcher()
else:
    PrintTestStep('create FLANN based matcher (prioritize speed)')
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
PrintTestStep('find matches with k-nearest neighbors with k = 2')
matches = matcher.knnMatch(des1,des2,k=2)

PrintTestStep('store all the good matches as per Lowe\'s ratio test.')
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# if there very few matches do nothing
MIN_MATCH_COUNT = 50
if len(good)<MIN_MATCH_COUNT:
#ACCEPT_RATIO = .1
#good_ratio = float(len(good)) / float(min(len(kp1),len(kp2)))
#if good_ratio < ACCEPT_RATIO:
    print("Not enough matches found : %d/%d" % (len(good),MIN_MATCH_COUNT))
    #print('ratio of good matches is ' + str(good_ratio) + ' and minimum acceptable is ' + str(ACCEPT_RATIO))
    key = cv2.waitKey(1)
    quit()

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

PrintTestStep('find homography between images using RANSAC (RANdom SAmple Consensus)')
transformMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #TODO considerar mudar o threshold
matchesMask = mask.ravel().tolist() #only used only in debug stuff


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
       \n-avoid loosing draw image parts that should be visible in the final output;
       \n-allow to apply alphaBlend function using the given images"""
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
    raise#quit()

PrintTestStep('build display image by: warping draw image; alphaBlend it over the user image; add annotations on top;')
try:
    PrintTestStep('warp draw image', level=2)
    im_temp = cv2.warpPerspective(draw_img_ext, transformMatrix, draw_img_ext.shape[1::-1])#TODO review this

    #This step is commented because is only needed when not using alph blending to find the position of  corners of te warped image on the user image
    #note: dw & dh are the extended draw image dimensions obtained on previous step
    #extended_drawn_image_points = np.array([[[0, 0]], [[dw - 1, 0]], [[dw - 1, dh - 1]], [[0, dh - 1]]], dtype=np.float32)
    #projpoints= cv2.perspectiveTransform(extended_drawn_image_points,transformMatrix)

    PrintTestStep('alpha blend draw image over user image', level=2)
    outimage = auxfuncs.alpha_blend(user_img_ext.astype(float)/255,im_temp.astype(float),channels=3)

    #TODO add annotations, maybe add dynamicly on a callback with mouse over
    #TODO resize out image to target scale
except:
    print('failed to build annotated image')
    print("Unexpected error:" + str(sys.exc_info()[0]))
    raise#quit()

cv2.imshow('draw',outimage)

#src_img = cv2.cvtColor(src_img,cv2.COLOR_RGB2GRAY)
h,w,dims = src_img.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, transformMatrix)

#TODO use M to apply perspective transform onto annotations

user_img_matches = cv2.polylines(user_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(src_img, kp1, user_img_matches, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray'),plt.show()


cv2.waitKey(0)