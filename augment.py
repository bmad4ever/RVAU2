import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import auxfuncs


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

# Find the keypoints and descriptors with SIFT
sift = cv2.xfeatures2d.SIFT_create()
#kp1, des1 = sift.detectAndCompute(src_img,None)
#kp1a, des1a = sift.detectAndCompute(src_img,None)
#print(str(kp1) + str(des1))
kp2, des2 = sift.detectAndCompute(user_img_grey, None)


# Find matches
# BFMatcher with default params
#bf = cv2.BFMatcher()
#matches = bf.knnMatch(des1,des2, k=2)
# FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
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


src_pts2 = np.float32([ kp1[m.queryIdx].pt for m in good ])
dst_pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ])
#print(str(src_pts2))

homo = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
transformMatrix, mask = homo#cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

#TODO also need to consider the reverse situation when src is bigger then user pic. extensiion on x & y must be separated
#largerImage = np.zeros(user_img.shape[:2],np.uint8)
#largerImage =
dw, dh = drawn_image.shape[1::-1]
uw,uh = user_img.shape[1::-1]
#maxw , maxh = (max(dw, uw), max(dh, uh))
#print('%d,%d,%d,%d,%d,%d,%d,%d,' % (dw, dh, uw, uh, maxw, maxh, maxh - dh, maxw - dw))

#resize draw image to original trainning mage size
sw,sh = src_img.shape[1::-1]
drawn_image = cv2.resize(drawn_image, (0, 0), fx=sw/dw, fy=sh/dh, interpolation=cv2.INTER_CUBIC)
dw, dh = drawn_image.shape[1::-1]
maxw , maxh = (max(dw, uw), max(dh, uh))
print('%d,%d,%d,%d,%d,%d,%d,%d,' % (dw, dh, uw, uh, maxw, maxh, maxh - dh, maxw - dw))

src_img_ext = cv2.copyMakeBorder(drawn_image, 0, maxh - dh, 0, maxw - dw, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

user_img_ext = cv2.copyMakeBorder(user_img,0,maxh-uh,0,maxw-uw,borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
#b, g, r = cv2.split(user_img_ext)
#user_img_ext = cv2.merge((b, g, r, np.ones((maxh,maxw, 1), np.uint8) ))
#user_img_ext = user_img_ext.astype(float)

im_temp = cv2.warpPerspective(src_img_ext, transformMatrix,drawn_image.shape[1::-1])#src_img_ext.shape[1::-1])
#projpoints = cv2.transform(
projpoints= cv2.perspectiveTransform(
    np.array([[[0, 0]], [[dw - 1, 0]], [[dw - 1, dh - 1]], [[0, dh - 1]]], dtype=np.float32)
                   ,transformMatrix#homo
                    #cv2.getPerspectiveTransform(src_pts2, dst_pts2)
)
projn = []
for i in range(len(projpoints)):
    projn.append(projpoints[i][0])
projn = np.array(projn, dtype=np.int32)
print(' > ' + str(projn))

#projpoints = np.array(projpoints,np)
#projpoints = homo * np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
#cv2.fillConvexPoly(user_img_ext,projn , 0, cv2.LINE_AA)
#outimage = user_img_ext + im_temp
outimage = auxfuncs.alpha_blend(user_img_ext.astype(float)/255,im_temp.astype(float),channels=3)
cv2.imshow('draw',outimage)

src_img = cv2.cvtColor(src_img,cv2.COLOR_RGB2GRAY)
h,w = src_img.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, transformMatrix)

#TODO use M to apply perspective transform onto annotations

user_img_grey = cv2.polylines(user_img_grey, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(src_img, kp1, user_img_grey, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray'),plt.show()


cv2.waitKey(0)