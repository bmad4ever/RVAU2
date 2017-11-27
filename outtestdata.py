#this program creates dummy data for augment.py

import numpy as np
import cv2
import pickle
import auxfuncs

src_img = cv2.imread('./images/poster4.jpg',cv2.IMREAD_COLOR) # trainImage
src_img_grey = cv2.cvtColor(src_img,cv2.COLOR_RGB2GRAY)
src_width, src_height, channels = tuple(src_img.shape)
#if channels is not None and channels < 4:
#    b, g, r = cv2.split(src_img)
#    src_img = cv2.merge((b, g, r, np.ones((src_width, src_height, 1), np.uint8) * 255))
#src_img = src_img.astype(float) / 255

draw_img = np.zeros((200,200,4))
cv2.circle(draw_img, (50, 50), 15, (1,1,0,1), -1)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(src_img_grey,None)

with open('testdata.pkl', 'wb') as output:
    pickle.dump(auxfuncs.pickle_keypoints(kp1, des1),output) #pickle.dump((kp1,des1), output, pickle.HIGHEST_PROTOCOL)
    pickle.dump([((12,0),'somestring')], output, pickle.HIGHEST_PROTOCOL)#annotations position are relative to src_img size
    pickle.dump(draw_img, output, pickle.HIGHEST_PROTOCOL)#visual annotations as user draws in a scaled down img
    pickle.dump(src_img, output, pickle.HIGHEST_PROTOCOL) #only need size but saving the entire image can be useful for debuging