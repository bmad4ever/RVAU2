import cv2
import numpy as np

######################################################################################################
# Delicately created by Laird Foret about almost 5 years ago
# available at https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
#
#   Pickle import/export keyppoints & descriptors for opencv
######################################################################################################


def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        i+=1
        temp_array.append(temp)
    return temp_array


def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

######################################################################################################
# Image related funcs
######################################################################################################

def alpha_blend(background, foreground,channels = 4):
    '''
    :param background: background image, should have 4 channels otherwise specify channels input param
    :param foreground: last layer should be the alphas layer
    :param channels: the number of channels in background
    :return:
    '''
    alpha = cv2.split(foreground)[-1]
    alpha = cv2.merge([alpha] * channels)
    background = cv2.multiply(1.0 - alpha, background)
    if channels<4:foreground = cv2.merge(cv2.split(foreground)[:channels])
    return cv2.add(foreground, background)
