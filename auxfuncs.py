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


######################################################################################################
# Label related funcs
######################################################################################################
def paint_label(image, x, y, text):
    size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=10, thickness=1)
    fill_color = (255, 255, 255)
    border_color = (0, 0, 0)
    margin = 50
    pointer = margin * 2
    cv2.rectangle(image,
                  (int(x - size[0][0] / 2 - margin * 2), int(y - size[0][1] - margin * 2 - pointer)),
                  (int(x + size[0][0] / 2 + margin * 2), y - pointer),
                  fill_color,
                  -1)
    cv2.rectangle(image,
                  (int(x - size[0][0] / 2 - margin * 2), int(y - size[0][1] - margin * 2 - pointer)),
                  (int(x + size[0][0] / 2 + margin * 2), int(y - pointer)),
                  border_color,
                  10)
    cv2.line(image, (x + pointer, y - pointer), (x - pointer, y - pointer), fill_color, 10)
    triangle = np.array([[x + pointer, y - pointer], [x - pointer, y - pointer], [x, y]])
    cv2.fillConvexPoly(image, triangle, fill_color)
    cv2.line(image, (x + pointer, y - pointer), (x, y), border_color, 10)
    cv2.line(image, (x - pointer, y - pointer), (x, y), border_color, 10)
    cv2.circle(image, (x, y), 10, (0, 0, 0))

    cv2.putText(img=image,
                text=text,
                org=(int(x - size[0][0] / 2), y - margin - pointer),
                color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=10,
                thickness=10)


def remove_label(image, annotation_to_remove, annotations_array):
    annotations_array.remove(annotation_to_remove)
    image = np.zeros((len(image), len(image[0]), 4), np.uint8)
    for annotation in annotations_array:
        paint_label(image, annotation[0][0], annotation[0][1], annotation[1])
