# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np
from scipy.signal import fftconvolve
from multiprocessing import Pool, cpu_count
import math
from math import pi

sum_imageset = 6
path = "/home/sheldon/camvideo_6/"

def normxcorr2(imagePath1, imagePath2, channel, mode="same"):

    if channel == "GrayScale":
        image1 = cv2.imread(imagePath1, 0)
        image2 = cv2.imread(imagePath2, 0)
    elif channel == "B":
        image1 = cv2.imread(imagePath1)
        image1 = image1[:, :, 0]
        image2 = cv2.imread(imagePath2)
        image2 = image2[:, :, 0]
    elif channel == "G":
        image1 = cv2.imread(imagePath1)
        image1 = image1[:, :, 1]
        image2 = cv2.imread(imagePath2)
        image2 = image2[:, :, 1]
    elif channel == "R":
        image1 = cv2.imread(imagePath1)
        image1 = image1[:, :, 2]
        image2 = cv2.imread(imagePath2)
        image2 = image2[:, :, 2]

    image_shape = image1.shape
    image1 = image1[0:int(image_shape[0] / 2)]
    image2 = image2[0:int(image_shape[0] / 2)]

    image1 = image1 - np.mean(image1)
    image2 = image2 - np.mean(image2)

    a1 = np.ones(image1.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(image1))
    out = fftconvolve(image2, ar.conj(), mode=mode)

    image2 = fftconvolve(np.square(image2), a1, mode=mode) - \
             np.square(fftconvolve(image2, a1, mode=mode)) / (np.prod(image1.shape))

    # Remove small machine precision errors after subtraction
    image2[np.where(image2 < 0)] = 0

    image1 = np.sum(np.square(image1))
    out = out / np.sqrt(image2 * image1)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    index = np.where(out == np.max(out))
    max_item = [np.max(out), imagePath2, int(index[1] - image_shape[1] / 2)]
    # print(max_item)
    return max_item


def ImageSetConvolution(imgpath1, imgpath2, channel):
    files1 = os.listdir(imgpath1)
    files2 = os.listdir(imgpath2)

    num_imgset1 = len(files1) - 1
    num_imgset2 = len(files2) - 1

    similarity_list = []
    shift_list = []
    sum_shift_list = []

    for i in range(0, num_imgset2):
        print("-------------" + str(i) + "--------------")
        sim_sum = 0
        item_shift_list = []
        for j in range(0, num_imgset1):
            imgFile1 = imgpath1 + "/camera_image" + str(j + 1) + ".jpeg"
            imgFile2 = imgpath2 + "/camera_image" + str((j + i) % num_imgset2 + 1) + ".jpeg"

            max_item = normxcorr2(imgFile1, imgFile2, channel)

            sim = max_item[0]
            sim_sum += sim
            shift = max_item[2]
            item_shift_list.append(shift)

        norm_sim = sim_sum / (len(files2) - 1)
        similarity_list.append(norm_sim)
        shift_list.append(item_shift_list)

    max_sim = max(similarity_list)
    max_index = similarity_list.index(max_sim)

    for i in shift_list:
        sum_shift = 0
        for j in i:
            sum_shift += j
        average_shift = sum_shift / len(i)
        sum_shift_list.append(average_shift)

    rotate_radians = max_index * 2 * pi / sum_imageset + sum_shift_list[max_index] * 1.3962634 / 800

    return_item = [imgpath1, imgpath2, max_sim, max_index, sum_shift_list[max_index], rotate_radians]

    return return_item

def calc_similarity(category, type, channel):
    similarity_list = []
    if type == "same":
        for junction in range(1, 7):
            for i in range(1, 6):
                for j in range(i+1, 6):
                    imageset1 = path+"junction_"+str(junction)+"/"+category+"/"+str(i)
                    imageset2 = path+"junction_"+str(junction)+"/"+category+"/"+str(j)
                    print(imageset1, imageset2)
                    return_item = ImageSetConvolution(imageset1, imageset2, channel)
                    similarity = return_item[2]
                    similarity_list.append(similarity)
                
    elif type == "different":
        for junction in range(1, 6):
            for i in range(1, 6):
                for diff_junction in range(junction+1, 7):
                    for j in range(1, 6):
                        imageset1 = path+"junction_"+str(junction)+"/"+category+"/"+str(i)
                        imageset2 = path+"junction_"+str(diff_junction)+"/"+category+"/"+str(j)
                        print(imageset1, imageset2)
                        return_item = ImageSetConvolution(imageset1, imageset2, channel)
                        similarity = return_item[2]
                        similarity_list.append(similarity)
    
    return similarity_list

if __name__ == '__main__':
    similarity_list = calc_similarity("perfect", "different", "GrayScale")
    print(len(similarity_list))
    print(similarity_list)