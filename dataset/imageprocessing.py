"""Functions for image processing
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
from PIL import Image, ImageFilter
import os
import math
import random
import numpy as np
from scipy import misc
import cv2

# Calulate the shape for creating new array given (h,w)
from dataset.face_data_augment import face_image_augment_cv


def get_new_shape(images, size=None, n=None):
    shape = list(images.shape)
    if size is not None:
        h, w = tuple(size)
        shape[1] = h
        shape[2] = w
    if n is not None:
        shape[0] = n
    shape = tuple(shape)
    return shape

def random_crop(image, size):
    _h, _w = image.shape[0], image.shape[1]
    h, w = tuple(size)
    y = np.random.randint(low=0, high=_h-h+1)
    x = np.random.randint(low=0, high=_w-w+1)
    image_new = image[y:y+h, x:x+w]
    return image_new

def center_crop(image, size):
    n, _h, _w = image.shape[:3]
    h, w = tuple(size)
    assert (_h>=h and _w>=w)
    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))
    image_new = image[:, y:y+h, x:x+w]
    return image_new

def random_flip(image):
    image_new = image.copy()
    if np.random.rand()>=0.5:
        image_new = np.fliplr(image)
    return image_new

def gaussian_blur(img, radius=5.0, p=0.2):
    if np.random.rand() < p:
        pil_img = Image.fromarray(img.astype(np.uint8))  # Convert to PIL.Image
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        img = np.asarray(pil_img).astype(np.uint8)
    return img

def preprocess_imgs_np(images, is_training=False):
    images_o = []
    for img in images:
        if type(img) == str:
            img = cv2.imread(img)
            img = img[..., ::-1]  # BGR to RGB
        img_c = cv2.resize(img, (112, 112))
        if is_training:
            img_c = random_flip(img_c)
            img_c = gaussian_blur(img_c, radius=np.random.randint(3)+1, p=0.3)
            img_c = face_image_augment_cv(img_c, aug_proba=0.05, isRgbImage=True, verbose=0)
            # cv2.imshow('img', img_c[..., ::-1])
            # cv2.waitKey(0)
        # img_c = cv2.resize(img_c, (96, 96))
        # img_c = gaussian_blur(img_c, radius=5, p=0.05)
        # img_c = cv2.resize(img_c, (112, 112))
        img_c = img_c.swapaxes(1, 2).swapaxes(0, 1)
        # ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        img_c = np.array(img_c, dtype=np.float32)
        # ccropped = (ccropped - 127.5) / 128.0
        img_c = (img_c/255. - 0.5) / 0.5
        images_o += [img_c]
    return np.array(images_o)


def preprocess_imgs_np2(images, is_training=True):
    images_o = []
    for img in images:
        if type(img) == str:
            img = cv2.imread(img)
            img = img[..., ::-1]  # BGR to RGB
        resized = cv2.resize(img, (128, 128))
        ccropped = resized
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.array(ccropped, dtype=np.float32)
        # ccropped = (ccropped - 127.5) / 128.0
        ccropped = (ccropped/255. - 0.5) / 0.5
        images_o += [ccropped]
    return np.array(images_o)


