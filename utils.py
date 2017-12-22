from PIL import Image
import numpy as np
import tensorflow as tf
import copy
import os
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []
    def __call__(self, image):
        if self.maxsize == 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img=self.num_img+1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp = copy.copy(self.images[idx])
            self.images[idx] = image
            return tmp
        else:
            return image

def load_image(im_path):
	im = Image.open(im_path)
	im = np.expand_dims(np.array(im).astype(np.float32), axis=0)
	im = im/127.5 - 1.0
	return im

def save_image(im, im_path, phase, ptype):
	im = np.uint8((im+1.)*127.5)
	im = Image.fromarray(np.squeeze(im))
	data_name = im_path.split(os.sep)[3]
	if phase=="train":
		im.save(os.path.join('checkpoints', ptype + data_name))
	else:
		im.save(os.path.join('result',  ptype + data_name))
def transform(A, B, scale_size, crop_size):
    r_A = tf.image.resize_images(A, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
    r_B = tf.image.resize_images(B, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - crop_size + 1)), dtype=tf.int32)
    if scale_size > crop_size:
        r_A = tf.image.crop_to_bounding_box(r_A, offset[0], offset[1], crop_size, crop_size)
        r_B = tf.image.crop_to_bounding_box(r_B, offset[0], offset[1], crop_size, crop_size)
    elif scale_size < crop_size:
        raise Exception("scale size cannot be less than crop size")
    return r_A, r_B