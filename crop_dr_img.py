import os
import PIL
import numpy as np
from PIL import Image

def img_to_array(img):
    return np.array(img.getdata()).reshape(img.width, img.width, 3) / 255

def trim(im):
    """trim black margin, http://stackoverflow.com/questions/10615901/trim-whitespace-using-pil"""
    bg = PIL.Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = PIL.ImageChops.difference(im, bg)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -20)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def calc_thumbnail_size(img):
    """calculate thumbnail size with constant aspect ratio"""
    width, length = img.size
    ratio = width / length

    # for some reason, if it's exactly 224, then thumnailed image is 223
    dim = 224 + 1          # output dim
    if ratio > 1:
        size = (dim * ratio, dim)
    else:
        size = (dim, dim / ratio)
#     print(size)
    return size

def calc_crop_coords(img, dim):
    """crop to square of desired dimension size"""

    width, length = img.size
    left = 0
    right = width
    bottom = length
    top = 0
    if width > dim:
        delta = (width - dim) / 2
        left = delta
        right = width - delta
    if length > dim:
        delta = (length - dim) / 2
        top = delta
        bottom = length - delta
    return (left, top, right, bottom)

def preprocess(img, dim):
    img = trim(img)
    tsize = calc_thumbnail_size(img)
    img.thumbnail(tsize)
    crop_coords = calc_crop_coords(img, dim)
    img = img.crop(crop_coords)
    return img

def crop_dr_img(img_root_path, newimg_root_path, dim = 224):
    """
    批量裁剪眼底图像
    :param img_root_path:
    :param newimg_root_path:
    :param dim: 裁剪后图像的大小
    :return:
    """
    img_names = os.listdir(img_root_path)
    number = 1
    for img_name in img_names:
        img_path = os.path.join(img_root_path, img_name)
        print("正在处理第%d张图像"%number)
        im = PIL.Image.open(img_path)
        try:
            im = preprocess(im, dim)
            newimg_path = os.path.join(newimg_root_path, img_name)
            im.save(newimg_path)
        except AttributeError:
            pass
        number += 1
        continue

    print("picture crop down")

if __name__ == '__main__':
    img_root_path = "F:/DR_detection_dataset/train"
    newimg_root_path = "F:/DR_detection_dataset/crop_train"
    crop_dr_img(img_root_path, newimg_root_path)




