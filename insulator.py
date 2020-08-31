import glob
import os
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image
import numpy as np


all_images = glob.glob(r'C:\Users\Lenovo\Desktop\insulator\*.jpg')  # 遍历文件图片
# 修改图片尺寸
filename = os.listdir("C://Users//Lenovo//Desktop//insulator//")
base_dir = "C://Users//Lenovo//Desktop//insulator//"
new_dir = "C://Users//Lenovo//Desktop//insulator//"
size_m = 416
size_n = 416  # 指定尺寸

for img in filename:
    image = Image.open(base_dir + img)
    image_size = image.resize((size_m, size_n), Image.ANTIALIAS)
    image_size.save(new_dir + img) # 保存修改尺寸后的图片


for path in all_images:
    name = os.path.basename(path)[:-4]  # 读取文件名字
    # print(name)
    images = cv2.imread(path, 0)  # 读取指定路径的图片
    images = [images, images, images, images, images, images, images]  # 指定图片增强数量
    # 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    # 建立一个名为seq的实例，定义增强方法，用于增强
    aug = iaa.Sequential(
     [
            iaa.Fliplr(0.5),  # 对50%的图像进行镜像翻转
            iaa.Flipud(0.2),  # 对20%的图像做左右翻转
            sometimes(iaa.Crop(percent=(0, 0.1))),
            sometimes(iaa.Affine(  # 部分图像做仿射变换
                scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # 图像缩放为80%到120%
                translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # 平移±20%
                rotate=(-30, 30),  # 旋转±30度
                shear=(-16, 16),  # 剪切变换±16度（矩形变平行四边形）
    cval =(0, 255),  # 全白全黑填充
           mode =ia.ALL  # 定义填充图像外区域的方法
    )),
     #  使用下面的0个到2个之间的方法增强图像
    iaa.SomeOf((0, 2),
               [
                   iaa.Sharpen(alpha=(0, 0.3), lightness=(0.9, 1.1)),  # 锐化处理
                   # 加入高斯噪声
                   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
                   iaa.Add((-10, 10), per_channel=0.5),  # 每个像素随机加减-10到10之间的数
                   iaa.Multiply((0.8, 1.2), per_channel=0.5),  # 像素乘上0.5或者1.5之间的数字
                   # 将整个图像的对比度变为原来的一半或者二倍
                   iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
               ],
               random_order=True)
    ],
    random_order = True  # 随机的顺序把这些操作用在图像上
    )

    images_aug = aug.augment_images(images)  # 应用数据增强

    n = 1
    for each in images_aug:
        # 保存到指定路径

        cv2.imwrite('C://Users//Lenovo//Desktop//insulator//%s%s%s.jpg' % (name, '.', n), each)
        n += 1

    ia.imshow(np.hstack(images_aug))  # 显示增强后的图像
