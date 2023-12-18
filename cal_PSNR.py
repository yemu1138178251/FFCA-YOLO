import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

input_path_ori = os.path.join("D:\\yemu\\code\\python\\yolov5-prune-light\\yolov5-prune\\datasets\\vehicle\\images\\val2017")
input_path_compare = "D:\\yemu\\code\\python\\yolov5-prune-light\\yolov5-prune\\datasets\\vehicle\\images\\fog0.2\\"

img_name = sorted(glob.glob(os.path.join(input_path_ori, "*.jpg")))

result_psnr = 0
zongshu = 0

for i, im in enumerate(img_name):

    file_name = os.path.basename(im)
    print(file_name)

    img_ori = cv2.imread(im)
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    img_de = cv2.imread(input_path_compare + file_name)
    img_gray2 = cv2.cvtColor(img_de, cv2.COLOR_BGR2GRAY)


    result_psnr += psnr(img_gray, img_gray2)
    # cv2.imwrite(output_path + im,img_)

print("psnr:", result_psnr / len(img_name))