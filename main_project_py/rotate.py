# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.29                                        #
# Author : W-yt                                             #
# File   : rotate                                           #
# ######################################################### #
import cv2 as cv
import numpy as np

Train_1_Test_2 = 2

# Parameters Define
piecetype_english = ["1-hei-ju", "2-hei-zu", "3-hei-jiang", "4-hei-ma", "5-hei-pao", "6-hei-shi", "7-hei-xiang", "8-hong-bing", "9-hong-ju", "10-hong-ma", "11-hong-pao", "12-hong-shi", "13-hong-shuai", "14-hong-xiang"]

piece_roi_size = 50

for type in piecetype_english:
    # set input and output dir
    if Train_1_Test_2 == 1:
        image_src_dir = "origin_train_data/" + type + "/"
        image_dst_dir = "rotate_train_data/" + type + "/"
    else:
        image_src_dir = "origin_test_data/" + type + "/"
        image_dst_dir = "rotate_test_data/" + type + "/"

    all_image_count = 0
    file_image_count = 1

    if Train_1_Test_2 == 1:
        image_num = 50
    else:
        image_num = 20

    while file_image_count < (image_num + 1):
        input_image_name = image_src_dir + str(file_image_count) + ".jpg"

        # (in windows)can not use Chinese(if use, you need to change the coding scheme)
        src_image = cv.imread(input_image_name)
        # src_image = cv.imread("origin_train_data/1-hei-ju/1.jpg", cv.IMREAD_COLOR)
        cv.imshow("src_image", src_image)

        center = (piece_roi_size//2, piece_roi_size//2)

        rotate_angle = 0

        while rotate_angle < 360:
            # set output image name
            output_image_name = image_dst_dir + str(all_image_count) + ".jpg"

            # image rotate
            rotate_matrix = cv.getRotationMatrix2D(center, rotate_angle, 1.0)
            dst_image = cv.warpAffine(src_image, rotate_matrix, (piece_roi_size,piece_roi_size))
            cv.imshow("dst_image", dst_image)
            cv.imwrite(output_image_name, dst_image)

            # prepare for next loop
            all_image_count += 1
            rotate_angle += 4
            cv.waitKey(1)

        # prepare for next loop
        file_image_count += 1
        cv.waitKey(10)



