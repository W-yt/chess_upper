# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.29                                        #
# Author : W-yt                                             #
# File   : roate                                            #
# ######################################################### #
import cv2 as cv
import numpy as np

# Parameters Define
piecetype = ["1-黑-車", "2-黑-卒", "3-黑-将", "4-黑-马", "5-黑-炮", "6-黑-士", "7-黑-象",
             "8-红-兵", "9-红-車", "10-红-马","11-红-炮","12-红-仕","13-红-帥","14-红-相"]

piece_roi_size = 50

for type in piecetype:
    # set input and output dir
    image_src_dir = "origin_train_data/" + type
    image_dst_dir = "rotate_train_data/" + type

    all_image_count = 0
    file_image_count = 1

    while file_image_count < 51:
        input_image_name = image_src_dir + str(file_image_count) + ".jpg"

        src_image = cv.imread(input_image_name)
        cv.imshow("src_image",src_image)

        center = (piece_roi_size//2, piece_roi_size//2)

        rotate_angle = 0

        while rotate_angle < 360:
            # set output image name
            output_image_name = image_dst_dir + str(all_image_count) + ".jpg"

            # image rotate
            rotate_matrix = cv.getRotationMatrix2D(center, rotate_angle, 1.0)
            dst_image = cv.warpAffine(src_image, rotate_matrix, (piece_roi_size,piece_roi_size))
            cv.imshow("dst_image", dst_image)
            # cv.imwrite(output_image_name, dst_image)

            # prepare for next loop
            all_image_count += 1
            rotate_angle += 4
            cv.waitKey(4)

        # prepare for next loop
        file_image_count += 1
        cv.waitKey(50)



