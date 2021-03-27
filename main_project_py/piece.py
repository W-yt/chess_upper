# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.27                                        #
# Author : W-yt                                             #
# File   : piece                                            #
# ######################################################### #
import cv2 as cv
import numpy as np

class Piece(object):
    def __init__(self, src_image):
        self.src_image = src_image


    def piece_detect(self, min_x, max_x, min_y, max_y, blue_ksize,
                     hough_dp, hough_mindist, hough_param1, hough_param2, hough_minradius, hough_maxradius):
        # cot off the chess board
        self.piece_image = self.src_image[min_y:max_y, min_x:max_x]
        piece_image_draw = self.piece_image.copy()
        # cv.imshow("piece image", self.piece_image)

        # image enhancement
        piece_image_b, piece_image_g, piece_image_r = cv.split(self.piece_image)
        # cv.imshow("piece_image_g", piece_image_g)
        thresh, piece_image_threshold = cv.threshold(piece_image_g, 128, 255, cv.THRESH_BINARY)
        # blue function's ksize is different with cornerHarris function
        piece_image_blue = cv.blur(piece_image_threshold, (blue_ksize,blue_ksize))
        # cv.imshow("piece_image_blue", piece_image_blue)

        # detect the circle of piece
        self.circles = cv.HoughCircles(piece_image_blue, cv.HOUGH_GRADIENT,
                                  dp = hough_dp, minDist = hough_mindist,
                                  param1 = hough_param1, param2 = hough_param2,
                                  minRadius = hough_minradius, maxRadius = hough_maxradius)
        circle_num = 0
        # (is not None) is different with (!= None)
        if self.circles is not None:
            circles = np.uint16(np.around(self.circles))
            for circle in circles[0]:
                x, y, radius = circle
                center = (x, y)
                # draw the outer circle
                cv.circle(piece_image_draw, center, radius, (155,50,255), 2, 8, 0)
                # draw the center of the circle
                cv.circle(piece_image_draw, center, 3, (0,255,0), -1, 8, 0)
                # print("center = ", center, "   ", "radius = ", radius)
                circle_num += 1
        print("circle num : ", circle_num)
        cv.imshow("piece_image_draw", piece_image_draw)


    def piece_save(self, piece_roi_size):
        # cut off each piece
        if self.circles is not None:
            circles = np.uint16(np.around(self.circles))
            for circle in circles[0]:
                center_x, center_y, radius = circle
                center = (center_x, center_y)
                # this need change to int, because /2 may change num to float
                piece_save = self.piece_image[int(center_y-piece_roi_size/2):int(center_y+piece_roi_size/2),
                                              int(center_x-piece_roi_size/2):int(center_x+piece_roi_size/2)]
                cv.imshow("piece_save", piece_save)

                # above tested
                ##############################################
                # below no test



