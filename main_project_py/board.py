# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.26                                        #
# Author : W-yt                                             #
# File   : board                                            #
# ######################################################### #
import cv2 as cv
import numpy as np

class Board(object):
    def __init__(self, src_image):
        self.src_image = src_image
        self.max_contour_area = 0
        self.max_contour_index = 0

    def border_detect(self, binary_edge):
        # channels split and extract the yellow (BRG ==> G channel - B channel)
        src_image_b, src_image_g, src_image_r = cv.split(self.src_image)
        yellow_image = cv.subtract(src_image_g, src_image_b)
        # cv.imshow("yellow_image", yellow_image)

        # picture binarization
        thresh, binary_image = cv.threshold(yellow_image, binary_edge, 255, cv.THRESH_BINARY)
        # cv.imshow("binary_image", binary_image)

        # morphlogy tranform
        element = np.ones((5, 5), np.uint8)
        morph_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, element)
        # cv.imshow("morph_image",morph_image)

        # find contours (want only one contour -> RETR_EXTRENAL, want more contours -> RETR_TREE)
        contours, hierarchy = cv.findContours(morph_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # find max contours and draw contours
        draw_image = np.zeros(self.src_image.shape, np.uint8)
        if(len(contours) > 0):
            for i in range(len(contours)):
                contour_area = cv.contourArea(contours[i])
                if contour_area > self.max_contour_area:
                    self.max_contour_area = contour_area
                    self.max_contour_index = i
                cv.drawContours(draw_image, contours, i, (255,0,0), 1, 8)
            max_contour = contours[self.max_contour_index]
            cv.drawContours(draw_image, contours, self.max_contour_index, (0,0,255), 1, 8)

            # find and draw the board border
            board_border_rect = cv.minAreaRect(max_contour)
            board_border_point = cv.boxPoints(board_border_rect)
            board_border_point = np.int0(board_border_point)
            cv.drawContours(draw_image, [board_border_point], 0, (0,255,0), 2)
            # cv.imshow("draw_image", draw_image)

            # cot off the chess board
            point_x = [i[0] for i in board_border_point]
            point_y = [i[1] for i in board_border_point]
            min_x = min(point_x)
            max_x = max(point_x)
            min_y = min(point_y)
            max_y = max(point_y)
            self.board_image = self.src_image[min_y:max_y, min_x:max_x]
            # cv.imshow("board_image", self.board_image)

            # # find the real board and then jump out the loop
            # if(self.max_contour_area > 250000 and self.max_contour_area < 330000):

    def grid_detect(self, canny_threshold1, canny_threshold2,
                    hough_threshold, hough_minlength, hough_maxgap,
                    harris_blocksize, harris_ksize, harris_k, harris_thresh):
        # board image canny
        board_image_gray = cv.cvtColor(self.board_image, cv.COLOR_BGR2GRAY)
        # cv.imshow("gray_image", board_image_gray)
        canny_image = cv.Canny(board_image_gray, canny_threshold1, canny_threshold2)
        # thresh, canny_image = cv.threshold(canny_image, 128, 255, cv.THRESH_BINARY)
        # cv.imshow("canny_image", canny_image)

        # morphlogy tranform
        element = np.ones((5, 5), np.uint8)
        canny_image = cv.morphologyEx(canny_image, cv.MORPH_CLOSE, element)
        cv.imshow("canny_image", canny_image)

        # hough detect the lines
        blank_board_image = np.ones(board_image_gray.shape, np.uint8)
        lines = cv.HoughLinesP(canny_image, 1, np.pi / 180, hough_threshold, minLineLength = hough_minlength, maxLineGap = hough_maxgap)
        # (is not None) is different with (!= None)
        if lines is not None:
            for line in lines:
                # print(type(line))
                x1, y1, x2, y2 = line[0]
                cv.line(blank_board_image, (x1, y1), (x2, y2), 255, 1)
        cv.imshow("blank_board_image",blank_board_image)

        # angular point detect
        harris_image = cv.cornerHarris(blank_board_image, harris_blocksize, harris_ksize, harris_k)
        # harris_image_show = self.board_image.copy()
        harris_image_show = np.zeros(self.board_image.shape, dtype = np.uint8)
        harris_image_normal = np.zeros(harris_image.shape, dtype = np.uint8)
        cv.normalize(harris_image, harris_image_normal, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX)
        for i in range(harris_image_normal.shape[0]):
            for j in range(harris_image_normal.shape[1]):
                if int(harris_image_normal[i, j]) > harris_thresh:
                    cv.circle(harris_image_show, (j, i), 10, (255,255,0), 3, 8)
        cv.imshow("harris_image_show", harris_image_show)
