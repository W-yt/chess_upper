# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.26                                        #
# Author : W-yt                                             #
# File   : board                                            #
# ######################################################### #

import cv2 as cv
import numpy as np

class Board(object):
    def __init__(self):
        self.max_contour_area = 0
        self.max_contour_index = 0
        print("board model begin!")


    def border_detect(self, src_image, binary_edge):
        self.src_image = src_image
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
        self.max_contour_area = 0
        draw_image = np.zeros(self.src_image.shape, np.uint8)
        if contours is not None:
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
            self.min_x = min(point_x)
            self.max_x = max(point_x)
            self.min_y = min(point_y)
            self.max_y = max(point_y)
            self.board_image = self.src_image[self.min_y:self.max_y, self.min_x:self.max_x]
            # cv.imshow("board_image", self.board_image)

            # # find the real board and then jump out the loop
            # if(self.max_contour_area > 250000 and self.max_contour_area < 330000):


    def grid_detect(self, canny_threshold1, canny_threshold2,
                    hough_threshold, hough_minlength, hough_maxgap,
                    harris_blocksize, harris_ksize, harris_k, harris_thresh):
        # find the real board and then begin the grid detect
        if self.max_contour_area > 250000 and self.max_contour_area < 330000:
            # board image canny
            board_image_gray = cv.cvtColor(self.board_image, cv.COLOR_BGR2GRAY)
            # cv.imshow("gray_image", board_image_gray)
            canny_image = cv.Canny(board_image_gray, canny_threshold1, canny_threshold2)
            # thresh, canny_image = cv.threshold(canny_image, 128, 255, cv.THRESH_BINARY)
            # cv.imshow("canny_image", canny_image)

            # morphlogy tranform
            element = np.ones((5, 5), np.uint8)
            canny_image = cv.morphologyEx(canny_image, cv.MORPH_CLOSE, element)
            # cv.imshow("canny_image", canny_image)

            # hough detect the lines
            blank_board_image = np.ones(board_image_gray.shape, np.uint8)
            lines = cv.HoughLinesP(canny_image, 1, np.pi / 180, hough_threshold, minLineLength = hough_minlength, maxLineGap = hough_maxgap)
            # (is not None) is different with (!= None)
            if lines is not None:
                for line in lines:
                    # print(type(line))
                    x1, y1, x2, y2 = line[0]
                    cv.line(blank_board_image, (x1, y1), (x2, y2), 255, 1)
            # cv.imshow("blank_board_image",blank_board_image)

            # angular point detect
            self.angular_point = []
            harris_image = cv.cornerHarris(blank_board_image, harris_blocksize, harris_ksize, harris_k)
            harris_image_show = self.board_image.copy()
            # pay attention to the image type
            harris_image_normal = np.empty(harris_image.shape, dtype = np.float32)
            cv.normalize(harris_image, harris_image_normal, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX)
            for i in range(harris_image_normal.shape[0]):
                for j in range(harris_image_normal.shape[1]):
                    if int(harris_image_normal[i, j]) > harris_thresh:
                        cv.circle(harris_image_show, (j, i), 5, (255,255,0), 1, 8)
                        self.angular_point.append((j,i))
            # cv.imshow("harris_image_show", harris_image_show)
            print("(all)angular point num = ", len(self.angular_point))

            # delete the edge point
            # notice : list loop delete element has two method(below method or inverted order loop)
            point_index = 0
            while point_index < len(self.angular_point):
                if self.angular_point[point_index][1] < 40 or self.angular_point[point_index][1] > (self.board_image.shape[0]-40) or self.angular_point[point_index][0] < 20 or self.angular_point[point_index][0] > (self.board_image.shape[1]-20):
                    del self.angular_point[point_index]
                else:
                    point_index += 1
            # print("(delete edge)angular point num = ", len(self.angular_point))
            # for one_point in self.angular_point:
            #     # one point[0] mean col
            #     # self.board_image.shape[0] means raw
            #     if one_point[1] < 40 or one_point[1] > (self.board_image.shape[0]-40) or one_point[0] < 20 or one_point[0] > (self.board_image.shape[1]-20):
            #         self.angular_point.remove(one_point)
            #         print("delete one edge point, the num of point is ", len(self.angular_point))

            # delete the coincide point
            point_index1 = 0
            while point_index1 < len(self.angular_point):
                point_index2 = point_index1 + 1
                while point_index2 < len(self.angular_point):
                    if ((abs(self.angular_point[point_index1][0]-self.angular_point[point_index2][0]) + abs(self.angular_point[point_index1][1]-self.angular_point[point_index2][1])) < 50):
                        del self.angular_point[point_index2]
                    else:
                        point_index2 += 1
                point_index1 += 1
            # print("(delete coincide)angular point num = ", len(self.angular_point))
            print("(filtrate)angular point num = ", len(self.angular_point))


    def grid_tag(self, chess_grid_rows, chess_grid_cols):
        # draw the point after filtrate
        # harris_image_filtrate = self.board_image.copy()
        for one_point in self.angular_point:
            cv.circle(self.board_image, (one_point[0],one_point[1]), 4, (255,255,255), 2, 8)
        # cv.imshow("harris_image_filtrate", self.board_image)

        # sort the 90 chess board points(self.angular_point sort by cols default)
        self.angular_point_rows = []
        row_index = 0
        while row_index < chess_grid_rows:
            col_index = 0
            # two dims list must first append this[]
            self.angular_point_rows.append([])
            while col_index < chess_grid_cols:
                self.angular_point_rows[row_index].append(self.angular_point[chess_grid_cols*row_index + col_index])
                col_index += 1
            # angular_point_rows[row_index] = self.angular_point[row_index*chess_grid_cols : ((row_index+1)*chess_grid_cols)]
            row_index += 1
        print("grid group finish!")

        row_index = 0
        for row_index in range(len(self.angular_point_rows)):
            self.angular_point_rows[row_index].sort(key = self.takeRow)
        print("grid sort finish!")

        # display the grid point tag
        row_index = 0
        while row_index < chess_grid_rows:
            col_index = 0
            while col_index < chess_grid_cols:
                text_tag = "(" + str(row_index) + "," + str(col_index) + ")"
                text_coord = (self.angular_point_rows[row_index][col_index][0]-24, self.angular_point_rows[row_index][col_index][1]-10)
                cv.putText(self.board_image, text_tag, text_coord, cv.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))
                col_index += 1
            row_index += 1
        cv.imshow("board tag image",self.board_image)
        print("board tag finish!")

    def takeRow(self,elem):
        return elem[0]






