# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.26                                        #
# Author : W-yt                                             #
# File   : vision                                           #
# ######################################################### #

import time
import board
import piece
import cv2 as cv
from PyQt5.QtCore import *

# Project Model Switch
CAMERA_ADJUST = 0
BOARD_DETECT  = 1
PIECE_DETECT  = 1
PIECE_PREDICT = 1

# Parameters Define
piecetype_chinese = ["1-黑-車", "2-黑-卒", "3-黑-将", "4-黑-马", "5-黑-炮", "6-黑-士", "7-黑-象",
                     "8-红-兵", "9-红-車", "10-红-马", "11-红-炮", "12-红-仕", "13-红-帥", "14-红-相"]
piecetype_chinese_black = ["1-黑-車", "2-黑-卒", "3-黑-将", "4-黑-马", "5-黑-炮", "6-黑-士", "7-黑-象"]
piecetype_chinese_red = ["8-红-兵", "9-红-車", "10-红-马", "11-红-炮", "12-红-仕", "13-红-帥", "14-红-相"]


# vision thread
class VersionThread(QThread):
    # define board detect finish signal
    board_signal = pyqtSignal()
    # define piece detect finish signal
    piece_signal = pyqtSignal()

    def run(self):
        # 0 means computer camera
        # 1 means external camera
        capture = cv.VideoCapture(0)
        if not capture.isOpened():
            print("Camera open failed!")
        # default camera resolution ratio
        # print("camera width : ", capture.get(cv.CAP_PROP_FRAME_WIDTH))
        # print("camera height: ", capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        # set the camera to correct resolution ratio
        capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        print("camera width : ", capture.get(cv.CAP_PROP_FRAME_WIDTH))
        print("camera height: ", capture.get(cv.CAP_PROP_FRAME_HEIGHT))

        # create the object
        board_object = board.Board()
        piece_object = piece.Piece(modelfile_red="../piece_train_red/piece_finder_red.h5",
                                   modelfile_black="../piece_train_black/piece_finder_black.h5",
                                   piecetype_black=piecetype_chinese_black,
                                   piecetype_red=piecetype_chinese_red)

        # function part for board
        while True:
            ret, frame = capture.read()
            frame = cv.flip(frame, 1)
            # take the middle square picture
            src_image = frame[0:720, 280:1000]
            # cv.imshow("src_image", src_image)

            if (CAMERA_ADJUST):
                c = cv.waitKey(30)

            if (BOARD_DETECT):
                board_object.border_detect(src_image=src_image, binary_edge=90)
                board_object.grid_detect(canny_threshold1=100, canny_threshold2=350,
                                         hough_threshold=50, hough_minlength=400, hough_maxgap=60,
                                         harris_blocksize=2, harris_ksize=3, harris_k=0.04, harris_thresh=175)
                if len(board_object.angular_point) == 90:
                    break
                c = cv.waitKey(30)
            else:
                break

        if (BOARD_DETECT):
            board_object.grid_tag(chess_grid_rows=9, chess_grid_cols=10)

            # send the board detect finish signal
            self.board_signal.emit()

            # after function for board you need place the piece on board and press the keyboard
            cv.waitKey(0)

        # function part for piece
        while True:
            # get loop begin time
            begin_time = round(time.time() * 1000)

            # get camera image
            ret, frame = capture.read()
            frame = cv.flip(frame, 1)
            # take the middle square picture
            src_image = frame[0:720, 280:1000]

            if (PIECE_DETECT):
                if (BOARD_DETECT):
                    piece_object.piece_detect(src_image=src_image,
                                              min_x=board_object.min_x, max_x=board_object.max_x,
                                              min_y=board_object.min_y, max_y=board_object.max_y,
                                              blue_ksize=3,
                                              hough_dp=1, hough_mindist=40, hough_param1=100, hough_param2=20,
                                              hough_minradius=21, hough_maxradius=22)
                else:
                    piece_object.piece_detect(src_image=src_image,
                                              min_x=65, max_x=638, min_y=83, max_y=646,
                                              blue_ksize=3,
                                              hough_dp=1, hough_mindist=40, hough_param1=100, hough_param2=20,
                                              hough_minradius=21, hough_maxradius=22)

                if (PIECE_PREDICT):
                    piece_object.piece_predict(piece_roi_size=50, distance_edge=289, thresh_color=90,
                                               mid_square_size=10, red_black_thresh=50 * 255)
                    piece_object.piece_locate(angular_point_sorted=board_object.angular_point_rows, chess_grid_rows=9,
                                              chess_grid_cols=10)

            keyboard = cv.waitKey(3)

            # get the loop end time and calculate the fps
            end_time = round(time.time() * 1000)
            loop_time = end_time - begin_time
            fps = 1000 / loop_time
            print("fps :", format(fps, '.2f'))

            # send piece detect finish signal
            self.piece_signal.emit()

        # cv.waitKey(0)

