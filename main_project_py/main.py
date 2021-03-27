# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.26                                        #
# Author : W-yt                                             #
# File   : main                                             #
# ######################################################### #

import board
import piece
import rotate
import train
import predict
import cv2 as cv

# Project Model Switch
CAMERA_ADJUST = 0
BOARD_DETECT  = 1
PIECE_DETECT  = 1
PIECE_SAVE    = 1

# Parameters Define

def MAIN():
    print("Project Start!")
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

    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        # take the middle square picture
        src_image = frame[0:720, 280:1000]
        # cv.imshow("src_image", src_image)

        if(CAMERA_ADJUST):
            c = cv.waitKey(30)

        if(BOARD_DETECT):
            board_object = board.Board(src_image = src_image)
            board_object.border_detect(binary_edge = 90)
            board_object.grid_detect(canny_threshold1 = 100, canny_threshold2 = 350,
                                     hough_threshold = 50, hough_minlength = 400, hough_maxgap = 60,
                                     harris_blocksize = 2, harris_ksize = 3, harris_k = 0.04, harris_thresh = 175)
            if len(board_object.angular_point) == 90:
                break
            c = cv.waitKey(30)

    if(BOARD_DETECT):
        board_object.grid_tag(chess_grid_rows = 9, chess_grid_cols = 10)

    cv.waitKey(0)

    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        # take the middle square picture
        src_image = frame[0:720, 280:1000]

        if(PIECE_DETECT):
            piece_object = piece.Piece(src_image = src_image)
            if(BOARD_DETECT):
                piece_object.piece_detect(min_x = board_object.min_x, max_x = board_object.max_x, min_y = board_object.min_y, max_y = board_object.max_y,
                                          blue_ksize = 3,
                                          hough_dp = 1, hough_mindist = 40, hough_param1 = 100, hough_param2 = 20, hough_minradius = 18, hough_maxradius = 21)
            else:
                piece_object.piece_detect(min_x = 65.4, max_x = 637.9, min_y = 82.6, max_y = 646.0,
                                          blue_ksize = 3,
                                          hough_dp = 1, hough_mindist = 40, hough_param1 = 100, hough_param2 = 20, hough_minradius = 18, hough_maxradius = 21)
            if(PIECE_SAVE):
                piece_object.piece_save(piece_roi_size = 50)

        cv.waitKey(30)

    # cv.waitKey(0)

if __name__ == "__main__":
    MAIN()