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

# Parameters Define
chess_grid_maxnum = 90

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
            if len(board_object.angular_point) == chess_grid_maxnum:
                break
            c = cv.waitKey(30)

    if(BOARD_DETECT):
        board_object.grid_tag(chess_grid_rows = 9, chess_grid_cols = 10)



    cv.waitKey(0)

if __name__ == "__main__":
    MAIN()