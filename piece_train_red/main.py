# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.26                                        #
# Author : W-yt                                             #
# File   : main                                             #
# ######################################################### #

import board
import piece
import cv2 as cv

# Project Model Switch
CAMERA_ADJUST = 0
BOARD_DETECT  = 0
PIECE_DETECT  = 1
PIECE_SAVE    = 1
PIECE_PREDICT = 0

# Parameters Define
piecetype_chinese = ["1-红-兵", "2-红-車", "3-红-马", "4-红-炮", "5-红-仕", "6-红-帥", "7-红-相"]

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

    # create the object
    board_object = board.Board()
    piece_object = piece.Piece(modelfile = "piece_finder_red_old.h5", piecetype = piecetype_chinese)

    # function part for board
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        # take the middle square picture
        src_image = frame[0:720, 280:1000]
        # cv.imshow("src_image", src_image)

        if(CAMERA_ADJUST):
            c = cv.waitKey(30)

        if(BOARD_DETECT):
            board_object.border_detect(src_image = src_image, binary_edge = 90)
            board_object.grid_detect(canny_threshold1=30, canny_threshold2=300,
                                     hough_threshold=48, hough_minlength=500, hough_maxgap=80,
                                     harris_blocksize=2, harris_ksize=3, harris_k=0.04, harris_thresh=175)
            if len(board_object.angular_point) == 90:
                break
            c = cv.waitKey(30)
        else:
            break

    if(BOARD_DETECT):
        board_object.grid_tag(chess_grid_rows = 9, chess_grid_cols = 10)

    # after function for board you need place the piece on board and press the keyboard
    # cv.waitKey(0)

    # function part for piece
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        # take the middle square picture
        src_image = frame[0:720, 280:1000]

        if(PIECE_DETECT):
            if(BOARD_DETECT):
                piece_object.piece_detect(src_image=src_image,
                                          min_x=board_object.min_x, max_x=board_object.max_x, min_y=board_object.min_y,
                                          max_y=board_object.max_y,
                                          blue_ksize=3,
                                          hough_dp=1, hough_mindist=50, hough_param1=100, hough_param2=15,
                                          hough_minradius=25, hough_maxradius=26)
            else:
                piece_object.piece_detect(src_image=src_image,
                                          min_x=0, max_x=719, min_y=0, max_y=719,
                                          blue_ksize=3,
                                          hough_dp=1, hough_mindist=50, hough_param1=100, hough_param2=15,
                                          hough_minradius=25, hough_maxradius=26)
            if (PIECE_SAVE):
                # draw the area all black out of the circle with 17 pixel radius
                piece_object.piece_save(piece_roi_size=50, distance_edge=400, save_dir="temp_save_dir/")

            if (PIECE_PREDICT):
                piece_object.piece_predict(piece_roi_size=50, distance_edge=400)

        keyboard = cv.waitKey(30)

        if(PIECE_SAVE):
            # press enter take one image
            if keyboard == 13:
               piece_object.save_flag = 1
            # press space reset the save image num
            if keyboard == 32:
                piece_object.save_num = 1


    # cv.waitKey(0)

if __name__ == "__main__":
    MAIN()