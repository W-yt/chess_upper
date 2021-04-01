# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.27                                        #
# Author : W-yt                                             #
# File   : piece                                            #
# ######################################################### #
import cv2 as cv
import numpy as np
from keras.models import load_model

class Piece(object):
    def __init__(self, modelfile_red, modelfile_black, piecetype_black, piecetype_red):
        self.model_red = load_model(modelfile_red)
        self.model_black = load_model(modelfile_black)
        print("cnn model load finish!")
        self.piecetype_black = piecetype_black
        self.piecetype_red = piecetype_red
        self.save_num = 1
        self.save_flag = 0
        self.circle_num = 0
        self.center_list = []
        self.type_list = []

    def piece_detect(self, src_image, min_x, max_x, min_y, max_y, blue_ksize,
                     hough_dp, hough_mindist, hough_param1, hough_param2, hough_minradius, hough_maxradius):
        self.src_image = src_image
        # cot off the chess board
        self.piece_image = self.src_image[min_y:max_y, min_x:max_x]
        self.piece_image_draw = self.piece_image.copy()
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
        # self.circle_num = 0
        # # (is not None) is different with (!= None)
        # if self.circles is not None:
        #     circles = np.uint16(np.around(self.circles))
        #     for circle in circles[0]:
        #         center_x, center_y, radius = circle
        #         center = (center_x, center_y)
        #         # draw the outer circle
        #         cv.circle(self.piece_image_draw, center, radius, (155,50,255), 2, 8, 0)
        #         # draw the center of the circle
        #         cv.circle(self.piece_image_draw, center, 3, (0,255,0), -1, 8, 0)
        #         print("center = ", center, "\t", "radius = ", radius)
        #         self.circle_num += 1
        # print("circle num : ", circle_num)
        # cv.imshow("piece_image_draw", self.piece_image_draw)


    def piece_predict(self, piece_roi_size, distance_edge, thresh_color, mid_square_size, red_black_thresh):
        self.circle_num = 0
        # if find any piece
        if self.circles is not None:
            circles = np.uint16(np.around(self.circles))
            for circle in circles[0]:
                self.circle_num += 1
                center_x, center_y, radius = circle
                center = (center_x, center_y)
                self.center_list.append(center)
                # this need change to int, because /2 may change num to float
                # notice this may be < 0
                piece_predict = self.piece_image[int(center_y-piece_roi_size/2):int(center_y+piece_roi_size/2),
                                              int(center_x-piece_roi_size/2):int(center_x+piece_roi_size/2)]
                # cv.imshow("piece_predict", piece_predict)

                # take the mid circle of piece image
                for pixel_x in range(piece_predict.shape[1]):
                    for pixel_y in range(piece_predict.shape[0]):
                        if (pow(pixel_x-piece_roi_size/2,2) + pow(pixel_y-piece_roi_size/2,2)) >= distance_edge:
                            piece_predict[pixel_y][pixel_x] = [0,0,0]
                # cv.imshow("piece_predict_mid", piece_predict)

                # split the three channel of piece image
                # choose the red channel for piece color detect
                # choose the green channel for piece type detect
                piece_predict_b, piece_predict_type, piece_predict_color = cv.split(piece_predict)
                # thresh and erode
                thresh, piece_predict_color = cv.threshold(piece_predict_color, thresh_color, 255, cv.THRESH_BINARY)
                element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
                piece_predict_color = cv.erode(piece_predict_color, element)
                # cv.imshow("piece_predict_color(red)", piece_predict_color)

                # change the predict image type
                piece_predict_type_array = np.array(piece_predict_type).astype("float32")/255.0
                # cnn model need a 4-dims array
                piece_predict_type_array = piece_predict_type_array.reshape(1,50,50,1)

                # judge the middle 10*10 square value
                min_edge = piece_roi_size//2 - mid_square_size//2
                max_edge = piece_roi_size//2 + mid_square_size//2
                square_color_sum = 0
                pixel_x = min_edge
                while pixel_x <= max_edge:
                    pixel_y = min_edge
                    while pixel_y <= max_edge:
                        square_color_sum += piece_predict_color[pixel_y][pixel_x]
                        pixel_y += 1
                    pixel_x += 1
                # print("square_color_sum : ", square_color_sum)
                # red piece
                if square_color_sum >= red_black_thresh:
                    # print("red piece!")
                    # predict the piece image
                    piece_prediction = self.model_red.predict(piece_predict_type_array)
                    probable_result = [result.argmax() for result in piece_prediction]
                    # set the piece id (between 1~14)(black piece:1~7)
                    piece_id = probable_result[0] + 7 + 1
                    # # get the predict type
                    # predict_type = self.piecetype_red[probable_result[0]]
                    # # get the predict probability
                    # predict_probability = piece_prediction[0][probable_result[0]]

                # black piece
                else:
                    # print("black piece!")
                    # predict the piece image
                    piece_prediction = self.model_black.predict(piece_predict_type_array)
                    probable_result = [result.argmax() for result in piece_prediction]
                    # set the piece id (between 1~14)(red piece:8~14)
                    piece_id = probable_result[0] + 1
                    # # get the predict type
                    # predict_type = self.piecetype_black[probable_result[0]]
                    # # get the predict probability
                    # predict_probability = piece_prediction[0][probable_result[0]]

                self.type_list.append(piece_id)
                cv.putText(self.piece_image_draw, str(piece_id), (center_x - 7, center_y - 20),
                           fontFace=cv.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=0.7, color=(255, 0, 0), thickness=2)
                # print("predict result : ", predict_type, end = "\t")
                # print("predict probability : ", format(predict_probability*100, '.2f'), "%")

            cv.imshow("piece_image_draw", self.piece_image_draw)

    def piece_locate(self):
        pass












