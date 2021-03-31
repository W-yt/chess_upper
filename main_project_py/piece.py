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
    def __init__(self, modelfile_red, modelfile_black, piecetype):
        self.model_red = load_model(modelfile_red)
        self.model_black = load_model(modelfile_black)
        print("cnn model load finish!")
        self.piecetype = piecetype
        self.save_num = 1
        self.save_flag = 0


    def piece_detect(self, src_image, min_x, max_x, min_y, max_y, blue_ksize,
                     hough_dp, hough_mindist, hough_param1, hough_param2, hough_minradius, hough_maxradius):
        self.src_image = src_image
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
                center_x, center_y, radius = circle
                center = (center_x, center_y)
                # draw the outer circle
                cv.circle(piece_image_draw, center, radius, (155,50,255), 2, 8, 0)
                # draw the center of the circle
                cv.circle(piece_image_draw, center, 3, (0,255,0), -1, 8, 0)
                # print("center = ", center, "\t", "radius = ", radius)
                circle_num += 1
        # print("circle num : ", circle_num)
        cv.imshow("piece_image_draw", piece_image_draw)


    def piece_predict(self, piece_roi_size, distance_edge, thresh_color, mid_square_size, black_red_thresh):
        # if find any piece
        if self.circles is not None:
            circles = np.uint16(np.around(self.circles))
            for circle in circles[0]:
                center_x, center_y, radius = circle
                center = (center_x, center_y)
                # this need change to int, because /2 may change num to float
                piece_predict = self.piece_image[int(center_y-piece_roi_size/2):int(center_y+piece_roi_size/2),
                                              int(center_x-piece_roi_size/2):int(center_x+piece_roi_size/2)]
                # cv.imshow("piece_predict", piece_predict)

                # take the mid circle of piece image
                for pixel_x in range(piece_predict.shape[1]):
                    for pixel_y in range(piece_predict.shape[0]):
                        if (pow(pixel_x-piece_roi_size/2,2) + pow(pixel_y-piece_roi_size/2,2)) >= distance_edge:
                            piece_predict[pixel_y][pixel_x] = [0,0,0]
                cv.imshow("piece_predict_mid", piece_predict)

                # choose the red channel
                piece_predict_b, piece_predict_g, piece_predict_r = cv.split(piece_predict)
                # thresh and erode
                thresh, piece_predict_process = cv.threshold(piece_predict_r, thresh_color, 255,cv.THRESH_BINARY)
                element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
                piece_predict_process = cv.erode(piece_predict_process, element)
                # piece_predict_process = cv.erode(piece_predict_process, element)
                cv.imshow("piece_predict_red_channel", piece_predict_process)

                # judge the middle 10*10 square value
                min_edge = piece_roi_size//2 - mid_square_size//2
                max_edge = piece_roi_size//2 + mid_square_size//2
                square_color_sum = 0
                pixel_x = min_edge
                while pixel_x <= max_edge:
                    pixel_y = min_edge
                    while pixel_y <= max_edge:
                        square_color_sum += piece_predict_process[pixel_y][pixel_x]
                        pixel_y += 1
                    pixel_x += 1
                if square_color_sum >= black_red_thresh:
                    print("black piece!")
                else:
                    print("red piece!")

                # # change the predict image type
                # piece_predict_array = np.array(piece_predict).astype("float32")/255.0
                # # cnn model need a 4-dims array
                # piece_predict_array = piece_predict_array.reshape(1,50,50,3)
                #
                # # predict the piece image
                # piece_prediction = self.model.predict(piece_predict_array)
                # probable_result = [result.argmax() for result in piece_prediction]
                # predict_type = self.piecetype[probable_result[0]]
                # predict_probability = piece_prediction[0][probable_result[0]]
                # predict_text = predict_type + str(predict_probability)
                # cv.putText(self.piece_image, predict_text, center, cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,0,0))
                # print("predict result : ", predict_type)




