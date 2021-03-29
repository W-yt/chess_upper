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
    def __init__(self, modelfile, piecetype):
        self.model = load_model(modelfile)
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
                x, y, radius = circle
                center = (x, y)
                # draw the outer circle
                cv.circle(piece_image_draw, center, radius, (155,50,255), 2, 8, 0)
                # draw the center of the circle
                cv.circle(piece_image_draw, center, 3, (0,255,0), -1, 8, 0)
                # print("center = ", center, "\t", "radius = ", radius)
                circle_num += 1
        print("circle num : ", circle_num)
        cv.imshow("piece_image_draw", piece_image_draw)


    def piece_save(self, piece_roi_size, distance_edge, save_dir):
        # cut off each piece
        if self.circles is not None:
            circles = np.uint16(np.around(self.circles))
            for circle in circles[0]:
                center_x, center_y, radius = circle
                center = (center_x, center_y)
                # this need change to int, because /2 may change num to float
                piece_save = self.piece_image[int(center_y-piece_roi_size/2):int(center_y+piece_roi_size/2),
                                              int(center_x-piece_roi_size/2):int(center_x+piece_roi_size/2)]
                # cv.imshow("piece_save", piece_save)

                # take the mid circle of piece image
                for pixel_x in range(piece_save.shape[1]):
                    for pixel_y in range(piece_save.shape[0]):
                        if (pow(pixel_x-piece_roi_size/2,2) + pow(pixel_y-piece_roi_size/2,2)) >= distance_edge:
                            piece_save[pixel_y][pixel_x] = [0,0,0]
                cv.imshow("piece_save_mid", piece_save)

                # save image to stated directory
                if self.save_flag == 1:
                    save_fullpath = save_dir + str(self.save_num) + ".jpg"
                    self.save_num += 1
                    cv.imwrite(save_fullpath, piece_save)
                    self.save_flag = 0


    def piece_predict(self, piece_roi_size):
        # if find any piece
        if self.circles is not None:
            circles = np.uint16(np.around(self.circles))
            for circle in circles[0]:
                center_x, center_y, radius = circle
                center = (center_x, center_y)
                # this need change to int, because /2 may change num to float
                piece_predict = self.piece_image[int(center_y-piece_roi_size/2):int(center_y+piece_roi_size/2),
                                              int(center_x-piece_roi_size/2):int(center_x+piece_roi_size/2)]
                cv.imshow("piece_predict", piece_predict)

                # take the mid circle of piece image
                for pixel_x in range(piece_predict.shape[1]):
                    for pixel_y in range(piece_predict.shape[0]):
                        if (pow(pixel_x-piece_roi_size/2,2) + pow(pixel_y-piece_roi_size/2,2)) >= distance_edge:
                            piece_predict[pixel_y][pixel_x] = [255,255,255]
                cv.imshow("piece_predict_mid", piece_predict)

                # change the predict image type
                piece_predict_array = np.array(piece_predict).astype("float32")/255.0

                # predict the piece image
                piece_prediction = self.model.predict(piece_predict_array)
                probable_result = [result.argmax() for result in piece_prediction]
                predict_type = self.piecetype[probable_result[0]]
                predict_probability = piece_prediction[0][probable_result[0]]
                predict_text = predict_type + str(predict_probability)
                cv.putText(self.piece_image, predict_text, center, cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,0,0))




