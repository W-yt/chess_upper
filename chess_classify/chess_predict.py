from keras.models import load_model
import matplotlib.image as processimage
import numpy as np
from PIL import Image
import os


class Prediction(object):
    def __init__(self,ModelFile,PredictDir,PieceType):
        self.model_file = ModelFile
        self.predict_dir = PredictDir
        self.PieceType = PieceType

    def Predict(self):

        model = load_model(self.model_file)

        type_counter = 0
        for type in self.PieceType:
            subfolder = os.listdir(self.predict_dir + type)
            file_counter = 1
            for subclass in subfolder:
                # # Resize
                # img_open = Image.open(self.predict_dir + type + "/" + str(subclass))
                # resieze_image = img_open.resize((50,50),Image.BILINEAR)
                # resieze_image.save(self.predict_dir + type + "/" + str(subclass))
                # load image
                image = processimage.imread(self.predict_dir + type + "/" + str(subclass))
                image_to_array = np.array(image).astype("float32") / 255.0
                image_to_array = image_to_array.reshape(1, 50, 50, 3)
                # Predict image
                prediction = model.predict(image_to_array)
                Final_Pred = [result.argmax() for result in prediction]
                print(self.PieceType[type_counter], "num", file_counter, "file-->","预测结果:", self.PieceType[Final_Pred[0]])
                file_counter += 1
                if PieceType[Final_Pred[0]] == type:
                    CorrectCount[type_counter] += 1
            type_counter += 1

        # Print the predict accuracy rate
        type_counter = 0
        for type in self.PieceType:
            print(self.PieceType[type_counter], "预测准确率为", CorrectCount[type_counter], "/", 1800, " = ", CorrectCount[type_counter]/18.0, "%")
            type_counter += 1

        # # Display the probability percent
        # count = 0
        # for i in prediction[0]:
        #     percent = "%.2f%%"%(i*100)
        #     print(self.PieceType[count],"概率",percent)
        #     count += 1
        # print("预测结果：",PieceType[Final_Pred[0]])

    def ShowPredImg(self):
        pass

# 爆内存的时候，测试模型的可行性时用的
# PieceType = ["黑-卒", "红-兵", "黑-将", "红-帥"]

PieceType = ["1-黑-車", "2-黑-卒", "3-黑-将", "4-黑-马", "5-黑-炮", "6-黑-士", "7-黑-象",
             "8-红-兵", "9-红-車", "10-红-马","11-红-炮","12-红-仕","13-红-帥","14-红-相"]

CorrectCount = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
FileCount = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

Pred = Prediction(PredictDir = "旋转_每种1800张/",ModelFile = "piecefinder.h5",PieceType = PieceType)
Pred.Predict()

