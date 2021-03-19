from keras.models import load_model
import matplotlib.image as processimage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Prediction(object):
    def __init__(self,ModelFile,PredictFile,PieceType):
        self.model_file = ModelFile
        self.predict_file = PredictFile
        self.PieceType = PieceType

    def Predict(self):
        model = load_model(self.model_file)

        # Deal the image's shape
        image = processimage.imread(self.predict_file)

        image_to_array = np.array(image).astype("float32")/255.0
        image_to_array = image_to_array.reshape(1,200,200,3)
        print("image reshape finish!")

        # Predict the image
        prediction = model.predict(image_to_array)
        Final_Pred = [result.argmax() for result in prediction]
        print(Final_Pred)
        # print(prediction)
        # print(prediction[0])

        # Display the probability percent
        count = 0
        for i in prediction[0]:
            percent = "%.2f%%"%(i*100)
            print(self.PieceType[count],"概率",percent)
            count += 1

    def ShowPredImg(self):
        pass


PieceType = ["黑-卒", "红-兵", "黑-将", "红-帥"]

Pred = Prediction(PredictFile = "测试数据目录/11.jpg",ModelFile = "piecefinder.h5",PieceType = PieceType)
Pred.Predict()

