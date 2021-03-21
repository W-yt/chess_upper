from keras.models import load_model
import matplotlib.image as processimage
import numpy as np
from PIL import Image

class Prediction(object):
    def __init__(self,ModelFile,PredictFile,PieceType):
        self.model_file = ModelFile
        self.predict_file = PredictFile
        self.PieceType = PieceType

    def Predict(self):
        model = load_model(self.model_file)

        # Deal the image's shape\
        origin_image = Image.open(self.predict_file)
        resieze_image = origin_image.resize((50,50),Image.BILINEAR)
        resieze_image.save(self.predict_file)

        image = processimage.imread(self.predict_file)
        image_to_array = np.array(image).astype("float32")/255.0
        image_to_array = image_to_array.reshape(1,50,50,3)
        print("image reshape finish!")

        # Predict the image
        prediction = model.predict(image_to_array)
        # Final_Pred = [result.argmax() for result in prediction]
        # print(Final_Pred)
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

# 爆内存的时候，测试模型的可行性时用的
# PieceType = ["黑-卒", "红-兵", "黑-将", "红-帥"]

PieceType = ["1-黑-車", "2-黑-卒", "3-黑-将", "4-黑-马", "5-黑-炮", "6-黑-士", "7-黑-象",
             "8-红-兵", "9-红-車", "10-红-马","11-红-炮","12-红-仕","13-红-帥","14-红-相"]

Pred = Prediction(PredictFile = "测试数据目录/1-黑-車/12.jpg",ModelFile = "piecefinder.h5",PieceType = PieceType)
Pred.Predict()

