import os
import warnings
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

# Ignore the hardware warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Pre process images
class PreFile(object):
    def __init__(self,FilePath,PieceType):
        self.FilePath = FilePath
        self.PieceType = PieceType

    def FileReName(self):
        type_counter = 0
        for type in self.PieceType:
            file_counter = 0
            subfolder = os.listdir(self.FilePath + type)
            for subclass in subfolder:
                file_counter += 1
                # print("file_counter:",file_counter)
                # print("type_counter:",type_counter)
                # print(subclass)
                os.rename(self.FilePath + type + "/" + subclass, self.FilePath + type + "/" + str(type_counter) + "_" + str(file_counter) + "_" + type + ".jpg")
            type_counter += 1
        print("rename finish!")

    def FileRemove(self,Output_folder):
        for type in self.PieceType:
            subfolder = os.listdir(self.FilePath + type)
            for subclass in subfolder:
                img_open = Image.open(self.FilePath + type + "/" + str(subclass))
                img_open.save(os.path.join(Output_folder, os.path.basename(subclass)))
        print("remove finish!")


class Training(object):
    def __init__(self,batch_size,num_batch,categorizes,train_folder):
        self.batch_size = batch_size
        self.number_batch = num_batch
        self.categories = categorizes
        self.train_folder = train_folder

    def read_train_images(self,filename):
        img = Image.open(self.train_folder+filename)
        return np.array(img)

    def train(self):
        train_image_list = []
        train_label_list = []
        for file in os.listdir(self.train_folder):
            files_img_in_array = self.read_train_images(filename = file)
            train_image_list.append(files_img_in_array)
            train_label_list.append(int(file.split("_")[0]))
            # print(Train_list_label)

        train_image_list = np.array(train_image_list)
        train_label_list = np.array(train_label_list)

        # print(train_image_list.shape)
        # print(train_label_list.shape)

        print("model label set finish!")

        train_label_list = np_utils.to_categorical(train_label_list,self.categories)
        train_image_list = train_image_list.astype("float32")
        train_image_list /=255.0

        model = Sequential()

        # CNN Layer —— 1
        model.add(Convolution2D( # input shape:(200,200,3)
            input_shape = (200,200,3),
            filters = 32, # next layer output:(200,200,32)
            kernel_size = (5,5), # pixel filtered
            padding = "same", # 外边距处理
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(
            pool_size = (2,2), # next layer output:(100,100,32)
            strides = (2,2),
            padding = "same"
        ))

        # CNN Layer —— 2
        model.add(Convolution2D(
            filters = 64, # next layer output:(100,100,64)
            kernel_size = (2,2), # pixel filtered
            padding = "same", # 外边距处理
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(
            pool_size = (2,2), # next layer output:(50,50,64)
            strides = (2,2),
            padding = "same"))

        # Fully connected Layer —— 1
        model.add(Flatten()) # 降维
        # model.add(Dense(1024))
        # model.add(Activation("relu"))
        # Fully connected Layer —— 2
        model.add(Dense(512))
        model.add(Activation("relu"))
        # Fully connected Layer —— 3
        model.add(Dense(256))
        model.add(Activation("relu"))
        # Fully connected Layer —— 4
        model.add(Dense(self.categories))
        model.add(Activation("softmax"))

        # Define Optimizer
        adam = Adam(lr = 0.001)

        # Compile the model
        model.compile(optimizer = adam,
                      loss = "categorical_crossentropy",
                      metrics = ["accuracy"])

        print("model compile finish!")

        #Fire up the network
        model.fit(x = train_image_list,
                  y = train_label_list,
                  epochs = self.number_batch,
                  batch_size = self.batch_size,
                  verbose = 1)

        # Save your work model
        model.save("./piecefinder.h5")

def MAIN():

    # PieceType = ["1-黑-車","2-黑-卒","3-黑-将" ,"4-黑-马"]

    PieceType = ["1-黑-車","2-黑-卒","3-黑-将" ,"4-黑-马" ,"5-黑-炮" ,"6-黑-士" ,"7-黑-象" ,
                 "8-红-兵","9-红-車","10-红-马","11-红-炮","12-红-仕","13-红-帥","14-红-相"]

    # # File pre processing
    # FILE = PreFile(FilePath = "原始数据目录/",PieceType = PieceType)
    #
    # # File rename and remove
    # FILE.FileReName()
    # FILE.FileRemove(Output_folder = "训练数据目录/")

    # Train the Network
    Train = Training(batch_size = 1, num_batch = 50, categorizes = 14, train_folder = "训练数据目录/")
    Train.train()


if __name__ == "__main__":
    MAIN()





