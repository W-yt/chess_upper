# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.3.29                                        #
# Author : W-yt                                             #
# File   : train                                            #
# ######################################################### #
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
class Prepare(object):
    def __init__(self,TrainFilePath, TestFilePath, PieceType):
        self.TrainFilePath = TrainFilePath
        self.TestFilePath = TestFilePath
        self.PieceType = PieceType


    def FileRename(self):
        # Train file rename
        type_counter = 0
        for type in self.PieceType:
            file_counter = 0
            sub_folder = os.listdir(self.TrainFilePath + type)
            for subclass in sub_folder:
                file_counter += 1
                # print("file_counter:",file_counter)
                # print("type_counter:",type_counter)
                # print(subclass)
                os.rename(self.TrainFilePath + type + "/" + subclass, self.TrainFilePath + type + "/" + str(type_counter) + "_" + str(file_counter) + "_" + type + ".jpg")
            type_counter += 1
            print("train image rename one type!")
        print("Train file rename finish!")
        # Test file rename
        type_counter = 0
        for type in self.PieceType:
            file_counter = 0
            sub_folder = os.listdir(self.TestFilePath + type)
            for subclass in sub_folder:
                file_counter += 1
                # print("file_counter:",file_counter)
                # print("type_counter:",type_counter)
                # print(subclass)
                os.rename(self.TestFilePath + type + "/" + subclass, self.TestFilePath + type + "/" + str(type_counter) + "_" + str(file_counter) + "_" + type + ".jpg")
            type_counter += 1
            print("test image rename one type!")
        print("Test file rename finish!")


    def FileRemove(self,Train_Output_folder, Test_Output_folder):
        # Train file remove
        for type in self.PieceType:
            sub_folder = os.listdir(self.TrainFilePath + type)
            for subclass in sub_folder:
                img_open = Image.open(self.TrainFilePath + type + "/" + str(subclass))
                img_open.save(os.path.join(Train_Output_folder, os.path.basename(subclass)))
            print("train image remove one type!")
        print("Train file remove finish!")
        # Test file remove
        for type in self.PieceType:
            sub_folder = os.listdir(self.TestFilePath + type)
            for subclass in sub_folder:
                img_open = Image.open(self.TestFilePath + type + "/" + str(subclass))
                img_open.save(os.path.join(Test_Output_folder, os.path.basename(subclass)))
            print("test image remove one type!")
        print("Test file remove finish!")


# Train the CNN model
class Training(object):
    def __init__(self,batch_size,num_batch,categorizes,train_folder,test_folder):
        self.batch_size = batch_size
        self.number_batch = num_batch
        self.categories = categorizes
        self.train_folder = train_folder
        self.test_folder = test_folder


    def read_train_images(self,filename):
        img = Image.open(self.train_folder + filename)
        return np.array(img)


    def read_test_images(self,filename):
        img = Image.open(self.test_folder + filename)
        return np.array(img)


    def train(self):
        train_image_list = []
        train_label_list = []
        test_image_list = []
        test_label_list = []
        for file in os.listdir(self.train_folder):
            files_img_in_array = self.read_train_images(filename = file)
            train_image_list.append(files_img_in_array)
            train_label_list.append(int(file.split("_")[0]))
            # print(Train_list_label)
        print("Train image load finish!")
        for file in os.listdir(self.test_folder):
            files_img_in_array = self.read_test_images(filename = file)
            test_image_list.append(files_img_in_array)
            test_label_list.append(int(file.split("_")[0]))
            # print(Test_list_label)
        print("Test image load finish!")

        train_image_list = np.array(train_image_list).reshape([18900,50,50,1])
        train_label_list = np.array(train_label_list)
        test_image_list = np.array(test_image_list).reshape([6300,50,50,1])
        test_label_list = np.array(test_label_list)
        # print(train_image_list.shape)
        # print(train_label_list.shape)
        print("Image to array finish!")

        train_label_list = np_utils.to_categorical(train_label_list,self.categories)
        train_image_list = train_image_list.astype("float32")
        train_image_list /=255.0
        test_label_list = np_utils.to_categorical(test_label_list,self.categories)
        test_image_list = test_image_list.astype("float32")
        test_image_list /=255.0
        print("Model to categorical finish!")

        model = Sequential()
        # CNN Layer —— 1
        model.add(Convolution2D(input_shape = (50,50,1), filters = 32, kernel_size = (5,5), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "same"))

        # CNN Layer —— 2
        model.add(Convolution2D(filters = 64, kernel_size = (2,2), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "same"))

        model.add(Flatten()) # 降维
        # # Fully connected Layer —— 1
        # model.add(Dense(1024))
        # model.add(Activation("relu"))
        # Fully connected Layer —— 2
        model.add(Dense(512))
        model.add(Activation("relu"))
        # Fully connected Layer —— 3
        model.add(Dense(256))
        model.add(Activation("relu"))
        # Fully connected Layer —— 4
        # model.add(Dropout(0.4))
        model.add(Dense(self.categories))
        model.add(Activation("softmax"))

        # Define Optimizer
        adam = Adam(lr = 0.002)

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
                  validation_data = (test_image_list,test_label_list),
                  verbose = 1)

        # Save your work model
        model.save("piece_finder_red.h5")


def MAIN():
    piecetype_english = ["1-hong-bing", "2-hong-ju", "3-hong-ma", "4-hong-pao", "5-hong-shi", "6-hong-shuai", "7-hong-xiang"]

    # File pre processing
    FILE = Prepare(TrainFilePath = "rotate_train_data/", TestFilePath = "rotate_test_data/", PieceType = piecetype_english)

    # File rename and remove
    # FILE.FileRename()
    # FILE.FileRemove(Train_Output_folder = "final_train_data/",Test_Output_folder = "final_test_data/")

    # Train the Network
    Train = Training(batch_size = 16, num_batch = 2, categorizes = 7, train_folder = "final_train_data/", test_folder = "final_test_data/")
    Train.train()

if __name__ == "__main__":
    MAIN()









