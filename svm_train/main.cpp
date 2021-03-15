#include <stdio.h>
#include <time.h>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

#define NEED_TRAIN			1
#define NEED_PREDICT		1

#define CLASS_NUM           2
#define NUM_INCLASS         2800
#define SRC_SIZE            50
#define DST_SIZE            200
#define TRAIN_DELAY         0
#define PREDICT_DELAY       10
#define THRESHOD_EDGE       110
#define DISTANCE_EDGE       3600
#define CANNY_THRESHOD      1
#define CANNY_THRESHOLD1    50
#define CANNY_THRESHOLD2    120
#define EXPECT_TYPE         0
#define SVM_DIR             "../svm.xml"
#define TRAIN_DIR           "../train/"
#define TEST_DIR            "../test/"

Moments mo;
double Hu[7];

Mat trainingData = Mat::zeros(CLASS_NUM*NUM_INCLASS,7,CV_32FC1);
Mat trainingLabel = Mat::zeros(CLASS_NUM*NUM_INCLASS,1,CV_32SC1);

Size dstsize = cv::Size(DST_SIZE, DST_SIZE);

static void getFiles(string path, vector<string>& files);
static bool svmTrain(string dataPath, string saveFile);
static bool svmPredict(string dataPath, string loadFile, int expectVaule);

int main()
{
    bool ret = true;

    if (NEED_TRAIN)
    {
        cout << "Begain train pictures..." << endl;
        ret = svmTrain(TRAIN_DIR, SVM_DIR);
        if (true != ret)
        {
            cout << "SVM Train Fail!" << endl;
            goto out;
        }
    }

    if (NEED_PREDICT)
    {
        cout << "Begain predict pictures..." << endl;
        ret = svmPredict(TEST_DIR, SVM_DIR, EXPECT_TYPE);
        if (true != ret)
        {
            cout << "SVM Predict Fail!" << endl;
            goto out;
        }
    }

    out:
    getchar();

    return ret;
}

static void getFiles(string path, vector<string>& filenames)
{
    DIR* pDir;
    struct dirent* ptr;

    /* 打开一个其中一个字符的训练集目录 */
    if(!(pDir = opendir(path.c_str())))
        return;

    while((ptr = readdir(pDir))!=0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
}

static bool svmTrain(string dataPath, string saveFile)
{
    int ret = 1, i = 0;
    cout << "setting label..." << endl;

    /* 卒——0 兵——1 */
    for (i = 0; i < CLASS_NUM; ++i)
    {
        vector<string> files;
        //static vector<int> imagenumber;
        getFiles(dataPath + to_string(i),files);
        //imagenumber.at(i) = files.size();

        /* 依次循环处理各个类中的训练图片 */
        for (int j = 0; j < NUM_INCLASS; ++j)
        {
            /* 作为灰度图读取图片 */
            Mat srcimage = imread(files[j].c_str(), 0);

            resize(srcimage,srcimage,dstsize);

            /* 边缘检测 */
            Mat cannyimage;
            if(CANNY_THRESHOD == 0)
                Canny(srcimage, cannyimage, CANNY_THRESHOLD1, CANNY_THRESHOLD2);
            else
                threshold(srcimage,cannyimage,THRESHOD_EDGE,255,THRESH_BINARY);

            for(int src_rows = 0; src_rows < DST_SIZE; ++src_rows)
            {
                for(int src_cols = 0; src_cols < DST_SIZE; ++src_cols)
                {
                    if(pow((src_rows-DST_SIZE/2),2) + pow((src_cols-DST_SIZE/2),2) >= DISTANCE_EDGE)
                    {
                        if(CANNY_THRESHOD == 0)
                            cannyimage.at<uchar>(src_rows, src_cols) = 0;
                        else
                            cannyimage.at<uchar>(src_rows, src_cols) = 255;
                    }
                }
            }

//            imshow("cannyimage",cannyimage);
//            waitKey(TRAIN_DELAY);

            /* 求Hu矩 */
            mo = moments(cannyimage,true);
            HuMoments(mo, Hu);

            /* 将Hu矩填入训练数据集 */
//            trainingData.push_back(Hu);
            float *dstPoi = trainingData.ptr<float>(i*NUM_INCLASS + j);
            for (int r = 0; r < 7; r++)
                dstPoi[r] = (float) Hu[r];

            /* 添加对该数据的分类标签 */
//            trainingLabel.push_back(i);
            int *labPoi = trainingLabel.ptr<int>(i*NUM_INCLASS + j);
            labPoi[0] = i;
//            cout << "label = " << i <<endl;
        }
    }

    /* 创建SVM支持向量机并训练数据 */
    Ptr<ml::SVM> svm;
    cout << "training SVM..." << endl;
    svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setNu(1);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS,100,1));
    svm->train(trainingData, ROW_SAMPLE, trainingLabel);
    svm->save(saveFile);
    cout << "SVM Train Finish!!!" << endl;

    return true;
}

static bool svmPredict(string dataPath, string loadFile, int expectVaule)
{
    int result = 0;

    vector<string> files;
    getFiles(dataPath + to_string(expectVaule),files);
    int number = files.size();

    Ptr<ml::SVM> svm;
    FileStorage svm_fs(loadFile,FileStorage::READ);

    if (svm_fs.isOpened())
        svm = StatModel::load<SVM>(loadFile);
    else
    {
        cout << "Cannot find this file!" << endl;
        return false;
    }

    for (int i = 0; i < number; i++)
    {
        Mat inputimage = imread(files[i].c_str(), 0);
        resize(inputimage,inputimage,dstsize);

        //imshow("inputimage",inputimage);
        Mat inputcannyimage;

        if(CANNY_THRESHOD == 0)
            Canny(inputimage, inputcannyimage, CANNY_THRESHOLD1, CANNY_THRESHOLD2);
        else
            threshold(inputimage,inputcannyimage,THRESHOD_EDGE,255,THRESH_BINARY);

        for(int dst_rows = 0; dst_rows < DST_SIZE; ++dst_rows)
        {
            for(int dst_cols = 0; dst_cols < DST_SIZE; ++dst_cols)
            {
                int dstence_power = pow((dst_rows - DST_SIZE/2),2) + pow((dst_cols-DST_SIZE/2),2);
                if(dstence_power >= DISTANCE_EDGE)
                {
                    if(CANNY_THRESHOD == 0)
                        inputcannyimage.at<uchar>(dst_rows, dst_cols) = 0;
                    else
                        inputcannyimage.at<uchar>(dst_rows, dst_cols) = 255;
                }
            }
        }

        imshow("cannyimage",inputcannyimage);

        mo = moments(inputcannyimage,true);
        HuMoments(mo,Hu);
        waitKey(PREDICT_DELAY);

        Mat pre(1, 7, CV_32FC1);
        float *p = pre.ptr<float>(0);
        for(int j = 0 ; j < 7 ; j++)
            p[j] = Hu[j];

        int response = (int)svm->predict(pre);

        if (response == expectVaule)
            result++;
        else
            cout << "The " << i << "-th image predict fail, expect: [" << expectVaule << "], predict: [" << response << "]" << endl;
    }

    cout << number << " files totally!!!" << endl;
    cout << result << " files predict Success!!!" << endl;

    return true;
}