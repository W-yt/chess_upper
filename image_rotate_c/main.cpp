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

string project_dir =  "/home/yutian/Project/chess_upper/";

int main()
{
    Mat srcimage(Size(200,200),CV_8UC3);
    Mat dstimage(Size(200,200),CV_8UC3);
    Mat rotate_matrix;

    string image_src_dir = project_dir + "hong_shuai/";
    string image_dst_dir = project_dir + "train_hong_shuai/";
    string image_name,output_name;

    int num = 0;

    for(int i = 1; i < 6; i++)
    {
        image_name = image_src_dir + to_string(i) + ".jpg";

        srcimage = imread(image_name);
        imshow("srcimage",srcimage);

        Point2f center = Point(srcimage.cols / 2, srcimage.rows / 2);

        for(int angle = 1; angle < 361; angle++)
        {
            output_name = image_dst_dir + to_string(num) + ".jpg";

            rotate_matrix = getRotationMatrix2D(center,angle,1);
            warpAffine(srcimage,dstimage,rotate_matrix,srcimage.size());
            imshow("dstimage",dstimage);

            imwrite(output_name,dstimage);
            num++;
            waitKey(2);
        }

        waitKey(0);
    }

    return 0;
}

