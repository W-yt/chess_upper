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

vector<string> PieceType;


int main()
{
    PieceType.push_back("1-黑-車/");
    PieceType.push_back("2-黑-卒/");
    PieceType.push_back("3-黑-将/");
    PieceType.push_back("4-黑-马/");
    PieceType.push_back("5-黑-炮/");
    PieceType.push_back("6-黑-士/");
    PieceType.push_back("7-黑-象/");
    PieceType.push_back("8-红-兵/");
    PieceType.push_back("9-红-車/");
    PieceType.push_back("10-红-马/");
    PieceType.push_back("11-红-炮/");
    PieceType.push_back("12-红-仕/");
    PieceType.push_back("13-红-帥/");
    PieceType.push_back("14-红-相/");
    Mat srcimage(Size(200,200),CV_8UC3);
    Mat dstimage(Size(200,200),CV_8UC3);
    Mat rotate_matrix;
    string image_name,output_name;

    for(int type = 0; type < 14; type++)
    {
        string image_src_dir = project_dir + "原始_每种20张/" + PieceType.at(type);
        string image_dst_dir = project_dir + "旋转_每种20x90张/" + PieceType.at(type);

        int num = 0;

        for(int i = 1; i < 21; i++)
        {
            image_name = image_src_dir + to_string(i) + ".jpg";

            srcimage = imread(image_name);
            imshow("srcimage",srcimage);

            Point2f center = Point(srcimage.cols / 2, srcimage.rows / 2);

            for(int angle = 1; angle < 361; angle+=4)
            {
                output_name = image_dst_dir + to_string(num) + ".jpg";

                rotate_matrix = getRotationMatrix2D(center,angle,1);
                warpAffine(srcimage,dstimage,rotate_matrix,srcimage.size());
                imshow("dstimage",dstimage);

                imwrite(output_name,dstimage);
                num++;
                waitKey(1);
            }

            waitKey(10);
        }

    }


    return 0;
}

