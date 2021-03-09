#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    /* project begin flag */
    std::cout << "Chess detecting proect begin!" << std::endl;

    /* define the drawing color */
    Scalar blue_color = Scalar(255,0,0);//Blue
    Scalar green_color = Scalar(0,255,0);//Green
    Scalar red_color = Scalar(0,0,255);//Red

    /* test the Opencv lib */
//    Mat image_test;
//    //now dir is build dir, so we need ../ to father dir
//    image_test = imread("../test.jpg");
//    imshow("picture show",image_test);

    /* open the camera (this camera need calibration)*/
    VideoCapture capture(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH,1280);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,720);

    Rect src_rect(280,0,720,720);

    while(1)
    {
        Mat src_image;
        capture>>src_image;

        /* teak the mid square picture */
        src_image = src_image(src_rect);
        imshow("[Src image]",src_image);
        Mat gray_image;
        cvtColor(src_image,gray_image,CV_BGR2GRAY);

        /* channels split and extract the yellow (BRG ==> G channel - B channel) */
        vector<Mat> src_image_channels;
        split(src_image,src_image_channels);
        Mat yellow_image;
        subtract(src_image_channels.at(2),src_image_channels.at(0),yellow_image);
        //imshow("[Yellow image]",yellow_image);

        /* picture binarization(受光照影响比较大) */
        Mat binary_image;
        int binary_edge = 90;
        threshold(yellow_image,binary_image,binary_edge,255,THRESH_BINARY);
        imshow("[Binary image]",binary_image);

        /* picture binarization(THRESH_OTSH大津法) */
//        Mat gray_image,binary_image;
////        cvtColor(src_image,gray_image,COLOR_BGR2GRAY);
////        imshow("[Gray image]",gray_image);
//        adaptiveThreshold(yellow_image,binary_image,255,THRESH_BINARY,ADAPTIVE_THRESH_GAUSSIAN_C,21,10);
//        imshow("[Binary image]",binary_image);
//
        /* morphlogy tranform */
        Mat morph_image;
        int element_size = 11;
        Mat element = getStructuringElement(MORPH_RECT,Size(element_size,element_size));
        morphologyEx(binary_image,morph_image,MORPH_CLOSE,element);
        imshow("[Morph image]",morph_image);

//        /* find the board keypoint using fast-detect */
////        vector<KeyPoint> board_keypoints;
////        Mat keypoint_image;
////        Ptr<FeatureDetector> fast = FastFeatureDetector::create(100);
////        fast->detect(binary_image,board_keypoints);
////        drawKeypoints(src_image,board_keypoints,src_image,Scalar::all(255),DrawMatchesFlags::DRAW_OVER_OUTIMG);
////        imshow("[board keypoint image]",src_image);
//
        /* find contours */
        vector<vector<Point>> contours;
        vector<Point> max_contour;
        double max_contour_area = 0;
        int max_contour_index = 0;
        vector<Vec4i> hierarchy;
        Mat threshold_image;
        findContours(morph_image,contours,hierarchy,CV_RETR_EXTERNAL,CHAIN_APPROX_SIMPLE,Point(0,0));

        /* find max contours and draw contours */
        Mat drawing_image = Mat::zeros(morph_image.size(),CV_8UC3);
        for(int i = 0;i<contours.size();++i)
        {
            double contour_area;
            contour_area = contourArea(contours[i],true);
            if(contour_area > max_contour_area)
            {
                max_contour_area = contour_area;
                max_contour_index = i;
            }
            drawContours(drawing_image,contours,i,blue_color,1,8,vector<Vec4i>(),0,Point());
        }
        max_contour = contours[max_contour_index];
        drawContours(drawing_image,contours,max_contour_index,red_color,1,8,vector<Vec4i>(),0,Point());

        /* find and draw the board rect */
        RotatedRect board_rect;
        board_rect = minAreaRect(max_contour);
        Point2f board_point[4];
        board_rect.points(board_point);
        for(int j=0;j<=3;j++)
            line(drawing_image,board_point[j],board_point[(j+1)%4],green_color,2);
        imshow("[drawing image]",drawing_image);


        //        vector<vector<Point>> hull(contours.size());
//        vector<vector<Point>> real_hull;
//        for(int i = 0;i<contours.size();++i)
//        {
//            convexHull(Mat(contours[i]),hull[i]);
//            if(hull[i].size()==5)
//                real_hull.push_back(hull[i]);
//        }
//        Mat drawing_image = Mat::zeros(morph_image.size(),CV_8UC3);
//        for(int i = 0;i<hull.size();++i)
//        {
//            Scalar color = Scalar(255,0,0);
//            //drawContours(drawing,contours,i,color,1,8,vector<Vec4i>(),0,Point());
//            drawContours(drawing_image,hull,i,color,1,8,vector<Vec4i>(),0,Point());
//        }
//        for(int i = 0;i<real_hull.size();++i)
//        {
//            Scalar color = Scalar(0,0,255);
//            //drawContours(drawing,contours,i,color,1,8,vector<Vec4i>(),0,Point());
//            drawContours(drawing,real_hull,i,color,1,8,vector<Vec4i>(),0,Point());
//        }
//        imshow("[drawing image]",drawing_image);

        waitKey(50);
    }

    waitKey(0);

    return 0;
}
