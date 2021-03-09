#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

Mat src_image;
Mat gray_image;
Mat yellow_image;
Mat binary_image;
Mat morph_image;
Mat threshold_image;
Mat board_image;
Mat board_image_gray;
Mat blank_board_image;
vector<Mat> src_image_channels;
int binary_edge = 90;
int element_size = 13;
vector<vector<Point>> contours;
vector<Point> max_contour;
double max_contour_area = 0;
int max_contour_index = 0;
vector<Vec4i> hierarchy;
int harris_thresh = 175;
vector<Point> board_point;
int chessboard_point_maxnum = 90;

void drawDetectLines(Mat&,const vector<Vec4i>&,Scalar);
void on_CornerHarris(int, void*);


int main() {
    /* project begin flag */
    std::cout << "Chess detecting proect begin!" << std::endl;

    /* test the Opencv lib */
//    Mat image_test;
//    //now dir is build dir, so we need ../ to father dir
//    image_test = imread("../test.jpg");
//    imshow("picture show",image_test);

    /* open the camera (this camera need calibration)*/
    VideoCapture capture(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH,1280);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,720);

    /* ome variate definition */
    Scalar blue_color = Scalar(255,0,0);//Blue
    Scalar green_color = Scalar(0,255,0);//Green
    Scalar red_color = Scalar(0,0,255);//Red
    Rect src_rect(280,0,720,720);

    while(1)
    {
        capture>>src_image;

        /* teak the mid square picture */
        src_image = src_image(src_rect);
        //imshow("[Src image]",src_image);
        cvtColor(src_image,gray_image,CV_BGR2GRAY);

        /* channels split and extract the yellow (BRG ==> G channel - B channel) */
        split(src_image,src_image_channels);
        subtract(src_image_channels.at(2),src_image_channels.at(0),yellow_image);
        //imshow("[Yellow image]",yellow_image);

        /* picture binarization(受光照影响比较大) */
        threshold(yellow_image,binary_image,binary_edge,255,THRESH_BINARY);
        //imshow("[Binary image]",binary_image);

        /* morphlogy tranform */
        Mat element = getStructuringElement(MORPH_RECT,Size(element_size,element_size));
        morphologyEx(binary_image,morph_image,MORPH_CLOSE,element);
        //imshow("[Morph image]",morph_image);

        /* find contours */
        findContours(morph_image,contours,hierarchy,CV_RETR_EXTERNAL,CHAIN_APPROX_SIMPLE,Point(0,0));

        /* find max contours and draw contours */
        Mat drawing_image = Mat::zeros(morph_image.size(),CV_8UC3);
        if(contours.size()>0)
        {
            for(int i = 0;i<contours.size();++i)
            {
                double contour_area;
                contour_area = contourArea(contours[i],false);
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
            static RotatedRect board_rect;
            board_rect = minAreaRect(max_contour);
            Point2f board_point[4];
            board_rect.points(board_point);
            for(int j=0;j<=3;j++)
                line(drawing_image,board_point[j],board_point[(j+1)%4],green_color,2);
            //imshow("[drawing image]",drawing_image);

            /* cot off the chess borad */
            Rect board_cutoff(board_point[0],board_point[2]);
            board_image = src_image(board_cutoff);
            imshow("board image",board_image);

//            /* find the real borad and then jump out the loop */
//            if(max_contour_area>250000 and max_contour_area<330000)
//                break;
        }

        /* board image canny detect */
        Mat canny_image;
        cvtColor(board_image,board_image_gray,CV_BGR2GRAY);
//      imshow("board_image_gray",board_image_gray);
        Canny(board_image_gray,canny_image,100,350);
        threshold(canny_image,canny_image,128,255,THRESH_BINARY);

        /* morphlogy tranform */
        Mat element2 = getStructuringElement(MORPH_RECT,Size(5,5));
        morphologyEx(canny_image,canny_image,MORPH_CLOSE,element);
//        imshow("canny_image",canny_image);

        /* hough detect the lines */
        vector<Vec4i> lines;
        blank_board_image = Mat::zeros(board_image.size(),CV_8UC1);
        HoughLinesP(canny_image,lines,1,CV_PI/180,50,400,60);
        drawDetectLines(blank_board_image,lines,Scalar(255));
        //imshow("blank_board_image",blank_board_image);

        /* angular point detect */
        Mat harris_dstImage;
        cornerHarris(blank_board_image,harris_dstImage,2,3, 0.04);
        Mat harris_showImage;
        board_image.copyTo(harris_showImage);
        Mat harris_normalImage;
        normalize(harris_dstImage, harris_normalImage, 0, 255, NORM_MINMAX);
        for (int i = 0; i < harris_showImage.rows; i++){
            for (int j = 0; j < harris_showImage.cols; j++){
                if (harris_normalImage.at<float>(i, j) > harris_thresh){
                    circle(harris_showImage,Point(j, i),5,Scalar(255,255,0),1,8);
                    board_point.push_back(Point(j,i));
                }
            }
        }
        //imshow("harris showImage",harris_showImage);

        cout<<"before board point num = "<<board_point.size()<<endl;

        /* delete the edge point */
        for(auto it = board_point.begin();it!=board_point.end();){
            if(it->x < 10 or it->x > board_image.cols-10 or it->y < 40 or it->y > board_image.rows-40)
                it = board_point.erase(it);
            else
                it++;
        }
        /* delete the coincide point */
        for(auto it1=board_point.begin();it1!=(board_point.end());++it1){
            if(it1==board_point.end())
                break;
            for(auto it2=it1+1;it2!=board_point.end();){
                if((abs(it1->x-it2->x)+abs(it1->y-it2->y)) < 50){
                    it2 = board_point.erase(it2);
                    //cout<<board_point.size()<<endl;
                }
                else{
                    it2++;
                }
            }
        }

        cout <<"after board point num = "<<board_point.size()<<endl;

        if(board_point.size() == chessboard_point_maxnum){
            break;
        }

        waitKey(50);
    }

    /* draw the point after delete */
    for(vector<Point>::iterator it = board_point.begin();it != board_point.end();++it){
        circle(board_image,Point(it->x, it->y),5,Scalar(255,255,0),2,8);
    }
    imshow("board image",board_image);
    cout<<"finial board point num = "<<board_point.size()<<endl;

    waitKey(0);

    return 0;
}

void drawDetectLines(Mat& image,const vector<Vec4i>& lines,Scalar color)
{
    vector<Vec4i>::const_iterator it=lines.begin();
    while(it!=lines.end())
    {
        Point pt1((*it)[0],(*it)[1]);
        Point pt2((*it)[2],(*it)[3]);
        line(image,pt1,pt2,color,1);
        ++it;
    }
}


