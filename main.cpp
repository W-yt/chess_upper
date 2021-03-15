#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

/* some project config flag */
#define CAMERA_ADJUST               0
#define CHESS_BOARD_RECOGNIZE_ON    0
#define CHESS_PIECE_DETECT_ON       1
#define CHESS_PIECE_SAVE            0

/* variate definition */
Mat src_image;
Mat gray_image;
Mat yellow_image;
Mat binary_image;
Mat morph_image;
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
vector<vector<Point>> chessboard_point_sorted;
Point2f boardedge_point[4];

Mat piece_image;
Mat piece_image_gray;
Mat piece_image_thre;
Mat piece_image_blur;
vector<Vec3f> circles_hough;
vector<Mat> piece_image_channels;

int piece_roi_size = 50;

/* founctions declare */
void drawDetectLines(Mat&,const vector<Vec4i>&,Scalar);
void on_CornerHarris(int, void*);
bool point_x_sort(const Point& a, const Point& b);
bool point_y_sort(const Point& a, const Point& b);

int main() {
    /* project begin flag */
    std::cout << "Chess detecting proect begin!" << std::endl;

    /* open the camera (this camera need calibration)*/
    VideoCapture capture(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH,1280);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,720);

    /* some variate definition */
    Scalar blue_color = Scalar(255,0,0);//Blue
    Scalar green_color = Scalar(0,255,0);//Green
    Scalar red_color = Scalar(0,0,255);//Red

    /* camera adjust mode */
    if(CAMERA_ADJUST)
    {
        while(1)
        {
            capture>>src_image;
            /* teak the mid square picture */
            Rect src_rect(280,0,720,720);
            src_image = src_image(src_rect);
            imshow("[Src image]",src_image);

            waitKey(30);
        }
    }

    /* first recognize the chess board(before placing the chess pieces) */
    if(CHESS_BOARD_RECOGNIZE_ON)
    {
        while(1)
        {
            capture>>src_image;

            /* teak the mid square picture */
            Rect src_rect(280,0,720,720);
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
//                Point2f boardedge_point[4];
                board_rect.points(boardedge_point);
                for(int j=0;j<=3;j++)
                    line(drawing_image,boardedge_point[j],boardedge_point[(j+1)%4],green_color,2);
                //imshow("[drawing image]",drawing_image);

                /* cot off the chess borad */
                Rect board_cutoff(boardedge_point[0],boardedge_point[2]);
                board_image = src_image(board_cutoff);
                //imshow("board image",board_image);

//            /* find the real borad and then jump out the loop */
//            if(max_contour_area>250000 and max_contour_area<330000)
//                break;
            }

            /* board image canny detect */
            Mat canny_image;
            cvtColor(board_image,board_image_gray,CV_BGR2GRAY);
//          imshow("board_image_gray",board_image_gray);
            Canny(board_image_gray,canny_image,100,350);
            threshold(canny_image,canny_image,128,255,THRESH_BINARY);

            /* morphlogy tranform */
            Mat element2 = getStructuringElement(MORPH_RECT,Size(5,5));
            morphologyEx(canny_image,canny_image,MORPH_CLOSE,element);
            //imshow("canny_image",canny_image);

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
            if(board_point.size() == chessboard_point_maxnum)
                break;
            waitKey(20);
        }

        /* draw the point after delete */
        for(vector<Point>::iterator it = board_point.begin();it != board_point.end();++it){
            circle(board_image,Point(it->x, it->y),4,Scalar(255,255,255),2,8);
        }

        /* sort the 90 chess board points */
        sort(board_point.begin(),board_point.end(),point_x_sort);
        vector<Point> temp_point_group(9);
        for(int j = 0; j != 10 ; ++j)
        {
            for(int i = 0; i != 9; ++i)
                temp_point_group.at(i) = board_point.at(9*j+i);
            sort(temp_point_group.begin(),temp_point_group.end(),point_y_sort);
            chessboard_point_sorted.push_back(temp_point_group);
        }

        /* display the chess board point tag */
        for(int j = 0; j != 10; ++j)
        {
            for(int i = 0; i != 9; ++i)
            {
                string num1 = to_string(j+1);
                string num2 = to_string(i+1);
                string point_tag = "("+num1+","+num2+")";
                Point tag_place = Point(chessboard_point_sorted.at(j).at(i).x-24,chessboard_point_sorted.at(j).at(i).y-10);
                putText(board_image,point_tag,tag_place,CV_FONT_HERSHEY_TRIPLEX,0.5,Scalar(255,255,255));
            }
        }

        imshow("board image",board_image);
        //cout<<"finial board point num = "<<board_point.size()<<endl;
    }

    /* then detect the chess pieces's place */
    if(CHESS_PIECE_DETECT_ON)
    {
        /* avoid extra cutoff, step only for debug(this need adjust when you move the camera or chess) */
        if(CHESS_BOARD_RECOGNIZE_ON == 0)
        {
            boardedge_point[0] = Point(67.8,646.0);
            boardedge_point[1] = Point(65.4,85.1);
            boardedge_point[2] = Point(635.5,82.6);
            boardedge_point[3] = Point(637.9,643.6);
        }

        while (1)
        {
            capture>>src_image;
            /* teak the mid square picture */
            Rect src_rect(280,0,720,720);
            src_image = src_image(src_rect);
            //imshow("[Src image]",src_image);

            /* cut off the mid image */
            Rect board_cutoff(boardedge_point[0],boardedge_point[2]);
            board_image = src_image(board_cutoff);
            piece_image = board_image.clone();
            //imshow("piece image",piece_image);

            /* image enhancement */
            cvtColor(piece_image,piece_image_gray,CV_BGR2GRAY);
            split(piece_image,piece_image_channels);
            //imshow("green channel image",piece_image_channels.at(1));
            threshold(piece_image_channels.at(1),piece_image_thre,128,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
            blur(piece_image_thre,piece_image_blur,Size(3,3));

            /* HoughCilcles函数参数说明：
             * param1:此参数是对应Canny边缘检测的最大阈值，最小阈值是此参数的一半 也就是说像素的值大于param1是会检测为边缘
             * param2:它表示在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了
             */
            HoughCircles(piece_image_blur,circles_hough,CV_HOUGH_GRADIENT,1,40,100,20,18,21);

            for (std::size_t i = 0; i < circles_hough.size(); i++)
            {
                /* cvRound返回和参数最接近的整数值 */
                Point center(cvRound(circles_hough[i][0]), cvRound(circles_hough[i][1]));
                int radius = cvRound(circles_hough[i][2]);

//                cout << i << "\t" << " center " << "= " << center << ";\n" << endl;
//                cout << i << "\t" << " radius " << "= " << radius << ";\n" << endl;
//                circle(piece_image_blur,center,3,Scalar(0, 255, 0),-1,8,0);
//                circle(piece_image_blur,center,radius,Scalar(155, 50, 255),2,8,0);

                if(CHESS_PIECE_SAVE == 0)
                {
                    circle(piece_image,center,3,Scalar(0, 255, 0),-1,8,0);
                    circle(piece_image,center,radius+3,Scalar(155, 50, 255),2,8,0);
                }

                /* save the piece image roi */
                if(CHESS_PIECE_SAVE == 1)
                {
                    /* cut off each piece */
                    Rect piece_ROI(Point(center.x-piece_roi_size/2,center.y-piece_roi_size/2),Point(center.x+piece_roi_size/2,center.y+piece_roi_size/2));
                    Mat piece_cutoff = piece_image(piece_ROI);
                    Mat piece_save = Mat::zeros(Size(piece_roi_size,piece_roi_size),CV_8UC3);
                    Mat piece_mask = Mat::zeros(Size(piece_roi_size,piece_roi_size),CV_8UC1);

                    for (std::size_t i = 0; i < circles_hough.size(); i++)
                    {
                        circle(piece_mask,Point(piece_roi_size/2,piece_roi_size/2),radius-2,Scalar(255),-1);
                        piece_cutoff.copyTo(piece_save,piece_mask);
                    }

                    /* show the piece roi to save */
                    imshow("piece cutoff",piece_cutoff);
                    imshow("piece mask",piece_mask);
                    imshow("piece save",piece_save);

//                    /* 卒 */
//                    static String savepath = "../zu/";
//                    static int savenum = 1;
//                    String savename = to_string(savenum++);
//                    static String savetype = ".jpg";
//                    String savefullname = savepath+savename+savetype;
//                    imwrite(savefullname,piece_save);

//                    /* 兵 */
//                    static String savepath = "../bing/";
//                    static int savenum = 1;
//                    String savename = to_string(savenum++);
//                    static String savetype = ".jpg";
//                    String savefullname = savepath+savename+savetype;
//                    imwrite(savefullname,piece_save);
                }
            }
            cout << "find " << circles_hough.size() << " circles" << ";\n" << endl;
            //imshow("piece image blur", piece_image_blur);
            imshow("piece image", piece_image);

            waitKey(20);
        }
    }

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

bool point_x_sort(const Point& a, const Point& b)
{
    if(a.x<=b.x)
        return true;
    else
        return false;
}

bool point_y_sort(const Point& a, const Point& b)
{
    if(a.y<=b.y)
        return true;
    else
        return false;
}
