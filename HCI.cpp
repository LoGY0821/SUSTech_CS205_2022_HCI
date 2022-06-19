#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <opencv2/imgproc/imgproc_c.h>
#include <windows.h>
#include <time.h>

using namespace cv;

Mat frame,binary,frame_mirror; //the current frame, binary image of skin,mirror image of current frame
std::vector<Point> hand;
std::vector<std::vector<Point>> hand_contours; //contours of hand, it is required by the function drawContours

Point2f origin_point;
Point2f current_point(0,0); //to store the center of the hand

std::vector<Point2f> track; //to store the track of the hand

int mode;//0 1 2 3 4
clock_t t1, t2;//定时器变量



void getHand();
void getSkin();
void getskin();
int getConvexHull();
void getCurrentCenter();
void drawTrack();
void clearTrack();
void shapeRecog(std::vector<Point2f> &recog_trace, std::string &shape);
void contourGene(std::vector<Point2f> &points, std::vector<std::vector<Point>> &trace);
void drawCircle(Mat &Image, std::vector<Point2f> &points);
void virtualMouse();

int main()
{
    cv::VideoCapture cap(0);
    cap.set(3,1080);
    cap.set(4,720);

    t1 = clock();
//    //output the information of the camera
//    std::cout << "The camera's information:" << std::endl;
//    std::cout << "width: " << cap.get(3) << std::endl;
//    std::cout << "height: " << cap.get(4) << std::endl;
//    std::cout << "fps: " << cap.get(5) << std::endl;
//    std::cout << "brightness: " << cap.get(6) << std::endl;
//    std::cout << "contrast: " << cap.get(7) << std::endl;
//    std::cout << "saturation: " << cap.get(8) << std::endl;
    while (cap.isOpened())
    {
        cap >> frame; //读取画面

        getSkin(); //将图片二值化，并初步提取出手
        getHand(); //找到二值图像的最大边界，应该就是手了。

        hand_contours.push_back(hand);

        clearTrack();


        //get the bounding area of hand
        Rect rect = boundingRect(hand); //boundingRect 返回手的最小矩形区域
        rectangle(frame,rect,Scalar(0,255,0),2,8); //draw the boundingRect

        //virtualMouse();

        //Mirror the frame symmetrically
        mode = getConvexHull(); //get the convexhull of the hand

        std::vector<Point2f> trace_ori;
        std::vector<std::vector<Point>> output;
        std::string shape = "null";
        if(mode == 3) {
            getCurrentCenter();
            drawTrack();

        }
        if(mode == 0 && track.size() > 30) {
            shapeRecog(trace_ori, shape);
            contourGene(trace_ori, output);
            if (trace_ori.size() > 4)
            {
                drawCircle(frame, trace_ori);
            }
            else
            {
                drawContours(frame, output, -1, Scalar(0, 255, 255), 4, 8);
            }

        }


        flip(frame,frame_mirror,1);
        if(mode == 4)
            putText(frame_mirror, std::to_string(mode+1), Point(10, 60), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
        else if(mode == 3)
            putText(frame_mirror, std::to_string(mode+1), Point(10, 60), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
        else
            putText(frame_mirror, std::to_string(mode), Point(10, 60), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
        //if shape is not null, write the shape on the frame_mirror
        if(shape != "null")
            putText(frame_mirror, shape, Point(500, 90), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);

        //predict the result
        //predictSVM(hand,frame_mirror);

        imshow("origin", frame_mirror);
        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }

}



//use cr_cb color space to get the skin
void getSkin()
{
    Mat Image = frame.clone();
    Mat ycrcb_Image;
    cvtColor(Image, ycrcb_Image, COLOR_BGR2YCrCb);//转换色彩空间

    std::vector<Mat>y_cr_cb;
    split(ycrcb_Image, y_cr_cb);//分离YCrCb

    Mat CR = y_cr_cb[1];//图片的CR分量
    Mat CR1;

    binary = Mat::zeros(Image.size(), CV_8UC1);
    GaussianBlur(CR, CR1, Size(3, 3), 0, 0);//对CR分量进行高斯滤波，得到CR1（注意这里一定要新建一张图片存放结果）
    threshold(CR1, binary, 0, 255, THRESH_OTSU);//用系统自带的threshold函数，对CR分量进行二值化，算法为自适应阈值的OTSU算法
    //imshow("cr_cb", binary);

}


//use hsv color space to get the skin
void getskin()
{
    Mat Image = frame.clone();
    Mat hsvImage;
    cvtColor(Image, hsvImage, COLOR_BGR2HSV); //转换色彩空间
    inRange(hsvImage, Scalar(0, 43, 55), Scalar(25, 255, 255), binary);
    //imshow("RGB", binary);
}



//get the contour of hand through a set of contours, according to the area of contours

void getHand()
{
    std::vector<std::vector<Point>> contours;
    findContours(binary, contours,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point(0,0));//提取出所有的轮廓
    if(contours.size() != 0) //如果图片中的轮廓不唯一
    {
        int max_contour = 0;
        double max_area = contourArea(InputArray(contours[0]), false);
        for(int i = 1; i < contours.size();i++)
        {
            double temp_area = contourArea(InputArray(contours[i]),false);
            if(max_area < temp_area)
            {
                max_area = temp_area;
                max_contour = i;
            }
        }
        hand = contours[max_contour]; //手应该是最大的轮廓，返回最大的轮廓
    }

}

void getHand(Mat& Binary,std::vector<Point>& Hand )
{
    std::vector<std::vector<Point>> contours;
    findContours(binary, contours,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point(0,0));//提取出所有的轮廓
    if(contours.size() != 0) //如果图片中的轮廓不唯一
    {
        int max_contour = 0;
        double max_area = contourArea(InputArray(contours[0]), false);
        for(int i = 1; i < contours.size();i++)
        {
            double temp_area = contourArea(InputArray(contours[i]),false);
            if(max_area < temp_area)
            {
                max_area = temp_area;
                max_contour = i;
            }
        }
        Hand = contours[max_contour]; //手应该是最大的轮廓，返回最大的轮廓
    }

}



//calculate Fourier Descriptor
//暂时还没用，就是写到这了，后面要用神经网络了 https://blog.csdn.net/qq_42884797/article/details/110917941
void getFourierDescriptor(std::vector<Point>& hand, Mat& FourierDescriptor)
{
    Point P;
    std::vector<float> f;
    std::vector<float> fd;
    Mat src1(Size(hand.size(),1),CV_8SC2);
    for(int i = 0; i < hand.size(); i++)
    {
        float x,y,sumx=0,sumy=0;
        for(int j = 0; j < hand.size(); j++)
        {
            P = hand[j];
            x = P.x;
            y = P.y;
            sumx += (float)(x * cos(2 * CV_PI * i * j / hand.size()) + y * sin(2 * CV_PI * i * j / hand.size()));
            sumy += (float)(-x * sin(2 * CV_PI * i * j / hand.size()) + y * cos(2 * CV_PI * i * j / hand.size()));
        }
        f.push_back(sqrt(sumx * sumx + sumy * sumy)); //求每个特征的模

    }

    fd.push_back(0); //0位标志位

    //进行归一化
    for(int k = 2; k <16; k++)
    {
        f[k] = f[k] /f[1];
        fd.push_back(f[k]);
    }

    FourierDescriptor = Mat::zeros(1,fd.size(),CV_32F);//CV32_F  float -像素是在0-1.0之间的任意值，这对于一些数据集的计算很有用，但是它必须通过将每个像素乘以255来转换成8位来保存或显示。

    for(int i = 0; i < fd.size(); i++)
    {
        FourierDescriptor.at<float>(i) = fd[i];
    }

}

void predictSVM(std::vector<Point>& hand, Mat& frame)
{
    Mat FourierDescriptor;
    getFourierDescriptor(hand,FourierDescriptor);
    Ptr<ml::SVM> psvm = ml::SVM::create();
    psvm= Algorithm::load<ml::SVM>("./svm_model.xml");
    int match = int (psvm->predict(FourierDescriptor));
    //show the result in the left top corner of frame
    putText(frame, std::to_string(match), Point(10, 60), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);

}


//blur the erode image and get the contour and find the convexhull return the size of convexhull
int getConvexHull()
{
    Mat Binary_erode;
    Mat Binary_close;
    //the kernel is 5x5, all the element is 1
    Mat kernel = Mat::ones(5,5,CV_8UC1);
    morphologyEx(binary, Binary_close, MORPH_CLOSE, kernel);

    erode(Binary_close, Binary_erode, kernel);
    //do a closa operation on Binary_erode

    //imshow("Binary_blur", Binary_close);
    std::vector<Point> Hand;
    getHand(Binary_close,Hand);
    //find the convex in hand
    std::vector<Point> hull;
    std::vector<int> hull_index(Hand.size());
    convexHull(Hand, hull);
    convexHull(Hand, hull_index, false);
    std::vector<Vec4i> defects(Hand.size());
    convexityDefects(Mat(Hand), hull_index, defects);
    //draw the defects
    std::vector<Vec4i>::iterator d = defects.begin();

    int cnt = 0;
    while (d != defects.end()) {

        Vec4i& v = (*d);
        //if(IndexOfBiggestContour == i)
        {

            int startidx = v[0];
            Point ptStart(Hand[startidx]); // point of the contour where the defect begins
            int endidx = v[1];
            Point ptEnd(Hand[endidx]); // point of the contour where the defect ends
            int faridx = v[2];
            Point ptFar(Hand[faridx]);// the farthest from the convex hull point within the defect
            int depth = v[3] / 256; // distance between the farthest point and the convex hull

            if (depth > 50 && depth < 500)
            {
                line(frame, ptStart, ptFar, CV_RGB(0, 255, 0), 2);
                line(frame, ptEnd, ptFar, CV_RGB(0, 255, 0), 2);
                circle(frame, ptStart, 4, Scalar(255, 0, 0), 2);
                circle(frame, ptEnd, 4, Scalar(255, 0, 0), 2);
                circle(frame, ptFar, 4, Scalar(100, 0, 255), 2);
                cnt++;
            }


        }
        //draw the cnt on the frame

        d++;
    }
    return cnt;
}

//if mode == 4, clear the data in track
void clearTrack()
{
    if(mode == 4)
    {
        track.clear();
    }
}



void getCurrentCenter()
{
    //get the center of hand
    std::vector<Moments> mu(hand.size());
    for(int i = 0; i < hand.size(); i++)
    {
        mu[i] = moments(hand,false);
    }
    std::vector<Point2f> mc(hand.size());
    for(int i = 0; i < hand.size(); i++)
    {
        mc[i] = Point2f(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
    }
    origin_point = mc[0];

    //if the current point is 3 pixel away from the origin point, then update the current point, use this to reduce the shaking
    if(abs(origin_point.x - current_point.x) > 5 || abs(origin_point.y - current_point.y) > 5)
    {
        current_point = origin_point;
        track.push_back(current_point); // push the current point to the track vector
    }
}

void drawTrack()
{
    //draw the track
    for(int i = 0; i < track.size(); i++)
    {
        circle(frame,track[i],2,Scalar(0,255,255),-1);
    }
    circle(frame,current_point,5,Scalar(255,0,0),-1); //center of the hand
}

//输出近似后的轨迹以及现状
void shapeRecog(std::vector<Point2f> &recog_trace, std::string &shape)
{
    approxPolyDP(track, recog_trace, 50, true);
    int count = (int)recog_trace.size();
    switch (count)
    {
        case 2:
            shape = "line";
            break;
        case 3:
            shape = "Triangle";
            break;
        case 4:
            shape = "Rectangle";
            break;
        default:
            shape = "Circle";
            break;
    }
}

//方便画
void contourGene(std::vector<Point2f> &points, std::vector<std::vector<Point>> &trace)
{
    std::vector<Point> line;
    for (int i = 0; i < points.size(); i++)
    {
        line.push_back(Point((int)points[i].x, (int)points[i].y));
    }
    trace.push_back(line);
}

//画圆
void drawCircle(Mat &Image, std::vector<Point2f> &points)
{
    int centerX=0;
    int centerY=0;
    int radius=0;
    for (int i = 0; i < points.size(); i++)
    {
        centerX += points[i].x;
        centerY += points[i].y;
    }
    centerX /= points.size();
    centerY /= points.size();
    for (int i = 0; i < points.size(); i++)
    {
        radius += sqrt(pow((points[i].x - centerX), 2) + pow((points[i].y - centerY), 2));
    }
    radius /= points.size();
    circle(frame, Point(centerX, centerY), radius, Scalar(0, 255, 255), 4);
}

void virtualMouse()
{
    rectangle(frame, Point(160, 180), Point(800, 540), Scalar(255, 0, 0), 2);
    Moments handMoment = moments(hand, false);
    Point center(handMoment.m10 / handMoment.m00, handMoment.m01 / handMoment.m00);
    int center_x = (int)center.x - 2 * ((int)center.x - 480);
    double x = (center_x-160) * 65536 / 640;//质心转换到鼠标
    double y = (center.y-180) * 65536 / 360;
    if (x <= 0) x = 0;
    else if (x >= 65536) x = 65536;
    if (y <= 0) y = 0;
    else if (y >= 65536) y = 65536;

    t2 = clock();
    //cout << t1 << "   " << t2 << endl;
    if (mode == 1 && center.x > 0 && center.y > 0 && center.x > 0 && center.y > 0 && (double)(t2 - t1)/CLOCKS_PER_SEC > 1)
    {
        mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, x, y + 3640, 0, 0);
        mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
        t1 = t2;
    }
    if (mode == 2 && center.x > 0 && center.y > 0 && center.x > 0 && center.y > 0 && (double)(t2 - t1) / CLOCKS_PER_SEC > 1)
    {
        mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, x, y + 3640, 0, 0);
        mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
        mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
        t1 = t2;
    }
    if (mode == 0 && center.x >= -100000)
    {
        mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, x, y, 0, 0);
    }
}
