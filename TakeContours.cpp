//
// Created by Msi-pc on 2022/6/12.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;

//get the area whose color is similar to hand color
void getSkin(Mat& ImageIn, Mat& Binary)
{
    Mat Image = ImageIn.clone();
    Mat ycrcb_Image;
    cvtColor(Image, ycrcb_Image, COLOR_BGR2YCrCb);//转换色彩空间

    std::vector<Mat>y_cr_cb;
    split(ycrcb_Image, y_cr_cb);//分离YCrCb

    Mat CR = y_cr_cb[1];//图片的CR分量
    Mat CR1;

    Binary = Mat::zeros(Image.size(), CV_8UC1);
    GaussianBlur(CR, CR1, Size(3, 3), 0, 0);//对CR分量进行高斯滤波，得到CR1（注意这里一定要新建一张图片存放结果）
    threshold(CR1, Binary, 0, 255, THRESH_OTSU);//用系统自带的threshold函数，对CR分量进行二值化，算法为自适应阈值的OTSU算法

}


//get the contour of hand through a set of contours, according to the area of contours

void getHand(Mat& Binary,std::vector<Point>& hand)
{
    std::vector<std::vector<Point>> contours;
    findContours(Binary, contours,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point(0,0));//提取出所有的轮廓
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

int main() {
    cv::VideoCapture cap(0);
    cap.set(3, 1080);
    cap.set(4, 720);

    int k = 0;
    while (cap.isOpened()) {
        Mat frame;
        cap >> frame; //读取画面
        Mat binary;
        getSkin(frame, binary); //将图片二值化，并初步提取出手
        std::vector<Point> hand;
        getHand(binary, hand); //找到二值图像的最大边界，应该就是手了。

        Mat fourierDescriptor;

        //If I press "k" in the keyboard, output the fourier descriptor of the hand in a txt file, in the directory named "1"
        if(waitKey(1) == 'k')
        {
            getFourierDescriptor(hand, fourierDescriptor);
            std::ofstream outfile;
            outfile.open(std::to_string(k)+".txt");
            for(int i = 0; i < fourierDescriptor.cols; i++)
            {
                outfile << fourierDescriptor.at<float>(i) << " ";
            }
            outfile.close();
            ++k;
        }
        std::vector<std::vector<Point>> hand_contours;
        hand_contours.push_back(hand);
        drawContours(frame,hand_contours,0,Scalar(0,0,255),2,8); //drawContours 不能过assertion，原因未知，不能正常显示

        Mat frame_mirror;
        flip(frame,frame_mirror,1);
        imshow("origin", frame_mirror);
        waitKey(1);
    }


}