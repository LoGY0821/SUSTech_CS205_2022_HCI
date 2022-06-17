#include <opencv2/opencv.hpp>
using namespace cv;
#include <time.h>

// get the area whose color is similar to hand color
// void getSkin(Mat &ImageIn, Mat &Binary)
// {
//     Mat Image = ImageIn.clone();
//     Mat ycrcb_Image;
//     cvtColor(Image, ycrcb_Image, COLOR_BGR2YCrCb); //转换色彩空间
//     std::vector<Mat> y_cr_cb;
//     split(ycrcb_Image, y_cr_cb); //分离YCrCb
//     Mat CR = y_cr_cb[1]; //图片的CR分量
//     Mat CR1;
//     Binary = Mat::zeros(Image.size(), CV_8UC1);
//     GaussianBlur(CR, CR1, Size(3, 3), 0, 0);     //对CR分量进行高斯滤波，得到CR1（注意这里一定要新建一张图片存放结果）
//     threshold(CR1, Binary, 0, 255, THRESH_OTSU); //用系统自带的threshold函数，对CR分量进行二值化，算法为自适应阈值的OTSU算法
// }

void getSkin(Mat &ImageIn, Mat &Binary)
{
    Mat Image = ImageIn.clone();
    Mat hsvImage;
    cvtColor(Image, hsvImage, COLOR_BGR2HSV); //转换色彩空间
    inRange(hsvImage, Scalar(0, 43, 55), Scalar(25, 255, 255), Binary);
}

// get the contour of hand through a set of contours, according to the area of contours

void getHand(Mat &Binary, std::vector<Point> &hand)
{
    std::vector<std::vector<Point>> contours;
    findContours(Binary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0)); //提取出所有的轮廓
    if (contours.size() != 0)                                                      //如果图片中的轮廓不唯一
    {
        int max_contour = 0;
        double max_area = contourArea(InputArray(contours[0]), false);
        for (int i = 1; i < contours.size(); i++)
        {
            double temp_area = contourArea(InputArray(contours[i]), false);
            if (max_area < temp_area)
            {
                max_area = temp_area;
                max_contour = i;
            }
        }
        hand = contours[max_contour]; //手应该是最大的轮廓，返回最大的轮廓
    }
}
/*
 * 这个问题的主要原因是在与InputArray需要初始化，即需要声明InputArray的内存空间大小，否则无法读取正常的值。
 * 代码比较多的时候一定要仔细检测这个问题。
 * 尤其在声明vector<vector<Point>> 时，注意申请内存空间。
 * 若使用cv::drawContours函数时可能会返回上述错误。
 */

void drawHand(Mat &frame, std::vector<Point> &hand, std::vector<std::vector<Point>> hand_contours)
{
    if (hand.size() != 0)
    {
        hand_contours.push_back(hand);
    }
    if (hand_contours.size() != 0)
    {
        drawContours(frame, hand_contours, 0, Scalar(0, 0, 255), 2, 8);
    }
}

void calcMoment(std::vector<Point> &hand, std::vector<Point2f> &mc)
{
    std::vector<Moments> mu(hand.size());
    for (int i = 0; i < hand.size(); i++)
    {
        mu[i] = moments(hand, false);
    }
    for (int i = 0; i < hand.size(); i++)
    {
        mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }
}

void recordPoint(Point2f &point, Point2f &origin_point, std::vector<Point2f> &track)
{
    if (abs(point.x - origin_point.x) > 5 || abs(point.y - origin_point.y) > 5)
    {
        origin_point = point;
        track.push_back(origin_point); // push the current point to the track vector
    }
}

void drawTrace(std::vector<Point2f> &track, Mat &frame)
{
    for (int i = 0; i < track.size(); i++)
    {
        circle(frame, track[i], 2, Scalar(255, 255, 0), -1);
    }
}

void shapeRecog(std::vector<Point2f> &track, std::vector<Point2f> &output, std::string &shape)
{
    approxPolyDP(track, output, 5, true);
    int count = (int)output.size();
    switch (count)
    {
    case 3:
        shape = "Triangle";
        break;
    case 4:
        shape = "Square";
        break;
    case 5:
        shape = "Pentagon";
        break;
    default:
        shape = "Circle";
        break;
    }
}

int main()
{
    clock_t start, now;
    start = clock();
    int wide = 480;
    int height = 640;
    cv::VideoCapture cap(0);
    cap.set(3, height);
    cap.set(4, wide);

    Point2f origin_point(0, 0);
    Point2f current_point;

    std::vector<Point2f> track;

    while (cap.isOpened())
    {
        //读取画面
        Mat frame;
        cap >> frame;
        std::vector<std::vector<Point>> hand_contours;
        //将图片二值化，并初步提取出手
        Mat binary;
        getSkin(frame, binary);
        std::vector<Point> hand;
        //找到二值图像的最大边界，应该就是手了。
        getHand(binary, hand);
        // Draw the hand contour
        drawHand(frame, hand, hand_contours);
        // get the center of hand
        if (hand.size() != 0)
        {
            std::vector<Point2f> mc(hand.size());
            calcMoment(hand, mc);
            current_point = mc[0];
            // if the current point is 3 pixel away from the origin point, then update the current point
            recordPoint(current_point, origin_point, track);
            // draw the track
            drawTrace(track, frame);
            // center of the hand
            circle(frame, mc[0], 5, Scalar(255, 0, 0), -1);
            // get the bounding area of hand
            Rect rect = boundingRect(hand);                  // boundingRect 返回手的最小矩形区域
            rectangle(frame, rect, Scalar(0, 255, 0), 2, 8); // draw the boundingRect
        }
        now = clock();
        std::vector<Point2f> output;
        std::string shape = "null";
        if (double(now - start) / CLOCKS_PER_SEC >= 5)
        {
            shapeRecog(track, output, shape);
            // drawContours(frame, output, -1, Scalar(0, 255, 255), 2, 8);
            std::cout << shape << std::endl;
        }

        // Mirror the frame symmetrically
        // Mat frame_mirror;
        // flip(frame, frame_mirror, 1);

        // imshow("origin", frame_mirror);
        imshow("origin", frame);

        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }
}