#include <opencv2/opencv.hpp>
using namespace cv;

// get the area whose color is similar to hand color
void getSkin(Mat &ImageIn, Mat &Binary)
{
    Mat Image = ImageIn.clone();
    Mat ycrcb_Image;
    cvtColor(Image, ycrcb_Image, COLOR_BGR2YCrCb); //转换色彩空间

    std::vector<Mat> y_cr_cb;
    split(ycrcb_Image, y_cr_cb); //分离YCrCb

    Mat CR = y_cr_cb[1]; //图片的CR分量
    Mat CR1;

    Binary = Mat::zeros(Image.size(), CV_8UC1);
    GaussianBlur(CR, CR1, Size(3, 3), 0, 0);     //对CR分量进行高斯滤波，得到CR1（注意这里一定要新建一张图片存放结果）
    threshold(CR1, Binary, 0, 255, THRESH_OTSU); //用系统自带的threshold函数，对CR分量进行二值化，算法为自适应阈值的OTSU算法
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
    hand_contours.push_back(hand);
    drawContours(frame, hand_contours, 0, Scalar(0, 0, 255), 2, 8);
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

void recordPoint(Point2f &point, std::vector<Point2f> &track)
{
    Point2f current_point(0, 0);
    if (abs(point.x - current_point.x) > 3 || abs(point.y - current_point.y) > 3)
    {
        current_point = point;
        track.push_back(current_point); // push the current point to the track vector
    }
}

void drawTrace(std::vector<Point2f> &track, Mat &frame)
{
    for (int i = 0; i < track.size(); i++)
    {
        circle(frame, track[i], 2, Scalar(100, 100, 100), -1);
    }
}

void shapeRecog(std::vector<Point2f> &track, std::vector<Point2f> &output, std::string &shape)
{
    approxPolyDP(track, output, 10, true);
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
    int wide = 480;
    int height = 640;
    cv::VideoCapture cap(0);
    cap.set(3, height);
    cap.set(4, wide);

    Point2f origin_point;
    Point2f current_point(0, 0);

    std::vector<Point2f> track;

    while (cap.isOpened())
    {
        Mat frame;
        cap >> frame; //读取画面
        Mat binary;
        getSkin(frame, binary); //将图片二值化，并初步提取出手
        std::vector<Point> hand;
        getHand(binary, hand); //找到二值图像的最大边界，应该就是手了。
        std::vector<std::vector<Point>> hand_contours;
        drawHand(frame, hand, hand_contours);
        // get the center of hand
        std::vector<Point2f> mc(hand.size());
        calcMoment(hand, mc);
        origin_point = mc[0];
        // if the current point is 3 pixel away from the origin point, then update the current point
        recordPoint(origin_point, track);
        // draw the track
        drawTrace(track, frame);
        // center of the hand
        circle(frame, mc[0], 5, Scalar(255, 0, 0), -1);
        // get the bounding area of hand
        Rect rect = boundingRect(hand);                  // boundingRect 返回手的最小矩形区域
        rectangle(frame, rect, Scalar(0, 255, 0), 2, 8); // draw the boundingRect
        // Mirror the frame symmetrically
        Mat frame_mirror;
        flip(frame, frame_mirror, 1);

        imshow("origin", frame_mirror);

        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }
}