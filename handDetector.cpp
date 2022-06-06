#include <opencv2/opencv.hpp>

Mat getSkin(Mat& ImageIn)//获取皮肤的区域,返回二值化图像
{
	vector<Mat> r_g_b;//用于存放RGB分量
	split(ImageIn,r_g_b);//分离RGB分量,顺序为B,G,R

	Mat Binary = Mat::zeros(ImageIn.size(),CV_8UC1);

	Mat R = r_g_b[2];
	Mat G = r_g_b[1];
	Mat B = r_g_b[0];

	for (int i = 0; i < ImageIn.rows; i++)
	{
		for (int j = 0; j < ImageIn.cols; j++)
		{
			if (R.at<uchar>(i, j) > 95 && G.at<uchar>(i, j) > 40 && B.at<uchar>(i, j) > 20 &&
				R.at<uchar>(i, j) > G.at<uchar>(i, j) && R.at<uchar>(i, j) > B.at<uchar>(i, j) &&
				MyMax(R.at<uchar>(i, j), G.at<uchar>(i, j), B.at<uchar>(i, j)) - MyMin(R.at<uchar>(i, j), G.at<uchar>(i, j), B.at<uchar>(i, j)) > 15
				&& abs(R.at<uchar>(i, j) - G.at<uchar>(i, j)) > 15)
			{
				Binary.at<uchar>(i, j) = 255;
			}
		}
	}
	
	return Binary;
}

int main()
{
    cv::VideoCapture cap(0);
    cap.set(3,1080);
    cap.set(4,720);

    while (cap.isOpened())
    {
        cv::Mat frame;
        cap >> frame;
        cv::imshow("test", frame);
        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }

}