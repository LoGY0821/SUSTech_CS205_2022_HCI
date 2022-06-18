//
// Created by Msi-pc on 2022/6/17.
//
#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <opencv2/imgproc/imgproc_c.h>
using namespace cv;
using namespace std;


float desc[6*14][15];
int labels[6*14][1];
//-----调参窗口的回调函数-------------------------------------//
int minH = 2, maxH = 13;//肤色分割阈值白天建议3 14
int minR = 95, minG = 40, minB = 20, max_min = 15, absR_G= 10;
int minCr = 138, maxCr = 243, minCb = 77, maxCb = 127;
int ecl_x = 113, ecl_y = 156, leng_x = 24, leng_y = 23, ang =43;
int match_number = -1, temp_number = -1;//第几种手势中的第几个模板

//图片编号
int num = 0, flag = 0, hand_num = 0 ;//调试时用来保存图片
Mat frame, frameH, frameHSV, frameYCrCb; //不同颜色空间下的图片
Mat  RIOframe, RIOresult; //二值化得到的图像，识别出的皮肤区域，最终结果,将结果显示在原图
Mat allContRIO, delContRIO,delContframe; //所有轮廓二值图片， 筛选后轮廓二值图片， 筛选后轮廓的RGB图片
float fd[32];//提取到的傅里叶描述子

vector <Mat> RGBchannels, HSVchannels;     //RGB通道分离,HSV通道分离
vector< vector<Point> > mContoursProc;  //当前图片的轮廓
vector< vector< vector<Point>> > mContoursLib; //模板库轮廓5*6条
vector< vector< Mat > > tempImageLib;  //模板库照片


//-----6种肤色识别方法-------------------------------------//
void hand_YCbCr();

//-------手势识别的功能函数----------------------------------//
void find_contours(Mat srcImage);//提取二值化图形的边界
void calcute_fft();//计算傅里叶描述子，这里没有用到
void draw_result();//得到匹配结果
void trainSVM();//训练svm模型
void predictSVM();//根据svm模型识别手势
using namespace cv;
int main()
{
    //训练svm模型
    //trainSVM();
    VideoCapture capture(0);
    while (true)
    {
        //获取图片帧
        capture >> frame;
        flag++;
        if(flag % 3 == 0)
        {
            //break;
            continue;
        }

        if (true == frame.empty())
        {
            cout << "get no frame" << endl;
            break;
        }
        resize(frame, frame, Size(frame.cols*0.3, frame.rows*0.3));//降采样

        namedWindow("1.原始图片", WINDOW_NORMAL);
        imshow("1.原始图片",frame);

        //--------------------------滤波处理------------------------------------------//
        //medianBlur(frameH, frameH, 5);	//中值滤波，用来去除椒盐噪声
        //GaussianBlur(frame, frame, Size(7, 7), 1, 1);// 高斯滤波，用来平滑图像
        //namedWindow("2.滤波后的图像", CV_WINDOW_NORMAL);
        //imshow("2.滤波后的图像",frame);

        hand_YCbCr();

        //----------------------------------------形态学运算-----------------------------//
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5)); //函数返回指定形状和尺寸的结构元素
        //morphologyEx(RIOresult, RIOresult, MORPH_CLOSE, kernel);//函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换
        //morphologyEx(RIOresult, RIOresult, MORPH_OPEN, kernel);//函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换
        namedWindow("4.形态学处理后的图片", WINDOW_NORMAL);
        imshow("4.形态学处理后的图片", RIOframe);// 显示肤色分割之后的图片


        //----------------------------------------检测边缘-----------------------------//
        find_contours(RIOresult);

        flag++;
        if(flag % 5 == 1)
        {
            num ++;
            calcute_fft( );//计算傅里叶描述子
        };

        //----------------------------------------SVM分类-----------------------------//
        predictSVM();
        //----------------------------------------绘制结果-----------------------------//
        draw_result();

        RIOframe.setTo(0);//否则会出现重影
        waitKey(1);
    }
    system("pause");
    return 0;
}

//--------手势识别的功能函数----------------------------------//
void hand_YCbCr()
{
    //----------------------------肤色分割调参窗口---------------------//
    //namedWindow("调参窗口", CV_WINDOW_AUTOSIZE);
    //createTrackbar("Cr_min", "调参窗口", &minCr, 255, trackBarMinCr);
    //createTrackbar("Cr_max", "调参窗口", &maxCr, 255, trackBarMaxCr);
    //createTrackbar("Cb_min", "调参窗口", &minCb, 255, trackBarMinCb);
    //createTrackbar("Cb_max", "调参窗口", &maxCb, 255, trackBarMaxCb);

    cvtColor(frame, frameYCrCb , COLOR_BGR2YCrCb);
    Mat tempresult = Mat(frame.rows, frame.cols, CV_8UC3, Scalar(0));

    inRange(frameYCrCb, Scalar(0,minCr,minCb), Scalar(255, maxCr, maxCb), RIOresult);
    //namedWindow("分割得到的RIO", CV_WINDOW_NORMAL);
    //imshow("分割得到的RIO",RIOresult);// 显示肤色分割之后的图片

    frame.copyTo(RIOframe, RIOresult);
    //namedWindow("提取到的区域RGB", CV_WINDOW_NORMAL);
    //imshow("提取到的区域RGB", RIOframe);// 显示肤色分割之后的图片
};






//提取二值化图形的边界
void find_contours(Mat srcImage)
{
    Mat imageProc = srcImage.clone();
    Size sz = srcImage.size();//尺寸
    Mat draw = Mat::zeros(sz, CV_8UC3);
    vector< vector<Point> > mContours;//轮廓点集
    vector< Vec4i > mHierarchy;//轮廓之间的索引号
    //findContours只能处理单通的二值化图像
    Mat binframe;
    if(srcImage.channels() == 3)
    {
        vector <Mat> channel;
        split(srcImage, channel);//分离通道
        binframe = channel[0].clone();
    }
    else
    {
        binframe = srcImage.clone();
    }

    findContours(binframe, mContours, mHierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));//只查找最外层轮廓

    mContoursProc.clear();//清空上次图像处理的轮廓

    if (mContours.size() > 0)
    {
        drawContours(draw, mContours, -1, Scalar(0, 0, 255), 2, 8 , mHierarchy);// 绘制所有轮廓()
        allContRIO = draw.clone();
        namedWindow("5.所有轮廓", WINDOW_NORMAL);
        imshow("5.所有轮廓", allContRIO);//显示所有轮廓
        //imwrite("data//frame6.jpg", allContRIO);

        double contArea = 0;
        double imageArea = sz.width * sz.height;
        const int SIZE = mContours.size();
        Rect bound; //Rect矩形类，矩形界限

        for (int i = 0; i < SIZE; i++)
        {
            contArea = contourArea(mContours[i]);
            if (contArea / imageArea < 0.015)// 过滤小面积的轮廓，原函数是0.015
            {
                continue;
            }
            mContoursProc.push_back(mContours[i]);//剩下的轮廓就是基本符合条件的轮廓，保存起来
        }

        draw = Scalar::all(0); //将矩阵所有元素赋值为某个值
        drawContours(draw, mContoursProc,0 , Scalar(0, 0, 255), 2, 8);
        delContRIO = draw.clone();
        namedWindow("6.过滤后的轮廓", WINDOW_NORMAL);
        imshow("6.过滤后的轮廓", delContRIO); //显示过滤后的轮廓
        //imwrite("data//frame7.jpg", delContRIO);

        delContframe = frame.clone();
        drawContours(delContframe, mContoursProc, -1, Scalar(0, 0, 255), 4, 8);
        namedWindow("8.原图的轮廓", WINDOW_NORMAL);
        imshow("8.原图的轮廓", delContframe); //显示过滤后的轮廓
        imwrite("data//frame8.jpg", delContframe);
        cout<<"lunjkuo:"<<mContoursProc.size()<<endl;
    }
}


//训练SVM模型
void trainSVM()
{
    //加载训练数据及标签
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j  < 14; j ++)
        {
            int num = i*14 + j;
            ostringstream oss;
            oss<< ".\\Testing\\" << i<<"\\"<<j <<".txt";
            ifstream data(oss.str().c_str());
            float da;
            int k = 0;
            while (data>>desc[num][k])
            {
                k++;
            }
            labels[num][0] = i;
            //cout<<i<<endl;
        }
    }
    //训练svm模型
    int train_sample_count = 6*14;//数据的数量
    int train_sample_size = 15;//数据的维度


    //--------------------------------------------------------------------
    CvMat *data_mat = NULL;//要训练的数据
    CvMat *class_mat = NULL;//数据的类别
    data_mat = cvCreateMat(train_sample_count,train_sample_size,CV_32FC1);
    class_mat = cvCreateMat(train_sample_count,1,CV_32FC1);
    for (int i = 0; i < train_sample_count; i++)
    {
        class_mat->data.fl[i]=labels[i][0];
        for (int j = 0; j < train_sample_size ; j++)
        {
            data_mat->data.fl[i*train_sample_size +j] = desc[i][j];
            //cout<< data_mat->data.fl[i*train_sample_size +j] <<endl;
        }
    }


    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::NU_SVR);
    svm->setKernel(ml::SVM::RBF);
    svm->setGamma(1./train_sample_size);
    svm->setNu(0.5);
    svm->setC(8);

    TermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_EPS,50000,0.001);
    svm->setTermCriteria(term_criteria);

    svm->trainAuto(cvarrToMat(data_mat),ml::SampleTypes::ROW_SAMPLE, cvarrToMat(class_mat));
    svm->save(".//svm_model.xml");
}

//计算轮廓傅里叶描述子
void calcute_fft()
{
//计算轮廓的傅里叶描述子
    Point p;
    int x, y, s;
    int i = 0,j = 0,u=0;
    s = (int)mContoursProc[0].size();
    Mat src1(Size(s,1),CV_8SC2);
    float f[9000];//轮廓的实际描述子
    //float fd[32];//归一化后的描述子，并取前15个
    for (u = 0; u < s; u++)
    {
        float sumx=0, sumy=0;
        for (j = 0; j < s; j++)
        {
            p = mContoursProc[0].at(j);
            x = p.x;
            y = p.y;
            sumx += (float)(x*cos(2*CV_PI*u*j/s) + y*sin(2 * CV_PI*u*j / s));
            sumy+= (float)(y*cos(2 * CV_PI*u*j / s) - x*sin(2 * CV_PI*u*j / s));
        }
        src1.at<Vec2b>(0, u)[0] = sumx;
        src1.at<Vec2b>(0, u)[1] = sumy;
        f[u] = sqrt((sumx*sumx)+(sumy*sumy));
    }
    //傅立叶描述字的归一化
    f[0] = 0;
    fd[0] = 0;
    for (int k = 2; k < 16; k++)
    {
        f[k] = f[k] / f[1];
        fd[k - 1] = f[k];
    }
}

//根据SVM模型预测结果
void predictSVM()
{
    int train_sample_size = 32;
    //CvSVM *pSvm = new CvSVM();

    cv::Ptr<cv::ml::SVM> psvm = cv::ml::SVM::create();

    psvm->load(".//svm_model.xml");
    CvMat *sample = cvCreateMat(1,train_sample_size,CV_32FC1);
    for (int i = 0; i < train_sample_size; i++)
    {
        sample->data.fl[i] = fd[i];
    }
    match_number = int (psvm->predict(cvarrToMat(sample)) + 0.5);//因为svm输出的是小数，要四舍五入一下
    cout << match_number<<endl;
}



void draw_result( )
{
    if (num <0)//如果未识别到任何数字则返回
    {
        return;
    }
    //在图像上绘制文字
    putText(delContframe , std::to_string(match_number), Point(50 ,50 ), FONT_HERSHEY_SIMPLEX, 2, Scalar( 255,0, 0), 4);
    namedWindow("手势识别结果", WINDOW_NORMAL);
    imshow("手势识别结果",delContframe);

};