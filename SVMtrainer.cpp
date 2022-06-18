//
// Created by Msi-pc on 2022/6/13.
//

#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <stdio.h>
#include <opencv2/ml.hpp>
#include <fstream>
#include <opencv2/imgproc/imgproc_c.h>
using namespace std;
using namespace cv;







int main()
{
    float desc[6*14][15];
    int labels[6*14][1];
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
        }
    }
    //训练svm模型
    int train_sample_count = 6*14;//数据的数量
    int train_sample_size = 15;//数据的维度

    //load the training data into Mat
    Mat trainingDataMat(6*14,15,CV_32FC1,desc);
    Mat labelMat(6*14,1,CV_32SC1,labels);


    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::NU_SVR);
    svm->setKernel(ml::SVM::RBF);
    svm->setGamma(1./train_sample_size);
    svm->setNu(0.5);
    svm->setC(8);

    TermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_EPS,50000,0.001);
    svm->setTermCriteria(term_criteria);

    svm->trainAuto(trainingDataMat,ml::SampleTypes::ROW_SAMPLE, labelMat);
    svm->save(".//svm_model.xml");
}