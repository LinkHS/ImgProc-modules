//
// Created by austin on 17/3/11.
//
#include "opencv2/opencv.hpp"

typedef unsigned char kz_pixel_t;

extern int CLAHE_main (uchar* pImage, int xRes, int yRes, int xGrids, int yGrids, float fCliplimit);

/// ./clahe_test test.jpg
int main(int argc, char* argv[])
{
    std::cout << "argc = " << argc << std::endl;
    std::cout << argv[0] << std::endl;
    std::cout << argv[1] << std::endl;

    cv::Mat mSrc = cv::imread(argv[1]);
        // Split the image into different channels

    cv::Mat mYUV, mTemp, mDst;
    cv::cvtColor(mSrc, mYUV, cv::COLOR_BGR2YUV);
    
    std::vector<cv::Mat> yuvChannels(3);
    cv::split(mYUV, yuvChannels);

    int64 e1 = cv::getTickCount();
    CLAHE_main(yuvChannels[0].data, yuvChannels[0].cols, yuvChannels[0].rows, 8, 8, 0);
    int64 e2 = cv::getTickCount();

    double time = (e2 - e1)/ cv::getTickFrequency();
    std::cout << "time" << time << std::endl;

    cv::merge(yuvChannels, mTemp);
    cv::cvtColor(mTemp, mDst, cv::COLOR_YUV2BGR);

#if 0
    cv::imshow("mSrc", mSrc);
    cv::imshow("mDst", mDst);
    cv::waitKey(0);
#else    
    cv::imwrite("./dst.jpg", mDst);
#endif

    return 0;
}