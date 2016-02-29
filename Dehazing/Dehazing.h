#pragma once

#include "opencv.hpp"

class DehazingImpl;

class Dehazing
{
public:
							Dehazing(cv::Mat &mSrc);
						   ~Dehazing();
    void                    doDehazing(cv::Mat &mDst);
private:
    DehazingImpl           *impl_;
};

