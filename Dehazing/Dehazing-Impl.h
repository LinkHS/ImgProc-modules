#pragma once

#include "opencv.hpp"

class DehazingImpl
{
public:

    virtual                             ~DehazingImpl();
    virtual void                         RestoreImage(cv::Mat& mDst) = 0;

protected:
    /**
     * Dehazing parameters
     */
    int	                             	ITranBlkSize;   //Block size for transmission estimation
    int		                        IGuidBlkSize;   //Block size for guided filter
    float                               ILambda;        //Loss cost

protected:
    /* image info */
    //int                               Idepth;
    int                                 IWidth;
    int                                 IHeight;
    int                                 IAtmosLight[3]; //atmospheric light

    cv::Mat                             ImSrc;
    cv::Mat                             ImSpp;          //superposition sum of each row
    cv::Mat                             ImSqrSpp;       //superposition sum of square of each row
    cv::Mat                             ImTransmission; 

//#if DEBUG
    int                                 DEBUG_AirLigthtBlock[4];
//#endif

protected:
    virtual void                        cptSppANDSqrSqq(void) = 0;
    virtual void                        cptMeanStdDev(const int *pBlockTBLR, int *pMean, int *pStddev) = 0;
    virtual void                        LookForSubBlock_MiniStdDev(int minBlockSize, int *pBlockTBLR) = 0;
    virtual void                        AirlightEstimation(int minSize) = 0;
            void                        AirlightEstimationMono(int minSize);
            void                        TransmissionEstimation(void);
    virtual float                       NFTrsEstimation(int nStartX, int nStartY) = 0;
            void                        GuidedFilter(cv::Mat &mGuide, int r, float eps);
    //virtual void                        RestoreImage(cv::Mat& mDst) = 0;
    //virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class DehazingMono : public DehazingImpl
{                                       
public:                                 
                                        DehazingMono(cv::Mat &mSrc);
                                        
private:                                
    virtual void                        cptSppANDSqrSqq(void);
    virtual void                        cptMeanStdDev(const int *pBlockTBLR, int *pMean, int *pStddev);
    virtual void                        LookForSubBlock_MiniStdDev(int minBlockSize, int *pBlockTBLR);
    virtual void                        AirlightEstimation(int minSize);
//            void                        TransmissionEstimation(void);
    virtual float                       NFTrsEstimation(int nStartX, int nStartY);

	    void                        RestoreImage(cv::Mat& mDst);
};

