#include "myDehazing.h"
#include "Dehazing-Impl.h"
#include "common/debug.h"
#include "basic/GuidedFilter.h"

#define MY_LOGD(fmt, ...)    do { printf("[%s] "fmt, __FUNCTION__, __VA_ARGS__); printf("\n"); } while (0)
#define MY_LOGW              MY_LOGD
#define MY_LOGE              MY_LOGD

#define DEBUG                1

using namespace cv;

DehazingImpl::~DehazingImpl()
{
    MY_LOGD("IAtmosLight: B %d, G %d, R %d", IAtmosLight[0], IAtmosLight[1], IAtmosLight[2]);

    /**
     * Draw a boundry on the block which is used to estimate atmospheric light
     */
#if DEBUG
    Mat mSrc_Debug;
    ImSrc.copyTo(mSrc_Debug);

    int row_start = DEBUG_AirLigthtBlock[0];
    int row_end = DEBUG_AirLigthtBlock[1];
    int col_start = DEBUG_AirLigthtBlock[2];
    int col_end = DEBUG_AirLigthtBlock[3];

    MY_LOGD("row_start %d, row_end %d, col_start %d, col_end %d", row_start, row_end, col_start, col_end);

    uchar* pSrc = mSrc_Debug.ptr<uchar>(row_start);
    for ( int j=col_start; j<col_end; j++)
        pSrc[j] = 0;

    pSrc = mSrc_Debug.ptr<uchar>(row_end-1);
    for ( int j=col_start; j<col_end; j++)
        pSrc[j] = 0;

    for (int i=row_start; i<row_end; i++)
    {  
        pSrc = mSrc_Debug.ptr<uchar>(i);
        pSrc[col_start] = 0;
        pSrc[col_end-1] = 0;
    }

	imshow("AtmosLight block", mSrc_Debug);
#endif

#if DEBUG
    Mat mUchar;
    ImTransmission.convertTo(mUchar, CV_8UC1, 255);
    imshow("Transmission", mUchar);
#endif
}


/**
 * Description: Estiamte the transmission.
 * 
 * @author J6000275 (2016/2/15)
 */
void DehazingImpl::TransmissionEstimation(void)
{
    float fTrans;

    if ( (ImTransmission.cols != IWidth) || (ImTransmission.rows != IHeight) )
        ImTransmission.create( IHeight, IWidth, CV_32FC1 );

    //if ( ImTransmission.isContinuous() == false )
    //    MY_LOGE("ImTransmission is not continuous");

    for( int nY=0; nY<IHeight; nY+=ITranBlkSize )
    {
        for( int nX=0; nX<IWidth; nX+=ITranBlkSize )
        {
            fTrans = NFTrsEstimation( nX, nY );
            //float fTrans = NFTrsEstimationColor( nX, nY );
            //printf("%.2f ", fTrans);
            for( int nYstep=nY; nYstep<nY+ITranBlkSize; nYstep++ )
            {
                float *pfTransmission = (float*)(ImTransmission.data + ImTransmission.step.p[0]*nYstep);

                for( int nXstep=nX; nXstep<nX+ITranBlkSize; nXstep++ )
                {
                    pfTransmission[nXstep] = fTrans;
                }
            }
        }
        //MY_LOGD("\nnY=%d\n",nY);
    }
}

void DehazingImpl::GuidedFilter(Mat &mGuide, int r, float eps)
{
    class GuidedFilter gf(mGuide, r, eps);

    Mat mFilted = gf.filter(ImTransmission, -1);

    ImTransmission = mFilted;
}



/* 
	Constructor: dehazing constructor using various options

	Parameters: 
		nW - width of input image
		nH - height of input image
		nTBLockSize - block size for transmission estimation 
		bPrevFlag - boolean for temporal cohenrence of video dehazing
		bPosFlag - boolean for postprocessing
		fL1 - information loss cost parameter (regulating)
		fL2 - temporal coherence paramter
		nGBlock - guided filter block size
*/
Dehazing::Dehazing(Mat &mSrc)
{
#if 0
    extern void SavePicAsTxt(Mat &mSrc, char* path);
    SavePicAsTxt(mSrc, "d:/mSrc.txt");
#endif

    impl_ = new DehazingMono(mSrc);
}

Dehazing::~Dehazing()
{
    delete impl_;
}

void Dehazing::doDehazing(cv::Mat &mDst)
{
    impl_->RestoreImage(mDst);
    imshow("res", mDst);
}


