#include "myDehazing.h"
#include "Dehazing-Impl.h"
#include "common/debug.h"
#include "basic/GuidedFilter.h"

#define MY_LOGD(fmt, ...)    do { printf("[%s] "fmt, __FUNCTION__, __VA_ARGS__); printf("\n"); } while (0)
#define MY_LOGW              MY_LOGD
#define MY_LOGE              MY_LOGD

#define DEBUG                1

using namespace cv;

                                       
DehazingMono::DehazingMono(Mat &mSrc)
{
MY_LOGD("+");
    ImSrc = mSrc;
    IWidth = mSrc.cols;
    IHeight = mSrc.rows;

    ITranBlkSize = 16;
    ILambda = 5.0;

#if DEBUG
    MY_LOGD("IWidth = %d, IHeight = %d", IWidth, IHeight);
    MY_LOGD("ITranBlkSize = %d, ILambda = %f", ITranBlkSize, ILambda);
#endif

    cptSppANDSqrSqq();
    AirlightEstimation(200);
    TransmissionEstimation();

    GuidedFilter(ImSrc, 40, (float)0.001);
}

void DehazingMono::cptSppANDSqrSqq(void)
{
MY_LOGD("+");
    int nRows = IHeight;
    int nCols = IWidth;

    /* Init ImSpp and ImSqrSpp if needed */
    if ( (ImSpp.cols != nCols) || (ImSpp.rows != nRows) )
        ImSpp.create( nRows, nCols, CV_32SC1 );

    if ( (ImSqrSpp.cols != nCols) || (ImSqrSpp.rows != nRows) )
        ImSqrSpp.create( nRows, nCols, CV_32SC1 );

    for (int i=0; i<nRows; i++)
    {  
        uchar* pSrc = ImSrc.ptr<uchar>(i);
        int *pSppRow    = ImSpp.ptr<int>(i);
        int *pSppSqrRow = ImSqrSpp.ptr<int>(i);

        pSppRow[0]    = pSrc[0];
        pSppSqrRow[0] = pSrc[0]*pSrc[0];

        for( int j=1; j<nCols; j++)
        {
            pSppRow[j]    = pSppRow[j-1]    + pSrc[j];
            pSppSqrRow[j] = pSppSqrRow[j-1] + pSrc[j]*pSrc[j];
        }
    }

#if 0
    extern void SaveMatAsTxt_int(Mat &mSrc, char* path);
    SaveMatAsTxt_int(ImSpp, "d:/ImSpp.txt");
#endif
}

/*
	Parameter: 
        @param pPointTBLR: top, bottom, left, right
        @param pMean output parameter: calculated mean value.
        @param pStddev output parameter: calculateded standard deviation.
*/
void DehazingMono::cptMeanStdDev(const int *pBlockTBLR, int *pMean, int *pStddev)
{
//MY_LOGD("+");
    /* block Top, block Bottom, block Left, block Right */
    int bT=pBlockTBLR[0], bB=pBlockTBLR[1], bL=pBlockTBLR[2], bR=pBlockTBLR[3];
MY_LOGD("bT %d, bB %d, bL %d, bR %d", bT, bB, bL, bR);

    int nRows = bB - bT; //bottom - top + 1
    int nCols = bR - bL; //right - left + 1
    int bsize = nRows * nCols; //block size

    uint lSum=0, lSqrSum=0;     //local sum, local sum of suqare
    /* to avoid overflow of local */
    int mult_bsize = 0;
    int mod_bsize = 0;

    if ( bL == 0 )
    {
        for (int i = bT; i < bB; i++)
        {  
            int *pSppRow    = ImSpp.ptr<int>(i);
            int *pSppSqrRow = ImSqrSpp.ptr<int>(i);

            lSum    += pSppRow[bR-1];
            lSqrSum = pSppSqrRow[bR-1];

            mult_bsize += lSqrSum / bsize;
            mod_bsize += lSqrSum % bsize;
        }
    }else
    {
        for (int i = bT; i < bB; i++)
        {  
            int *pSppRow    = ImSpp.ptr<int>(i);
            int *pSppSqrRow = ImSqrSpp.ptr<int>(i);

            lSum    += pSppRow[bR-1]   - pSppRow[bL-1];
            lSqrSum = pSppSqrRow[bR-1] - pSppSqrRow[bL-1];

            mult_bsize += lSqrSum / bsize;
            mod_bsize += lSqrSum % bsize;
        }
    }

	uint lMean = lSum / bsize;
    /* (lSqrSum-(lMean*lSum)) / bsize */
    uint lVar = mult_bsize + mod_bsize/bsize - lMean*lMean;
    *pMean = lMean;
    *pStddev = (int)sqrt(lVar);

MY_LOGD("bsize %d, lSum %d, lMean %d, lVar %d, mult_bsize %d, mod_bsize %d, stddev %d", bsize, lSum, lMean, lVar, mult_bsize, mod_bsize, *pStddev );
}


/**
 * Look for the sub block, which has minimum std-dev.
 * 
 * @author J6000275 (2016/2/1)
 * 
 * @param minBlockSize: threshold of minimum block size, default=200
 * @param pBlockTBLR(output): boundry of the detected block, 
 *                   T->Top Row, B->Bottom Row, L->Left Col, R->Right Col
 */
void DehazingMono::LookForSubBlock_MiniStdDev(int minBlockSize, int *pBlockTBLR)
{
MY_LOGD("+");
    //TODO: pBlockTBLR != NULL

    /* block Left, block Right, block Top, block Bottom */
    int bL=0, bR=IWidth, bT=0, bB=IHeight;

    int halfWid = IWidth / 2; //half of the width
    int halfHei = IHeight / 2; //half of the height

    /* compare to threshold(50) --> bigger than threshold, divide the block */
    while ( halfWid*halfHei > 1000 )
    {
        int nMaxIndex;
        float afScore;
        float nMaxScore = 0;
		int dpScore[4];

        halfWid = (bR - bL) / 2; //half of the width
        halfHei = (bB - bT) / 2; //half of the height

        for( int i=0; i<4; i++ )
        {
            int PointTBLR[4];
            int mean, stddev;

            /*
            i=0, UpperLeft sub-block:  Rect(bL,         bT,         halfWid, halfHei)
            i=1, UpperRight sub-block: Rect(bL+halfWid, bT,         halfWid, halfHei)
            i=2, LowerLeft sub-block:  Rect(bL,         bT+halfHei, halfWid, halfHei)
            i=3, LowerRight sub-block: Rect(bL+halfWid, bT+halfHei, halfWid, halfHei)
            */  
            PointTBLR[0] = bT + i/2*halfHei;       //top of sub-block
            PointTBLR[1] = PointTBLR[0] + halfHei; //bottom of sub-block
            PointTBLR[2] = bL + i%2*halfWid;       //left of sub-block
            PointTBLR[3] = PointTBLR[2] + halfWid; //right of sub-block

            /* compute the mean and std-dev in the sub-block */
            cptMeanStdDev(PointTBLR, &mean, &stddev);
            dpScore[i] = mean - stddev;
        }

        for( int i=0; i<4; i++ )
        {
            afScore = (float)dpScore[i];
            MY_LOGD("afScore %f", afScore);
            if( afScore > nMaxScore )
            {
                nMaxScore = afScore;
                nMaxIndex = i;
            }
        }

        /* Set new region */
        switch (nMaxIndex)
        {
        /* upper left */
        case 0: 
            bR = bR - halfWid;
            bB = bB - halfHei;
            break;
        /* upper right */
        case 1:
            bL = bL + halfWid;
            bB = bB - halfHei;
            break;
        /* lower left */
        case 2:
            bR = bR - halfWid;
            bT = bT + halfHei;
            break;
        /* lower right */
        case 3:
            bL = bL + halfWid; 
            bT = bT + halfHei;
            break;
        }  

        MY_LOGD("L %d, R %d, T %d, B %d, nMaxIndex %d, halfWid %d, halfHei %d", bL, bR, bT, bB, nMaxIndex, halfWid, halfHei);
    }

    /* copy the block coordinate */
    pBlockTBLR[0] = bT;
    pBlockTBLR[1] = bB;
    pBlockTBLR[2] = bL;
    pBlockTBLR[3] = bR;
}

/*
	Function: AirlightEstimation
	Description: estimate the atmospheric light value in a hazy image.
			     it divides the hazy image into 4 sub-block and selects the optimal block, 
				 which has minimum std-dev and maximum average value.
				 *Repeat* the dividing process until the size of sub-block is smaller than 
				 pre-specified threshold value. Then, We select the most similar value to
				 the pure white.
				 IT IS A RECURSIVE FUNCTION.
	Parameter: 
		imInput - input image (cv iplimage)
	Return:
		m_anAirlight: estimated atmospheric light value
 */
void DehazingMono::AirlightEstimation(int minSize)
{
MY_LOGD("+");
    int BlockTBLR[4];
    int bT, bB, bL, bR;
    int nCols;
    int nMaxDistance = 0;

    LookForSubBlock_MiniStdDev(200, BlockTBLR);
    bT=BlockTBLR[0], bB=BlockTBLR[1], bL=BlockTBLR[2], bR=BlockTBLR[3];
    nCols = bR - bL + 1; //mono channel
   
    /* select the atmospheric light value in the sub-block */
    for( int j=bT; j<bB; j++ )
    {
        uchar *pSrc = ImSrc.ptr<uchar>(j);
        
        for( int i=bL; i<bL+nCols; i++ )
        {
            if( nMaxDistance < pSrc[i] )
                nMaxDistance = pSrc[i];
        }
    }
    IAtmosLight[0] = nMaxDistance;

#if DEBUG
    DEBUG_AirLigthtBlock[0] = BlockTBLR[0];
    DEBUG_AirLigthtBlock[1] = BlockTBLR[1];
    DEBUG_AirLigthtBlock[2] = BlockTBLR[2];
    DEBUG_AirLigthtBlock[3] = BlockTBLR[3];
#endif
}


float DehazingMono::NFTrsEstimation(int nStartX, int nStartY)
{
	int nCounter;	
	int nX, nY;		
	int nEndX;
	int nEndY;

	int nOut;						// Restored image
	int nSquaredOut;				// Squared value of restored image
	int nSumofOuts;					// Sum of restored image
	int nSumofSquaredOuts;			// Sum of squared restored image
	float fTrans, fOptTrs;			// Transmission and optimal value
	int nTrans;						// Integer transformation 
	int nSumofSLoss;				// Sum of loss info
	float fCost, fMinCost, fMean;	 
	int nNumberofPixels, nLossCount;

	nEndX = std::min(nStartX+ITranBlkSize, IWidth); // End point of the block
	nEndY = std::min(nStartY+ITranBlkSize, IHeight); // End point of the block

	nNumberofPixels = (nEndY-nStartY)*(nEndX-nStartX);	

	fTrans = 0.3f;	// Init trans is started from 0.3
	nTrans = 427;	// Convert transmission to integer 

	for(nCounter=0; nCounter<7; nCounter++)
	{
		nSumofSLoss = 0;
		nLossCount = 0;
		nSumofSquaredOuts = 0;
		nSumofOuts = 0;

		for(nY=nStartY; nY<nEndY; nY++)
		{
            /* uchar* pSrc = mSrc.ptr<uchar>(nY); */
			uchar* pSrc = (uchar*)(ImSrc.data + ImSrc.step.p[0] * nY);

			for(nX=nStartX; nX<nEndX; nX++)
			{
                /* (I-A)/t + A --> ((I-A)*k*128 + A*128)/128 */
				nOut = ((pSrc[nX] - IAtmosLight[0])*nTrans + 128*IAtmosLight[0])>>7;
				nSquaredOut = nOut * nOut;

				if(nOut>255)
				{
					nSumofSLoss += (nOut - 255)*(nOut - 255);
					nLossCount++;
				}
				else if(nOut < 0)
				{
					nSumofSLoss += nSquaredOut;
					nLossCount++;
				}
				nSumofSquaredOuts += nSquaredOut;
				nSumofOuts += nOut;
			}
		}
		fMean = (float)(nSumofOuts)/(float)(nNumberofPixels);  
		fCost = ILambda * (float)nSumofSLoss/(float)(nNumberofPixels) 
			- ((float)nSumofSquaredOuts/(float)nNumberofPixels - fMean*fMean); 
		
		if(nCounter==0 || fMinCost > fCost)
		{
			fMinCost = fCost;
			fOptTrs = fTrans;
		}

		fTrans += 0.1f;
		nTrans = (int)(1.0f/fTrans*128.0f);
	}

	return fOptTrs; 
}

/*
	Function: RestoreImage
	Description: Dehazed the image using estimated transmission and atmospheric light.
	Parameter: 
		imInput - Input hazy image.
	Return:
		imOutput - Dehazed image.
*/
void DehazingMono::RestoreImage(Mat& mDst)
{
    float atmosl;
    int nCols;
    uchar GammaLUT[256];

    mDst.create( IHeight, IWidth, ImSrc.type() );

	for(int nIdx=0; nIdx<256; nIdx++)
	{
		GammaLUT[nIdx] = (uchar)(pow((float)nIdx/255, 0.7) * 255.0f);
	}

	atmosl = (float)IAtmosLight[0];
	nCols = IWidth * ImSrc.channels();

    // (2) I' = (I - Airlight)/Transmission + Airlight
    for( int nY=0; nY<IHeight; nY++ )
    {
        uchar* pLineSrc = ImSrc.ptr<uchar>(nY);
        uchar* pLineDst = mDst.ptr<uchar>(nY);
        float* pTransm = ImTransmission.ptr<float>(nY);

        for( int nX=0; nX<IWidth; nX++ )
        {
            // (3) Gamma correction using LUT
            float temp1 = (float)(pLineSrc[nX]) - atmosl;
            float temp2 = pTransm[nX];
            uchar temp3 = (uchar)saturate_cast<uchar>( temp1/temp2 + atmosl );

            pLineDst[nX+0] = GammaLUT[temp3];
        }
    }
}


