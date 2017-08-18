#include <stdio.h>
#include <string.h>
#include <stdlib.h>			 /* To get prototypes of malloc() and free() */
#include <math.h> /* sqrt() */

/*************************************************************************************************/
#define ONLY_THIS_FILE 1

#if ONLY_THIS_FILE
  #if (__arm__) || (_WIN32)
    #define FIH_MAX_INT		(int)2147483647
  #elif (__aarch64__) || (_WIN64) || (__x86_64__)
    #define FIH_MAX_INT		(int)0x7FFFFFFFFFFFFFFF
  #else
    #pragma error("Unknow building environment")
  #endif
  
  typedef unsigned char uchar;	 /* for 8 bit-per-pixel images */
#else
  #include "fihimgproc_inc.h"
#endif
/*************************************************************************************************/

const int MAX_X_GRIDS = 32;	  /* max. # contextual regions in x-direction */
const int MAX_Y_GRIDS = 32;	  /* max. # contextual regions in y-direction */
const int histSize = 256;

static void Interpolate (uchar*, int, int*, int*, int*, int*, unsigned int, unsigned int);

static int MatData_CptVar(uchar *pSrc, int xRes, int yRes, float *pMean, float *pStddev, int xBlocks, int yBlocks)
{
#ifndef CPU_NUM
	const int cpus = 8;
#else
    const int cpus = CPU_NUM; 
#endif
	const int MAX_SUM = FIH_MAX_INT - 255;      // MAX_INT - 255
	const int MAX_SQRSUM = FIH_MAX_INT - 255*255; // MAX_INT - 255*255

	if( (xRes%xBlocks) || (yRes%yBlocks)) return -1; // xRes/yRes is no multiple of xBlocks/yBlocks
	if( (pSrc==NULL) || (pMean==NULL) || (pStddev==NULL) ) return -2; //fatal error
	if( (xBlocks == 0) || (yBlocks == 0) ) return -3;

	/*
	 * Calculate mean and stddev of each block(total block number is xBlocks*yBlocks)
	 */	
	int subw = xRes / xBlocks; //sub width
	int subh = yRes / yBlocks; //sub height
	int subsize = subw * subh;
#pragma omp parallel for
	for(int tblV=0; tblV<yBlocks; tblV++) //tblV: table vertical
	{
		int mem[16*3];
		int *pSum = mem;
		int *pSqrMeanQuot = mem + xBlocks; //square sum mean quotient
		int *pSqrMeanRema = pSqrMeanQuot + xBlocks; //square sum mean remain
		if( xBlocks > 16 ) //mem[32] is not enough
		{
			pSum = new int[xBlocks * 3];
			pSqrMeanQuot = pSum + xBlocks;
			pSqrMeanRema = pSqrMeanQuot + xBlocks;
		}

		/* clear mem */
		for(int i=0; i<xBlocks; i++)
		{
			pSum[i] = 0;
			pSqrMeanQuot[i] = 0;
			pSqrMeanRema[i] = 0;
		}

		uchar *pData = pSrc + subh*xRes*tblV;
		for(int row=0; row<subh; row++)
		{
			for (int xblk=0; xblk<xBlocks; xblk++)
		    {
				int sum = 0;
				int sqrsum = 0;
				int sqrmeanquot = 0; //square sum mean quotient
				int sqrmeanrema = 0; //square sum mean remain

				for(int x=0; x<subw; x++)
				{
					int value = *pData++; 
					sum += value;
					sqrsum += value*value;
					if( sqrsum >= MAX_SQRSUM )
					{
						sqrmeanquot += sqrsum / subsize;
						sqrmeanrema += sqrsum % subsize;
						sqrsum = 0;
					}
				}
				pSum[xblk] += sum;
				pSqrMeanQuot[xblk] += sqrmeanquot + sqrsum/subsize;
				pSqrMeanRema[xblk] += sqrmeanrema + sqrsum%subsize;;
			}
		}	

		/* mean and stddev */
		for (int xblk=0; xblk<xBlocks; xblk++)
		{
			float mean = pSum[xblk] / (float)subsize;
			float sqrmean = pSqrMeanQuot[xblk] + pSqrMeanRema[xblk]/(float)subsize;
			pMean[tblV*xBlocks + xblk] = mean;
			pStddev[tblV*xBlocks + xblk] = sqrt(sqrmean - mean*mean);
		}

	    if( xBlocks > 16 ) delete pSum;
	}

	return 0;
}


static void SmoothClipTable(float *pClipTable, int blkX, int blkY)
{
	for( int j=0; j<blkY; j++ )
	{
		int rowUp = j - 1;
		int rowCr = j ;
		int rowDn = j + 1;
		rowUp = (rowUp < 0) ? (j + 1) : rowUp;
		rowDn = (rowDn >= blkY) ? (j - 1) : rowDn;

		for( int i=0; i<blkX; i++ )
		{
			int colLf = i - 1;
			int colCr = i ;
			int colRt = i + 1;
			colLf = (colLf < 0) ? (i + 1) : colLf;
			colRt = (colRt >= blkX) ? (i - 1) : colRt;

			float sum = pClipTable[rowUp*blkX + colLf] + pClipTable[rowUp*blkX + colCr] + pClipTable[rowUp*blkX + colRt];
			sum += pClipTable[rowCr*blkX + colLf] + pClipTable[rowCr*blkX + colCr] + pClipTable[rowCr*blkX + colRt];
			sum += pClipTable[rowDn*blkX + colLf] + pClipTable[rowDn*blkX + colCr] + pClipTable[rowDn*blkX + colRt];
			pClipTable[j*blkX + i] = sum / 9;
		}
	}
}

static void MakeClipTable(uchar *pYChannel, int xRes, int yRes, float *pClipTable, int blkX, int blkY)
{
    float *Memory = new float[blkX*blkY*2];
    float *Mean = Memory;
	float *Stddev = Memory + blkX*blkY;
		
	MatData_CptVar(pYChannel, xRes, yRes, Mean, Stddev, blkX, blkY);
	
//TODO #pragma omp parallel for
	for( int j=0; j<blkY; j++ )
	{
	    float *pMeanTable = Mean + j*blkX;
		float *pStddevTable = Stddev + j*blkX;
		float *pTable = pClipTable + j*blkX;
	    for( int i=0; i<blkX; i++ )
		{
			float brightness = *pMeanTable++;
			float offset = (8.0 - log10(brightness+10.0)*3.3) * 0.22;
			float clip = *pStddevTable++;
			*pTable++ = log10(clip+10) - 1.0 + offset;
		}
	}
	
	SmoothClipTable(pClipTable, blkX, blkY);
	
	delete Memory;
}

/************************** main function CLAHE ******************/
int CLAHE_main (uchar* pImage, int xRes, int yRes, int xGrids, int yGrids, float fCliplimit)
{
    int *pulMapArray; /* pointer to histogram and mappings*/
    float *clipTable; /* Allocate in pulMapArray, size is xGrids*yGrids */
	
    if ( (xGrids > MAX_X_GRIDS) || (yGrids > MAX_Y_GRIDS) ) return -1;	   /* # of regions x/y-Grids too large */
	if ( (xGrids < 2) || (yGrids < 2) ) return -2;                         /* at least 4 contextual regions required */
    if ( (xRes % xGrids) || (yRes % yGrids) ) return -3;	  /* x/y-resolution no multiple of x/yGrids */

    int wGrid = xRes / xGrids;  /* width of grid */
    int hGrid = yRes / yGrids;  
    int GridPixels = wGrid * hGrid;
	
    pulMapArray = new int[xGrids*yGrids*histSize + xGrids*yGrids]; /* xGrids*yGrids is the size of clipTable */
    if (pulMapArray == NULL) return -4;	  /* Not enough memory! */
	memset(pulMapArray, 0, sizeof(int)*xGrids*yGrids*histSize);

    /* Calculate actual cliplimit */
    int ClipLimit = (int)(fCliplimit * GridPixels / histSize);
    ClipLimit = (ClipLimit < 1) ? 1 : ClipLimit;
	if( fCliplimit == 0 )
	{
	    clipTable = (float *)pulMapArray + xGrids*yGrids*histSize;
		MakeClipTable(pImage, xRes, yRes, clipTable, xGrids, yGrids);
	}

    /* Calculate greylevel mappings for each contextual region */
	const float lutScale = 255.0 / GridPixels;	

#pragma omp parallel for
	for (int j = 0; j < yGrids; j++)
	{
		uchar *pImPointer = pImage + (hGrid*j)*xRes; /* skip lines, set pointer */

		for (int i = 0; i < xGrids; i++)
		{
			if( fCliplimit == 0 )
			{
				float clip = (float)(clipTable[j*xGrids+i]);
				ClipLimit = (int)(clip * GridPixels / histSize);
			}
				
			/* pulHist = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)]; */
			int *tileHist = &pulMapArray[histSize * (j*xGrids + i)];
			uchar* pTile = pImPointer + i*wGrid;

			/* MakeHistogram */
			for (int i=0; i<hGrid; i++)
			{
				for( int j=0; j<wGrid; j++)
					tileHist[*pTile++]++;
				pTile += xRes - wGrid;
			}

			/* ClipHistogram */
			// how many pixels were clipped
			int clipped = 0;
			for (int i = 0; i < histSize; ++i)
			{
				if (tileHist[i] > ClipLimit)
				{
					clipped += tileHist[i] - ClipLimit;
					tileHist[i] = ClipLimit;
				}
			}

			// redistribute clipped pixels
			int redistBatch = clipped / histSize;
			int residual = clipped - redistBatch * histSize;

			for (int i = 0; i < histSize; ++i)
				tileHist[i] += redistBatch;

			for (int i = 0; i < residual; ++i)
				tileHist[i]++;
			
			/* MapHistogram */
			// calc Lut
			int sum = 0;
			for (int i = 0; i < histSize; ++i)
			{
				sum += tileHist[i];
				//tileLut[i] = cv::saturate_cast<T>(sum * lutScale_);
				tileHist[i] = (int)(sum * lutScale);
			}
		}		  
	}
	
    /* Interpolate greylevel mappings to get CLAHE image */
  #pragma omp parallel for //private(uiSubX, uiSubY) 
    for (int uiY=0; uiY<yGrids; uiY++)
	{
        unsigned int uiSubX, uiSubY; /* size of context. reg. and subimages */
        unsigned int uiXL, uiXR, uiYU, uiYB; /* auxiliary variables interpolation routine */
        int *pulLU, *pulLB, *pulRU, *pulRB; /* auxiliary pointers interpolation */
        uchar *pImPointer = pImage;

		if (uiY == 0) {					  /* special case: top row */
			uiSubY = hGrid >> 1;  uiYU = 0; uiYB = 0;
		}
        else {					  /* default values */
            uiSubY = hGrid; uiYU = uiY - 1; uiYB = uiYU + 1;
            pImPointer += (hGrid*uiY - (uiSubY>>1)) * xRes;
        }

        for (int i=0; i<2; i++) /* only loop twice when uiY == 0 */
        {
            for (int uiX = 0; uiX <= xGrids; uiX++)
    		{
    			if (uiX == 0) {				  /* special case: left column */
    				uiSubX = wGrid >> 1; uiXL = 0; uiXR = 0;
    			}
    			else if (uiX == xGrids) {	  /* special case: right column */
                    uiSubX = wGrid >> 1;  uiXL = xGrids - 1; uiXR = uiXL;
                }
                else {					      /* default values */
                    uiSubX = wGrid; uiXL = uiX - 1; uiXR = uiXL + 1;
                }

    			pulLU = &pulMapArray[histSize * (uiYU * xGrids + uiXL)];
    			pulRU = &pulMapArray[histSize * (uiYU * xGrids + uiXR)];
    			pulLB = &pulMapArray[histSize * (uiYB * xGrids + uiXL)];
    			pulRB = &pulMapArray[histSize * (uiYB * xGrids + uiXR)];
    			Interpolate(pImPointer,xRes,pulLU,pulRU,pulLB,pulRB,uiSubX,uiSubY);
    			pImPointer += uiSubX;			  /* set pointer on next matrix */
    		}	

    		if (uiY == 0) {					 
				/* special case: top row, to process bottom row */
                uiSubY = hGrid >> 1; uiYU = yGrids-1; uiYB = uiYU;
                pImPointer = pImage + (hGrid*yGrids - uiSubY) * xRes;
    		}else
                break;
        }
    }

    delete pulMapArray;			      /* free space for histograms */
    return 0;						  /* return status OK */
}

void Interpolate (uchar * pImage, int uiXRes, int * pulMapLU,
     int * pulMapRU, int * pulMapLB,  int * pulMapRB,
     unsigned int uiXSize, unsigned int uiYSize)
/* pImage      - pointer to input/output image
 * uiXRes      - resolution of image in x-direction
 * pulMap*     - mappings of greylevels from histograms
 * uiXSize     - uiXSize of image submatrix
 * uiYSize     - uiYSize of image submatrix
 * pLUT	       - lookup table containing mapping greyvalues to bins
 * This function calculates the new greylevel assignments of pixels within a submatrix
 * of the image with size uiXSize and uiYSize. This is done by a bilinear interpolation
 * between four different mappings in order to eliminate boundary artifacts.
 * It uses a division; since division is often an expensive operation, I added code to
 * perform a logical shift instead when feasible.
 */
{
    const unsigned int uiIncr = uiXRes-uiXSize; /* Pointer increment after processing row */
    uchar GreyValue; unsigned int uiNum = uiXSize*uiYSize; /* Normalization factor */

    unsigned int uiXCoef, uiYCoef, uiXInvCoef, uiYInvCoef, uiShift = 0;

    for (uiYCoef = 0, uiYInvCoef = uiYSize;  uiYCoef < uiYSize;  uiYCoef++, uiYInvCoef--, pImage+=uiIncr)
    {
        for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize; uiXCoef++, uiXInvCoef--)
        {
            GreyValue = *pImage;		   /* get histogram bin value */
            *pImage++ = (uchar ) ((uiYInvCoef * (uiXInvCoef*pulMapLU[GreyValue]
                                      + uiXCoef * pulMapRU[GreyValue])
                                      + uiYCoef * (uiXInvCoef * pulMapLB[GreyValue]
                                      + uiXCoef * pulMapRB[GreyValue])) / uiNum);
        }
    }
}
