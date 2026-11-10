#pragma once

#include <iostream>
#include <vector>
#include <algorithm> 
//#include <stdlib.h>
#include <fstream>


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/background_segm.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include "spray.h"

#include "EDCircles.h"


//#define DebugJJ

constexpr auto m_result_Max = 300;
constexpr auto candidatesBall_Num_Max = 30;

constexpr auto m_searchFrameRange_Max = 30;   //golfzone : 25-30

using namespace cv;
using namespace std;
using namespace cv::dnn;

struct candidateBall
{
	int x;
	int y;
	int size;
	int idx;  // 0 : bilateral filter blob, 1:threshold blob
};



struct analysisFactor
{
	int idx;
	int localImpactBGSCount;
	//float rightGradient;
	//int detectedBall;
	float flowEnergySum;
	candidateBall ballCandidates[candidatesBall_Num_Max];
    std::vector<candidateBall> paperCandidates;
	int ballCandidatesNum;
	//int impactIdx;
	//int topOfBackIdx;
	//int followIdx;
	Mat curBlueImg;
	Mat curRedImg;
	Mat curGreenImg;

	Mat curGrayImg;
	Mat preGrayImg;
	Mat fuGrayImg;
	Mat curImg;
	Mat curImgROI;

	//추가
	Mat m_preGrayBlurROIImg;
	Mat m_fuGrayBlurROIImg;
	Mat m_curGrayBlurROIImg;
	Mat curBGSImg;

	float vel;
	float accel;
	//float angle;
	bool ckSkipEndPoint;   //1=no end point 
	int state;   // 1 : from top to down direction, 2 : from down to top direction, 0 : init
	int nameNum;
	cv::Point endP;  //based on the m_roiEx size 

};

struct cciIdx
{
	Point p;
	int idx;
	int cc;
	float dist;   // 0 - 180 현재-다음 포인트 직선 기준 각  -----@--->
	bool curIn;    //1:in, o:out
};

class kGolfPose
{

public:
	
	kGolfPose(string cfg, string weight);
	~kGolfPose();

	void analysisFactorInit(analysisFactor& tmp);
	//threshold = 0.4
	void detectedPerson(Mat inputImg, float threshold, Rect& outRec, Point& personCenter);

	void setSideViewBallRegion(Rect personROI, Size inputImgSize, int marginWidth, int marginHeight, int setWidth, int setHeight, Point center, Rect& outRegion);
	void setFrontRightRegion(Rect personROI, Size inputImgSize, int marginWidth, int marginHeight, int setWidth, int setHeight, Point center, Rect& outRegion);


	std::ofstream m_csvFile;          // CSV 파일 스트림
	void recordMetrics(int idx);      // 프레임별 지표 기록 함수


	//setShadows : 1 = shadow on
	//initGrayImg : gray
	void initBGS(Rect ROI, Mat inputROIGrayImg, float shadowThres, bool setShadows, float backRatio, int blurParam, Ptr<BackgroundSubtractor>& kBGS, Mat& RoiRegion);

	void kBGS(Rect ROI, Ptr<BackgroundSubtractor> kBGS, Mat inputGrayImg, int blurParam, Mat& outBGSImg);

	//Mat initImg, Mat input : gray full image
	//filter : MORPH_ERODE -> MORPH_DILATE
	void kBGS(Mat initGrayImg, Mat inputGray, Rect ROI, bool checkShadow, float BGSRatio, Mat filterEleErode, Mat filterEleDilate, Mat& outBGSImg);

	// location : 0 = left, 1 = right
	// outBGSImg.cols*lengthRatio

	float ballCandidates(Mat ori, Mat ballImg, cv::KeyPoint keypointsBall, float thresIn, float thresOut);

	void searchRegion(Mat& img, int xx, int yy, float& sum, int& cc);

	bool findBallPosition();

	void deletekGolfPose();
	bool setROIs(Mat inputColor, Rect& persionROI, Rect& ballROI, Rect& MotionROI);
	void imgEnhance(Mat inputROI, Mat& out);
	void blobDetection(Mat inputImg, analysisFactor& ballCandidates);
	bool blobDetectionIdx(Mat inputImg, int curIdx);


	void kFlowEstimation(analysisFactor& flowBody);
	void kFlowEstimationIdx(int curIdx);



	float ballCandidatesHomogeneous(Mat ori, cv::KeyPoint& keypointsBall);
	void kGolfBallDetection(analysisFactor& analy);
	void kGolfBallDetectionIdx(int curIdx);


	int estimationImpact(Mat initGrayImg);
	int estimationTop(int impactIdx);
	int estimationFollow(int impactIdx);
	int estimationImpactSide();

	void sortIdx();
	void initResult();

	bool kGolfPreProcessing(Mat firstImg);
	bool kGolfPreProcessing(int initImgIdx);
	void kGolfPoseDetection(Mat curRedImg, Mat curBlueImg, int curImgIndex);
	void kGolfPoseDetectionIdx(int curImgIndex);
	
	//int resultSortFlow(const void* x, const void* y);
	//int resultSortIdx(const void* x, const void* y);

	bool checkLeftRighHand(Rect persionROI);
	//void detectedBall(Mat inputImg);

public:
	//Mat input;
	//Mat inputGray;
	//vector<analysisFactor> m_results;
	
	int m_ballImgSizeUpScale;
	Net kNet;
    string m_paperMaskPath;
    bool m_v4Ball=false, m_v4Temporal=false, m_v4Occlusion=false;
    cv::Mat m_otherPersonMask;
    string m_failureReason;

	bool m_firstCheckROI;
	Rect m_detectedPersonRect;
	Rect m_ballRegion;
	Rect m_upRegion;
	Rect m_flowRegion;
	Mat m_inputBlueC;
	Mat m_inputBlueP;
	Point m_ballPosition;
	Rect m_finalBallROI;  //최종 ball fitted Rect
	Mat m_inputC;
	Mat m_inputP;

	Mat m_blob;
	bool m_onece;
	
	//추가
	Rect m_persionExROI;  //swing bgs 계산을 위한 ROI
	bool calBGS();
	bool clustering();
	Point m_personCenter;
	Rect m_ballRegionEx;

	int totalImgNumMember;
	int m_impactIdxNew;

	//int m_dnnBallDetectedCC;
	////Rect m_ballPositionMean;
	//Rect m_ballPositionMeanPre;
	//Rect m_ballPositionSum;
	//bool m_checkBallPoistion;
	bool clusteringFixed();
	
	int m_windowSizeHalf;

public:
	analysisFactor m_results[m_result_Max];
	int m_resultsNum;
	int m_impactIdx;

	int m_backIdx;
	int m_realFirstIdx;
	
};


extern int m_dnnBallDetectedCC;
//Rect m_ballPositionMean;
extern Rect m_ballPositionMeanPre;
extern Rect m_ballPositionSum;
extern bool m_checkBallPoistion;