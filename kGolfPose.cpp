#include "kGolfPose.h"
#include "paper_rules.h"
#include <limits>
#include <cmath>




//int resultSortIdx(const void* x, const void* y)
//{
//	analysisFactor *ptA = (analysisFactor *)x;
//	analysisFactor *ptB = (analysisFactor *)y;
//
//	if (ptA->idx < ptB->idx)
//		return -1;
//	else if (ptA->idx > ptB->idx)
//		return 1;
//	else
//		return 0;
//}
//int resultSortFlow(const void* x, const void* y)
//{
//	analysisFactor *ptA = (analysisFactor *)x;
//	analysisFactor *ptB = (analysisFactor *)y;
//
//	if (ptA->flowEnergySum < ptB->flowEnergySum) return -1;
//	else if (ptA->flowEnergySum > ptB->flowEnergySum) return 1;
//	else return 0;
//}

bool resultSortIdx(const analysisFactor &u, const analysisFactor &v)
{
	if (u.idx < v.idx) {
		return true;
	}
	else if (u.idx == v.idx) {
		return u.idx < v.idx;
	}
	else {
		return false;
	}
}

bool resultSortFlow(const analysisFactor &u, const analysisFactor &v)
{
	if (u.flowEnergySum < v.flowEnergySum) {
		return true;
	}
	else if (u.flowEnergySum == v.flowEnergySum) {
		return u.flowEnergySum < v.flowEnergySum;
	}
	else {
		return false;
	}
}




//min first
//bool comparatorVerticalPositionLeft(const analysisFactor& s1, const analysisFactor& s2) {
//	return s1.leftGradient < s2.leftGradient;
//}
//
////max first
//bool comparatorVerticalPositionRight(const analysisFactor& s1, const analysisFactor& s2) {
//	return s1.rightGradient > s2.rightGradient;
//}

//min first
bool comparatorVerticalPositionFlow(const analysisFactor& s1, const analysisFactor& s2) {
	return s1.flowEnergySum < s2.flowEnergySum;
}

kGolfPose::kGolfPose(string cfg, string weight)
{
	kNet = readNetFromDarknet(cfg, weight);
	kNet.setPreferableBackend(DNN_BACKEND_DEFAULT);
	kNet.setPreferableTarget(DNN_TARGET_CPU);

	m_ballImgSizeUpScale = 3;
	m_windowSizeHalf = 5;   //if 10=>20 *20 window area =  400 , if 5 ==> 10*10 = 100
	

	//m_results.reserve(500);

	//for (size_t i = 0; i < 500; i++)
	//{
	//	m_results[i].ballCandidates.reserve(10);
	//}
	m_onece = true;

	// CSV 파일 열기 및 헤더 쓰기
	m_csvFile.open("metrics.csv");
	m_csvFile << "frame,flowEnergy,bgsCount\n";
}

void kGolfPose::initResult()
{
    for (int i=0; i<m_result_Max; ++i) {
        m_results[i] = analysisFactor{};
        m_results[i].idx=i;
        m_results[i].localImpactBGSCount=-1;
        m_results[i].flowEnergySum=-1;
        m_results[i].ckSkipEndPoint=true;
    }
    m_resultsNum=0; m_ballPosition=Point(-1,-1);
    m_finalBallROI=Rect(); m_impactIdx=-1; m_impactIdxNew=-1;
}

kGolfPose::~kGolfPose()
{
	// CSV 닫기
	if (m_csvFile.is_open())
		m_csvFile.close();
	//for (std::size_t i = 0; i < m_result_Max; ++i)
	//	m_results[i].curBlueImg.release();
	deletekGolfPose();
}

void kGolfPose::deletekGolfPose()
{
	for (std::size_t i = 0; i < m_result_Max; ++i)
		m_results[i].curBlueImg.release();
	/*int ss = m_results.size();
	if (ss > 0)
	{
		for (int r = 0; r < ss; r++)
		{
			int bs = m_results[r].ballCandidates.size();
			if (bs > 0)
			{
				m_results[r].ballCandidates.~vector<candidateBall>();
			}
		}
		m_results.~vector<analysisFactor>();
	}*/
}

void kGolfPose::detectedPerson(Mat inputImg, float threshold, Rect& outRec, Point& personCenter)
{
	
	Mat inputBlob = blobFromImage(inputImg, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
	kNet.setInput(inputBlob, "data");                   //set the network input
	Mat detectionMat = kNet.forward("detection_out");   //compute output

	float confidenceThreshold = threshold; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;        // Width of network's input image
	int inpHeight = 416;
	//outRec = Rect(Point(0, 0), Point(0, 0));   //max
	outRec = Rect();
    vector<Rect> people;    //min
	personCenter = Point(0, 0);

	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

		//0:person
		size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

	/*	if ((objectClass == 32))
		{
			float x_center = detectionMat.at<float>(i, 0) * inputImg.cols;
			float y_center = detectionMat.at<float>(i, 1) * inputImg.rows;
			float width = detectionMat.at<float>(i, 2) * inputImg.cols;
			float height = detectionMat.at<float>(i, 3) * inputImg.rows;
			Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
			Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));

			Rect outRecTemp = Rect(p1, p2);

			Mat BallROIFinal = inputImg.clone();
			rectangle(BallROIFinal, outRecTemp, Scalar(0, 0, 255), 1, 8, 0);
			cv::imshow("yolo", BallROIFinal);
			cvWaitKey(1);
			
		}*/

		if ((objectClass == 0) && (confidence > confidenceThreshold))
		{
			//std::cout << "detectionMat = " << objectClass << "confedence = " << confidence << endl;
			
			float x_center = detectionMat.at<float>(i, 0) * inputImg.cols;
			float y_center = detectionMat.at<float>(i, 1) * inputImg.rows;
			float width = detectionMat.at<float>(i, 2) * inputImg.cols;
			float height = detectionMat.at<float>(i, 3) * inputImg.rows;
			Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
			Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));

			Rect outRecTemp = Rect(p1, p2);

			//max height
			/*if (outRecTemp.height > outRec.height)
			{
				outRec = outRecTemp;
				personCenter.x = outRec.x + (outRec.width*0.5);
				personCenter.y = outRec.y + (outRec.height*0.5);
			}*/
			people.push_back(outRecTemp);
            if (outRec.empty() || outRecTemp.height < outRec.height)  // 작은것을 선택(카메라에서 먼 사람을 선택)
			{
				outRec = outRecTemp;
				personCenter.x = outRec.x + (outRec.width*0.5);
				personCenter.y = outRec.y + (outRec.height*0.5);
			}
					
		}
	}
    for (const auto& box : people) {
        if (box != outRec && (box & outRec).area()>0) {
            outRec=Rect();m_failureReason="person_overlap";return;
        }
    }
}


void kGolfPose::initBGS(Rect ROI, Mat initGrayImg, float shadowThres, bool setShadows, float backRatio, int blurParam, Ptr<BackgroundSubtractor>& kBGS, Mat& RoiRegion)
{
	kBGS = createBackgroundSubtractorMOG2(); //MOG2 approach
	if (kBGS.dynamicCast<cv::BackgroundSubtractorMOG2>() != NULL)
	{
		kBGS.dynamicCast<cv::BackgroundSubtractorMOG2>()->setShadowThreshold(shadowThres);
		kBGS.dynamicCast<cv::BackgroundSubtractorMOG2>()->setDetectShadows(setShadows);
		kBGS.dynamicCast<cv::BackgroundSubtractorMOG2>()->setBackgroundRatio(backRatio);
	}

	RoiRegion = Mat(initGrayImg, ROI);
	Mat tempOut;
	cv::blur(RoiRegion, RoiRegion, Size(blurParam, blurParam));
	kBGS->apply(RoiRegion, tempOut);
	kBGS->apply(RoiRegion, tempOut);

}

void kGolfPose::kBGS(Rect ROI, Ptr<BackgroundSubtractor> kBGS, Mat inputROIGrayImg, 
	int blurParam, Mat& outBGSImg)
{
	//Mat tempOut = Mat(inputGrayImg, ROI);
	
	cv::blur(inputROIGrayImg, inputROIGrayImg, Size(blurParam, blurParam));

	kBGS->apply(inputROIGrayImg, outBGSImg);
}

void kGolfPose::setSideViewBallRegion(Rect personROI, Size inputImgSize, int marginWidth, 
	int marginHeight, int setWidth, int setHeight, Point center, Rect& downRegion)
{
	downRegion = Rect(personROI.x + personROI.width + marginWidth, center.y, setWidth, inputImgSize.height - center.y - marginHeight);
}


void kGolfPose::setFrontRightRegion(Rect personROI, Size inputImgSize, int marginWidth, 
	int marginHeight, int setWidth, int setHeight, Point center, Rect& outRegion)
{
	outRegion = Rect(personROI.x + personROI.width + marginWidth, center.y, setWidth, setHeight);
}


void kGolfPose::kBGS(Mat initGrayImg, Mat inputGray, Rect ROI, bool checkShadow, float BGSRatio, Mat filterEleErode, Mat filterEleDilate, Mat& outBGSImg)
{
	Ptr<BackgroundSubtractor> BGS;
	BGS = createBackgroundSubtractorMOG2(); //MOG2 approach
	if (BGS.dynamicCast<cv::BackgroundSubtractorMOG2>() != NULL)
	{
		BGS.dynamicCast<cv::BackgroundSubtractorMOG2>()->setShadowThreshold(0.1);   //0.4
		BGS.dynamicCast<cv::BackgroundSubtractorMOG2>()->setDetectShadows(checkShadow);
		BGS.dynamicCast<cv::BackgroundSubtractorMOG2>()->setBackgroundRatio(BGSRatio);
	}

	//Mat outBGSTemp;
	Mat tempOut = initGrayImg(ROI).clone();
	cv::blur(tempOut, tempOut, Size(3, 3));
	BGS->apply(tempOut, outBGSImg);
	BGS->apply(tempOut, outBGSImg);
	Mat tempLeftImg = inputGray(ROI).clone();
	cv::blur(tempLeftImg, tempLeftImg, Size(3, 3));
	BGS->apply(tempLeftImg, outBGSImg);

	if (!filterEleErode.empty())
		cv::morphologyEx(outBGSImg, outBGSImg, cv::MORPH_ERODE, filterEleErode);
	if (!filterEleDilate.empty())
		cv::morphologyEx(outBGSImg, outBGSImg, cv::MORPH_DILATE, filterEleDilate);
	
}

void kGolfPose::searchRegion(Mat& img, int xx, int yy, float& sum, int& cc)
{
	if (yy > 0 && yy < img.rows && xx > 0 && xx < img.cols)
	{
		uchar tempValue = img.at<uchar>(yy, xx);

		if (tempValue == 0)
			return;
		else
		{
			cc++;
			//sum += tempValue;
			if (tempValue > sum)
			{
				sum = tempValue;
			}
			img.at<uchar>(yy, xx) = 0;
			//searchRegion(img, xx - 1, yy - 1, sum, cc);
			searchRegion(img, xx, yy - 1, sum, cc);
			//searchRegion(img, xx + 1, yy - 1, sum, cc);
			searchRegion(img, xx, yy + 1, sum, cc);
			searchRegion(img, xx + 1, yy, sum, cc);
			//searchRegion(img, xx + 1, yy - 1, sum, cc);
			searchRegion(img, xx - 1, yy, sum, cc);
			//searchRegion(img, xx + 1, yy + 1, sum, cc);
		}

	}
	
}

float kGolfPose::ballCandidates(Mat ori, Mat ballImg, cv::KeyPoint keypointsBall, float thresIn, float thresOut)
{
	//float check=0;
	
	//in
	float sum = 0;
	int cc = 0;
	Mat tmepIn = ballImg.clone();
	//tmepIn.at<uchar>(keypointsBall.pt.y , keypointsBall.pt.x ) = 0;
	cc++;
	searchRegion(tmepIn, keypointsBall.pt.x , keypointsBall.pt.y , sum ,cc);

	float inAvg = sum;
	float errorAvg = 0;
	for (int y = 0; y < tmepIn.rows; y++)
	{
		for (int x = 0; x < tmepIn.cols; x++)
		{
			if (ballImg.at<uchar>(y, x) == 0)
			{
				errorAvg += abs(inAvg - ori.at<uchar>(y, x));
			}
		}
	}

	errorAvg = errorAvg / cc;
	cout << errorAvg << endl;
	//if (errorAvg < thresIn)
	//	check = true;
	
	return errorAvg;

}

float euclidean_distance(cv::Point center, cv::Point point, int radius, float homogeneous) {
	float distance = std::sqrt(std::pow(center.x - point.x, 2) + std::pow(center.y - point.y, 2));
	if (distance <= 1)
	{
		return homogeneous;
	}
	else
	{
		distance = homogeneous / distance;
		return distance;
	}
	
}

float euclidean_distanceW(cv::Point center, cv::Point point, int radius, float homogeneous) {
	float distance = std::sqrt(std::pow(center.x - point.x, 2) + std::pow(center.y - point.y, 2));
	if (distance <= 1)
	{
		return homogeneous*2;
	}
	else
	{
		distance = (homogeneous / distance)*2;
		return distance;
	}

}

bool kGolfPose::findBallPosition()
{
    Mat vote(m_ballRegion.size(),CV_64FC1,Scalar(0));
    int count=0;
    for(int i=1;i<m_resultsNum/2;++i) for(const auto& b:m_results[i].paperCandidates) {
        if(b.size<=0) continue;
        // 개선버전 2
        for(int y=0;y<vote.rows;++y) for(int x=0;x<vote.cols;++x) {
            double dx=x+m_ballRegion.x-b.x,dy=y+m_ballRegion.y-b.y;
            vote.at<double>(y,x)+=hit::unitPeakVote(dx,dy,b.size);
        }
        ++count;
    }
    if(!count) {m_failureReason="no_ball_candidates";return false;}
    Point peak; minMaxLoc(vote,nullptr,nullptr,nullptr,&peak);
    m_ballPosition=peak+m_ballRegion.tl();
    m_finalBallROI=Rect(m_ballPosition.x-5,m_ballPosition.y-5,15,15);
    return true;
}


bool kGolfPose::checkLeftRighHand(Rect persionROI)
{
	Mat out;
	//kBGS(m_results[i].preGrayImg, m_results[i].curGrayImg, m_finalBallROI, 0, 0.1, eleT, eleEmpty, out);
	int ccount = 0;
	for (int yy = 0; yy < out.rows; yy++)
	{
		for (int xx = 0; xx < out.cols; xx++)
		{
			if (out.at<uchar>(yy, xx) > 200)
				ccount++;
		}
	}

	return 0;
}



bool kGolfPose::setROIs(Mat inputColor, Rect& person, Rect& ball, Rect& motion)
{
    detectedPerson(inputColor,0.5f,person,m_personCenter);
    Rect bounds(0,0,inputColor.cols,inputColor.rows);
    if(person.empty() || (person & bounds)!=person) {
        if(m_failureReason.empty()) m_failureReason="invalid_person";
        return false;
    }
    // Cache contains separate full-resolution person masks and their boxes.
    FileStorage masks(m_paperMaskPath,FileStorage::READ);
    if(!masks.isOpened()) {m_failureReason="missing_segmentation_cache";return false;}
    FileNode instances=masks["instances"];
    double best=0; Mat selected;
    for(auto it=instances.begin();it!=instances.end();++it) {
        int x,y,w,h; string file;
        (*it)["x"]>>x; (*it)["y"]>>y; (*it)["w"]>>w; (*it)["h"]>>h;
        (*it)["mask"]>>file;
        Rect box(x,y,w,h); double intersection=(person & box).area();
        double overlap=intersection/(person.area()+box.area()-intersection);
        if(overlap>best) {best=overlap;selected=imread(file,0);}
    }
    if(selected.empty() || selected.size()!=inputColor.size()) {
        m_failureReason="missing_matched_person_mask";return false;
    }
    double sum=0,count=0;
    for(int y=person.height/2;y<person.height;++y)
        for(int x=0;x<person.width;++x) if(selected.at<uchar>(person.y+y,person.x+x)) {
            sum+=x+0.5; ++count;
        }
    if(count==0) {m_failureReason="missing_lower_body_mask";return false;}
    int xg=(sum/count<=person.width/2.0)?person.x+person.width+40:person.x-240;
    ball=Rect(xg,person.y+person.height-70,200,80);
    motion=Rect(person.x,int(person.y+0.3*person.height-20),person.width,int(0.3*person.height));
    m_ballRegionEx=Rect(ball.x,ball.y-ball.height,ball.width,2*ball.height);
    m_persionExROI=person | ball;
    if((ball & bounds)!=ball || (motion & bounds)!=motion || (m_ballRegionEx & bounds)!=m_ballRegionEx) {
        m_failureReason="invalid_roi";return false;
    }
    return true;
}

// inputROI & out = 3 ch image
void kGolfPose::imgEnhance(Mat inputROI, Mat& out)
{
	
	int m_imgenhanceIteration = 250;
	//int sizeUpScale = 3;
	//cv::cvtColor(inputROI, inputROI, CV_GRAY2BGR);
	//int N = 1;
	//int n = m_imgenhanceIteration;  //250
	int inputKernelSize = 25;
	int intensityChangeKernelSize = 25;
	int r = 1;
	int c = 1;
	double upperBound = 255.0;
	bool normalizeIntensityChange = false;
	cv::Mat result;
	PerformLightRandomSpraysRetinex(inputROI, out, 1, m_imgenhanceIteration,inputKernelSize, 0, intensityChangeKernelSize, 0, r, c, normalizeIntensityChange, upperBound);
	
	//imshow("sprayRe", BallROI);
	//cv::cvtColor(out, out, CV_BGR2GRAY);
	
	
	//cv::resize(out, out, Size(out.cols * sizeUpScale, out.rows * sizeUpScale), 1, 1, CV_INTER_LANCZOS4);
	
	//cv::imshow("imgEnhance-out", out);
	//cv::imshow("BallROI", BallROI);
}

void sharpen(const cv::Mat &image, cv::Mat &result)
{
	int nchannels = image.channels(); 
	for (int j = 1; j < image.rows - 1; j++)
	{
		const uchar* previous = image.ptr<const uchar>(j - 1); 
		const uchar* current = image.ptr<const uchar>(j); 
		const uchar* next = image.ptr<const uchar>(j + 1); 
		uchar* output = result.ptr<uchar>(j); 
		for (int i = nchannels; i < (image.cols - 1)*nchannels; i++) 
			*output++ = cv::saturate_cast<uchar>(5 * current[i] - current[i - nchannels] - current[i + nchannels] - previous[i] - next[i]); 	
	}
	
	result.row(0).setTo(cv::Scalar(0, 0, 0));
	result.row(result.rows - 1).setTo(cv::Scalar(0, 0, 0));
	result.col(0).setTo(cv::Scalar(0, 0, 0));
	result.col(result.cols - 1).setTo(cv::Scalar(0, 0, 0));
}

void sharpen2D(const cv::Mat &image, cv::Mat &result) {

	// Construct kernel (all entries initialized to 0)
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	// assigns kernel values
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;

	//filter the image
	cv::filter2D(image, result, image.depth(), kernel);


}


float kGolfPose::ballCandidatesHomogeneous(Mat ori, cv::KeyPoint& keypointsBall)
{
	//float check=0;

	////in
	float sum = 0;
	int cc = 0;
	Mat tmepIn = ori.clone();
	////tmepIn.at<uchar>(keypointsBall.pt.y , keypointsBall.pt.x ) = 0;
	//cc++;
	searchRegion(tmepIn, keypointsBall.pt.x, keypointsBall.pt.y, sum, cc);

	float inAvg = sum / cc;
	float errorAvg = 0;
	for (int y = 0; y < tmepIn.rows; y++)
	{
		for (int x = 0; x < tmepIn.cols; x++)
		{
			if (ori.at<uchar>(y, x) == 0)
			{
				errorAvg += abs(inAvg - ori.at<uchar>(y, x));
			}
		}
	}

	errorAvg = errorAvg / cc;
	//cout << errorAvg << endl;
	////if (errorAvg < thresIn)
	////	check = true;

	return errorAvg;

}


template<typename T, typename T_, typename T__>
inline
T clamp(const T_ min, const T__ max, const T x)
{
	return
		(x < static_cast<T>(min)) ? static_cast<T>(min) :
		(x < static_cast<T>(max)) ? static_cast<T>(x) :
		static_cast<T>(max);
}

template<typename T>
inline
T
trilinear_interpolation(const cv::Mat mat,
	const double y,
	const double x,
	const double z)
{
	const size_t height = mat.size[0];
	const size_t width = mat.size[1];
	const size_t depth = mat.size[2];

	const size_t y_index = clamp(0, height - 1, static_cast<size_t>(y));
	const size_t yy_index = clamp(0, height - 1, y_index + 1);
	const size_t x_index = clamp(0, width - 1, static_cast<size_t>(x));
	const size_t xx_index = clamp(0, width - 1, x_index + 1);
	const size_t z_index = clamp(0, depth - 1, static_cast<size_t>(z));
	const size_t zz_index = clamp(0, depth - 1, z_index + 1);
	const double y_alpha = y - y_index;
	const double x_alpha = x - x_index;
	const double z_alpha = z - z_index;

	return
		(1.0 - y_alpha) * (1.0 - x_alpha) * (1.0 - z_alpha) * mat.at<T>(y_index, x_index, z_index) +
		(1.0 - y_alpha) * x_alpha       * (1.0 - z_alpha) * mat.at<T>(y_index, xx_index, z_index) +
		y_alpha * (1.0 - x_alpha) * (1.0 - z_alpha) * mat.at<T>(yy_index, x_index, z_index) +
		y_alpha * x_alpha       * (1.0 - z_alpha) * mat.at<T>(yy_index, xx_index, z_index) +
		(1.0 - y_alpha) * (1.0 - x_alpha) * z_alpha       * mat.at<T>(y_index, x_index, zz_index) +
		(1.0 - y_alpha) * x_alpha       * z_alpha       * mat.at<T>(y_index, xx_index, zz_index) +
		y_alpha * (1.0 - x_alpha) * z_alpha       * mat.at<T>(yy_index, x_index, zz_index) +
		y_alpha * x_alpha       * z_alpha       * mat.at<T>(yy_index, xx_index, zz_index);

}


void bilateralFilterImpljj(cv::Mat1d src, cv::Mat1d dst,
	double sigma_color, double sigma_space)
{
	using namespace cv;
	const size_t height = src.rows, width = src.cols;
	const size_t padding_xy = 2, padding_z = 2;
	double src_min, src_max;
	cv::minMaxLoc(src, &src_min, &src_max);

	const size_t small_height = static_cast<size_t>((height - 1) / sigma_space) + 1 + 2 * padding_xy;
	const size_t small_width = static_cast<size_t>((width - 1) / sigma_space) + 1 + 2 * padding_xy;
	const size_t small_depth = static_cast<size_t>((src_max - src_min) / sigma_color) + 1 + 2 * padding_xy;

	int data_size[] = { small_height, small_width, small_depth };
	cv::Mat data(3, data_size, CV_64FC2);
	data.setTo(0);

	// down sample
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const size_t small_x = static_cast<size_t>(x / sigma_space + 0.5) + padding_xy;
			const size_t small_y = static_cast<size_t>(y / sigma_space + 0.5) + padding_xy;
			const double z = src.at<double>(y, x) - src_min;
			const size_t small_z = static_cast<size_t>(z / sigma_color + 0.5) + padding_z;

			cv::Vec2d v = data.at<cv::Vec2d>(small_y, small_x, small_z);
			v[0] += src.at<double>(y, x);
			v[1] += 1.0;
			data.at<cv::Vec2d>(small_y, small_x, small_z) = v;
		}
	}

	// convolution
	cv::Mat buffer(3, data_size, CV_64FC2);
	buffer.setTo(0);
	int offset[3];
	offset[0] = &(data.at<cv::Vec2d>(1, 0, 0)) - &(data.at<cv::Vec2d>(0, 0, 0));
	offset[1] = &(data.at<cv::Vec2d>(0, 1, 0)) - &(data.at<cv::Vec2d>(0, 0, 0));
	offset[2] = &(data.at<cv::Vec2d>(0, 0, 1)) - &(data.at<cv::Vec2d>(0, 0, 0));

	for (int dim = 0; dim < 3; ++dim) { // dim = 3 stands for x, y, and depth
		const int off = offset[dim];
		for (int ittr = 0; ittr < 2; ++ittr) {
			cv::swap(data, buffer);

			for (int y = 1; y < small_height - 1; ++y) {
				for (int x = 1; x < small_width - 1; ++x) {
					cv::Vec2d *d_ptr = &(data.at<cv::Vec2d>(y, x, 1));
					cv::Vec2d *b_ptr = &(buffer.at<cv::Vec2d>(y, x, 1));
					for (int z = 1; z < small_depth - 1; ++z, ++d_ptr, ++b_ptr) {
						cv::Vec2d b_prev = *(b_ptr - off), b_curr = *b_ptr, b_next = *(b_ptr + off);
						*d_ptr = (b_prev + b_next + 2.0 * b_curr) / 4.0;
					} // z
				} // x
			} // y

		} // ittr
	} // dim

	// upsample

	for (cv::MatIterator_<cv::Vec2d> d = data.begin<cv::Vec2d>(); d != data.end<cv::Vec2d>(); ++d)
	{
		(*d)[0] /= (*d)[1] != 0 ? (*d)[1] : 1;
	}

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const double z = src.at<double>(y, x) - src_min;
			const double px = static_cast<double>(x) / sigma_space + padding_xy;
			const double py = static_cast<double>(y) / sigma_space + padding_xy;
			const double pz = static_cast<double>(z) / sigma_color + padding_z;
			dst.at<double>(y, x) = trilinear_interpolation<cv::Vec2d>(data, py, px, pz)[0];
		}
	}

}


void bilateralFilterjj(cv::InputArray _src, cv::OutputArray _dst,
	double sigmaColor, double sigmaSpace)
{
	cv::Mat src = _src.getMat();

	CV_Assert(src.channels() == 1);

	// bilateralFilterImpl runs with double depth, single channel
	if (src.depth() != CV_64FC1) {
		src = cv::Mat(_src.size(), CV_64FC1);
		_src.getMat().convertTo(src, CV_64FC1);
	}

	cv::Mat dst_tmp = cv::Mat(_src.size(), CV_64FC1);
	bilateralFilterImpljj(src, dst_tmp, sigmaColor, sigmaSpace);

	_dst.create(dst_tmp.size(), _src.type());
	dst_tmp.convertTo(_dst.getMat(), _src.type());

}

Mat src_gray;
Mat detected_edges;
int lowThreshold = 10;
int highThreshold = 80;

static void CannyThreshold(int, void*)
{
	//blur(src_gray, detected_edges, Size(3, 3));
	Canny(src_gray, detected_edges, lowThreshold, highThreshold, 3);

	imshow("Canny Edge", detected_edges);
}



void kGolfPose::blobDetection(Mat inputImg, analysisFactor& ballCandidates)
{

	//int m_ballImgSizeUpScale = 2;  //m_ballImgSizeUpScale
	int m_meanShiftSp = 5;
	int m_meanShiftSr = 20;
	float m_blobMinDistBetweenBlobs = 40.0f;
	float m_blobMinConvexity = 0.7f;   //jj 0925
	float m_blobMinCircularity = 0.4f;
	float m_blobMaxCircularity = 1.0f;

	// scale= 3 fixed
	int m_blobMinArea = 400;
	int m_blobMaxArea = 2000;
	// scale= 4
	//int m_blobMinArea = 600;
	//int m_blobMaxArea = 2000;


	//cv::imshow("meanfilter", re);
	//cvWaitKey(1);

	//cv::blur(re, re, Size(3, 3));
	
	//cv::resize(re, re, Size(re.cols * m_ballImgSizeUpScale, re.rows * m_ballImgSizeUpScale), 1, 1, CV_INTER_LANCZOS4);
	//cv::blur(re, re, Size(3, 3));
	//cv::cvtColor(inputImg, inputImg, CV_BGR2GRAY);
	cv::resize(inputImg, inputImg, Size(inputImg.cols * m_ballImgSizeUpScale, inputImg.rows * m_ballImgSizeUpScale), 1, 1, CV_INTER_LANCZOS4);
	
	Mat bilateralFilterOutRed , bilateralFilterOutA, bilateralFilterOutB;
	//Mat bilateralFilterOutPre;
	//bilateralFilter(inputImg, bilateralFilterOutPre, 10, 100, 50);
	//bilateralFilter(bilateralFilterOutPre, bilateralFilterOutRed, 10, 100, 50);
	//Mat bilateralFilterOutaaa;
	//bilateralFilter(inputImg, bilateralFilterOutaaa, 10, 100, 50);
	//Mat reee;
	
	//clock_t start, end;
	//start = clock();
	//Mat asdf;

	//input : 1 ch
	//bilateralFilterjj(inputImg, bilateralFilterOutRed, 10, 10);
	bilateralFilterjj(inputImg, bilateralFilterOutRed, 20, 10);

	//bilateralFilterjj(inputImg, bilateralFilterOutRed, 20, 10);
	//bilateralFilterjj(inputImg, bilateralFilterOutA, 50, 10);
	//bilateralFilterjj(inputImg, bilateralFilterOutB, 20, 500);
	//pyrMeanShiftFiltering(bilateralFilterOutRed, bilateralFilterOutRed, m_meanShiftSp, m_meanShiftSr);
	//cv::blur(inputImg, inputImg, Size(3, 3));
	//Mat threshold_output;
	//end = clock(); //시간 측정 끝
	//double resultTime = (double)(end - start);
	//printf("bilateralFilterjj time : %f\n", resultTime);
	//imwrite(".\\BF.jpg", bilateralFilterOutRed);
	//imwrite(".\\BF_before.jpg", inputImg);
	//Mat motion;
	//cv::absdiff(inputImg, bilateralFilterOutRed, motion);


	//src_gray = bilateralFilterOutRed.clone();
	//namedWindow("Canny Edge", WINDOW_AUTOSIZE);
	//createTrackbar("Min Threshold:", "Canny Edge", &lowThreshold, 1000, CannyThreshold);
	//createTrackbar("Max Threshold:", "Canny Edge", &highThreshold, 1000, CannyThreshold);
	//CannyThreshold(0, 0);
	//waitKey(0);
	


	//vector<KeyPoint> keypointsD;
	//Ptr<FastFeatureDetector> detectorFast = FastFeatureDetector::create();
	//vector<Mat> descriptor;

	//Mat fastOut = bilateralFilterOutRed.clone();
	//detectorFast->detect(bilateralFilterOutRed, keypointsD, Mat());
	//drawKeypoints(fastOut, keypointsD, fastOut);

	//Mat dst;
	//int kernel_size = 3;
	//int scale = 1;
	//int delta = 0;
	//int ddepth = CV_16S;
	//Mat abs_dst;

	//Laplacian(inputImg, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(dst, abs_dst);


	EDCircles testEDCircles = EDCircles(bilateralFilterOutRed);
	std::vector<mCircle> circlesEDout = testEDCircles.getCircles();

	//Mat circleImg = testEDCircles.drawResult(false, ImageStyle::CIRCLES);
	//cv::imshow("ED", circleImg);
	//mCircle outEDCircles;



#if 0

	std::vector<std::vector<cv::Point>> contours;
	Mat cannyOut;
	cv::Canny(bilateralFilterOutRed, cannyOut, 15, 80);

	

	cv::findContours(cannyOut,
		contours,    // 외곽선 벡터 
		CV_RETR_EXTERNAL,  // 외부 외곽선 검색
		CV_CHAIN_APPROX_NONE); // 각 외곽선의 모든 화소 탐색
		// 지정된 플래그 - 첫 번째는 외부 외곽선이 필요함을 나타내며 객체의 구멍을 무시
		// 두 번째 플래그는 외곽선의 형태를 지정 - 현재 옵션으로 벡터는 외곽선 내의 모든 화소 목록
		// CV_CHAIN_APPROX_SIMPLE 플래그로 하면 마지막 점이 수평 또는 수직, 대각선 외곽선에 포함됨
		// 다른 플래그는 간결한 표현을 얻기 위해 외곽선의 정교하게 연결된 근사치를 제공
	   // 이전 영상으로 9개 외곽선을 contours.size()로 얻음
	   // drawContours() 함수는 영상 내의 각 외곽선을 그릴 수 있는 함수
	   // 하얀 영상 내 검은 외곽선 그리기
	cv::Mat result(bilateralFilterOutRed.size(), CV_8U, cv::Scalar(255));
	cv::drawContours(result, contours,
		-1,    // 모든 외곽선 그리기
		cv::Scalar(0), // 검게
		2);    // 두께를 2로
		// 세 번째 파라미터가 음수라면 모든 외곽선이 그려짐
		// 반면 그려져야 하는 외곽선의 첨자를 지정할 수 있음


	vector<Vec3f> circles;
	// 2,   // 누적기 해상도(영상크기/2)
	//50,  // 두 원 간의 최소 거리
	//	200, // 캐니 최대 경계값
	//	100, // 투표 최소 개수
	//	25, 100); // 최소와 최대 반지름
	cv::HoughCircles(bilateralFilterOutRed, circles, HOUGH_GRADIENT, 1, 
		50,  // 두 원 간의 최소 거리
		20,   // 캐니 최대 경계값
		20,   // 투표 최소 개수 10
		10, 20);  // 최소와 최대 반지름
	Mat houghOut = bilateralFilterOutRed.clone();
	cv::cvtColor(houghOut, houghOut, CV_GRAY2BGR);


	Mat sobel;

	Mat sobelX;

	Mat sobelY;

	cv::Sobel(bilateralFilterOutRed, sobelX, CV_8U, 1, 0);

	cv::Sobel(bilateralFilterOutRed, sobelY, CV_8U, 0, 1);

	sobel = abs(sobelX) + abs(sobelY);


	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center(c[0], c[1]);
		int radius = c[2];

		circle(houghOut, center, radius, Scalar(0, 255, 0), 2);
		circle(houghOut, center, 2, Scalar(0, 0, 255), 3);

		//ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].x = c[0];
		//ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].y = c[1];
		//ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].size = c[2];
		//ballCandidates.ballCandidatesNum++;

	}

	cv::imshow("houghOut", houghOut);
	//cv::imwrite(".\\BallROI.jpg", img_color);
	
#endif
	//start = clock();
	Mat thresholdRe;
	threshold(bilateralFilterOutRed, thresholdRe, 140, 255, THRESH_BINARY | THRESH_OTSU);
	//end = clock(); //시간 측정 끝
	//resultTime = (double)(end - start);
	//printf("threshold time : %f\n", resultTime);
	//cv::imshow("thresholdRe", thresholdRe);

//	start = clock();
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = m_blobMinDistBetweenBlobs;
	params.filterByConvexity = true;
	params.minConvexity = m_blobMinConvexity;  //1 : very convexity,  0.9;
	params.filterByColor = false;
	params.blobColor = 255;
	params.filterByCircularity = false;
	params.minCircularity = m_blobMinCircularity;
	params.maxCircularity = m_blobMaxCircularity;
	params.filterByArea = true;
	params.minArea = m_blobMinArea;
	params.maxArea = m_blobMaxArea;

	SimpleBlobDetector::Params paramsSecond;
	paramsSecond.minDistBetweenBlobs = m_blobMinDistBetweenBlobs;
	paramsSecond.filterByConvexity = true;
	paramsSecond.minConvexity = m_blobMinConvexity;  //1 : very convexity,  0.9;
	paramsSecond.filterByColor = false;
	paramsSecond.blobColor = 255;
	paramsSecond.filterByCircularity = false;
	paramsSecond.minCircularity = m_blobMinCircularity;
	paramsSecond.maxCircularity = m_blobMaxCircularity;
	paramsSecond.filterByArea = true;
	paramsSecond.minArea = m_blobMinArea;
	paramsSecond.maxArea = m_blobMaxArea;

	std::vector<KeyPoint> firstTmp;
	std::vector<KeyPoint> secondTmp;
	
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	Ptr<SimpleBlobDetector> detectorSecond = SimpleBlobDetector::create(paramsSecond);


	detector->detect(bilateralFilterOutRed, firstTmp);
	detectorSecond->detect(thresholdRe, secondTmp);
	//detectorSecond->detect(cannyOut, secondTmp);

	//end = clock(); //시간 측정 끝
	//resultTime = (double)(end - start);
	//printf("SimpleBlobDetector time : %f\n", resultTime);

	
	for (size_t f = 0; f < firstTmp.size(); f++)
	{
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].x = firstTmp[f].pt.x;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].y = firstTmp[f].pt.y;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].size = firstTmp[f].size;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].idx = 0;
		ballCandidates.ballCandidatesNum++;
	}

	//jj 0924
	for (size_t s = 0; s < secondTmp.size(); s++)
	{
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].x = secondTmp[s].pt.x;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].y = secondTmp[s].pt.y;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].size = secondTmp[s].size;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].idx = 1;
		ballCandidates.ballCandidatesNum++;
	}

	for (size_t s = 0; s < circlesEDout.size(); s++)
	{
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].x = circlesEDout[s].center.x;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].y = circlesEDout[s].center.y;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].size = circlesEDout[s].r;
		ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].idx = 1;
		ballCandidates.ballCandidatesNum++;
	}

#if 1
	Mat im_with_keypoints = bilateralFilterOutRed.clone();
	cv::cvtColor(im_with_keypoints, im_with_keypoints, CV_GRAY2BGR);
	for (size_t t = 0; t < firstTmp.size(); t++)
		circle(im_with_keypoints, CvPoint(firstTmp[t].pt.x, firstTmp[t].pt.y), firstTmp[t].size*0.5, Scalar(0, 0, 255), 1, -1);
	for (size_t at = 0; at < secondTmp.size(); at++)
		circle(im_with_keypoints, CvPoint(secondTmp[at].pt.x, secondTmp[at].pt.y), secondTmp[at].size*0.5, Scalar(200, 255, 0), 1, -1);
	for (size_t at = 0; at < circlesEDout.size(); at++)
		circle(im_with_keypoints, CvPoint(circlesEDout[at].center.x, circlesEDout[at].center.y), circlesEDout[at].r, Scalar(255, 0, 0), 1, -1);
	//cv::imwrite(".\\BallROI.jpg", im_with_keypoints);
	cv::imshow("im_with_keypoints", im_with_keypoints);
	cvWaitKey(1);
#endif
	//circlesEDout.~vector< mCircle>();
	//firstTmp.~vector< KeyPoint>();
	//secondTmp.~vector< KeyPoint>();
}


float attract(float x, float y, float z, float t, float  expose) {
	//expose = 1.;
	//if(expose!=1.){expose=3.;}
	expose = 1. / 2.;
	return pow((-2 * pow(x, expose) + 10 * pow(y, expose) + 10 * pow(z, expose) + -2 * pow(t, expose)) / 16., 1. / expose);
}

Mat interpolate_rgb(Mat X) {

	int ws = 4;//window_size;
	double theta = 1.15;
	int gama = 5;
	int right_lim = (ws - 1) / 2 + 1;
	int left_lim = -(ws - 1) / 2;
	Mat X_mirrors;
	Mat R_mirrors;
	cout << "test1" << endl;
	copyMakeBorder(X, X_mirrors, ws - 1, ws - 1, ws - 1, ws - 1, BORDER_REFLECT);
	Mat Result(2 * X.rows, 2 * X.cols, CV_8UC3, Scalar(0, 0, 0));

	for (int i = ws - 1; i < X_mirrors.rows - ws + 1; i++) {
		for (int j = ws - 1; j < X_mirrors.cols - ws + 1; j++) {

			int ii = 2 * (i - ws + 1);
			int jj = 2 * (j - ws + 1);
			Vec3b pixel = X_mirrors.at<Vec3b>(i, j);

			// midles """""""""""""""""""""""""""""""""
			Result.at<Vec3b>(ii, jj) = pixel;

		}
	}

	for (int i = ws - 1; i < X_mirrors.rows - ws + 1; i++) {
		for (int j = ws - 1; j < X_mirrors.cols - ws + 1; j++) {

			int ii = 2 * (i - ws + 1) + 1;
			int jj = 2 * (j - ws + 1) + 1;

			Vec3b value;

			for (int col = 0; col < 3; col++) {
				// d1 and d2 strength
				double d1 = 0;
				double d2 = 0;
				for (int x = -1; x < 2; x++) {
					for (int y = -1; y < 2; y++) {
						//d1:
						Vec3b pix11 = X_mirrors.at<Vec3b>(i + x + 1, j + y);
						Vec3b pix12 = X_mirrors.at<Vec3b>(i + x, j + y + 1);
						d1 += abs(double(pix11[col]) - double(pix12[col]));
						//d2:
						Vec3b pix21 = X_mirrors.at<Vec3b>(i + x, j + y);
						Vec3b pix22 = X_mirrors.at<Vec3b>(i + x + 1, j + y + 1);
						d2 += abs(double(pix21[col]) - double(pix22[col]));
					}
				}
				Vec3b pix00 = X_mirrors.at<Vec3b>(i - 1, j - 1);
				Vec3b pix11 = X_mirrors.at<Vec3b>(i, j);
				Vec3b pix22 = X_mirrors.at<Vec3b>(i + 1, j + 1);
				Vec3b pix33 = X_mirrors.at<Vec3b>(i + 2, j + 2);
				Vec3b pix30 = X_mirrors.at<Vec3b>(i + 2, j - 1);
				Vec3b pix21 = X_mirrors.at<Vec3b>(i + 1, j);
				Vec3b pix12 = X_mirrors.at<Vec3b>(i, j + 1);
				Vec3b pix03 = X_mirrors.at<Vec3b>(i - 1, j + 2);

				if (100 * (1 + d1) > 100 * theta*(1 + d2)) {
					double val = attract(double(pix00[col]), double(pix11[col]), double(pix22[col]), double(pix33[col]), 1.);
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value[col] = val;
				}
				if (100 * (1 + d2) > 100 * theta*(1 + d1)) {
					double val = attract(double(pix30[col]), double(pix21[col]), double(pix12[col]), double(pix03[col]), 1.);
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value[col] = val;
				}
				if (100 * (1 + d1) <= 100 * theta*(1 + d2) && 100 * (1 + d2) <= 100 * theta*(1 + d1)) {
					double w1 = 1 / (1 + pow(d1, gama));
					double w2 = 1 / (1 + pow(d2, gama));
					double weight1 = w1 / (w1 + w2);
					double weight2 = w2 / (w1 + w2);
					double drval = attract(double(pix00[col]), double(pix11[col]), double(pix22[col]), double(pix33[col]), 1. / 6.);
					double urval = attract(double(pix30[col]), double(pix21[col]), double(pix12[col]), double(pix03[col]), 1. / 6.);
					double val = drval * weight1 + urval * weight2;
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value[col] = val;
				}
			}
			Result.at<Vec3b>(ii, jj) = value;
		}
	}


	copyMakeBorder(Result, R_mirrors, 4, 4, 4, 4, BORDER_REFLECT);

	for (int i = ws - 1; i < X_mirrors.rows - ws + 1; i++) {
		for (int j = ws - 1; j < X_mirrors.cols - ws + 1; j++) {

			//left
			int ii1 = 2 * (i - ws + 1) + 1;
			int jj1 = 2 * (j - ws + 1) + 0;
			//down
			int ii2 = 2 * (i - ws + 1) + 0;
			int jj2 = 2 * (j - ws + 1) + 1;

			Vec3b value1;
			Vec3b value2;

			//left
			for (int col = 0; col < 3; col++) {
				// d1 and d2 strength
				int x = ii1 + 4;
				int y = jj1 + 4;

				Vec3b pix1_2 = R_mirrors.at<Vec3b>(x + 1, y - 2);
				Vec3b pix_1_2 = R_mirrors.at<Vec3b>(x - 1, y - 2);
				Vec3b pix2_1 = R_mirrors.at<Vec3b>(x + 2, y - 1);
				Vec3b pix0_1 = R_mirrors.at<Vec3b>(x, y - 1);
				Vec3b pix_2_1 = R_mirrors.at<Vec3b>(x - 2, y - 1);
				Vec3b pix30 = R_mirrors.at<Vec3b>(x + 3, y);
				Vec3b pix10 = R_mirrors.at<Vec3b>(x + 1, y);
				Vec3b pix_10 = R_mirrors.at<Vec3b>(x - 1, y);
				Vec3b pix_30 = R_mirrors.at<Vec3b>(x - 3, y);
				Vec3b pix21 = R_mirrors.at<Vec3b>(x + 2, y + 1);
				Vec3b pix01 = R_mirrors.at<Vec3b>(x, y + 1);
				Vec3b pix_21 = R_mirrors.at<Vec3b>(x - 2, y + 1);
				Vec3b pix12 = R_mirrors.at<Vec3b>(x + 1, y + 2);
				Vec3b pix_12 = R_mirrors.at<Vec3b>(x - 1, y + 2);
				Vec3b pix03 = R_mirrors.at<Vec3b>(x, y + 3);
				Vec3b pix0_3 = R_mirrors.at<Vec3b>(x, y - 3);

				double d1 = abs(double(pix1_2[col]) - double(pix_1_2[col])) + \
					abs(double(pix2_1[col]) - double(pix0_1[col])) + \
					abs(double(pix0_1[col]) - double(pix_2_1[col])) + \
					abs(double(pix30[col]) - double(pix10[col])) + \
					abs(double(pix10[col]) - double(pix_10[col])) + \
					abs(double(pix_10[col]) - double(pix_30[col])) + \
					abs(double(pix21[col]) - double(pix01[col])) + \
					abs(double(pix01[col]) - double(pix_21[col])) + \
					abs(double(pix12[col]) - double(pix_12[col]));

				double d2 = abs(double(pix_21[col]) - double(pix_2_1[col])) + \
					abs(double(pix_12[col]) - double(pix_10[col])) + \
					abs(double(pix_10[col]) - double(pix_1_2[col])) + \
					abs(double(pix03[col]) - double(pix01[col])) + \
					abs(double(pix01[col]) - double(pix0_1[col])) + \
					abs(double(pix0_1[col]) - double(pix0_3[col])) + \
					abs(double(pix12[col]) - double(pix10[col])) + \
					abs(double(pix10[col]) - double(pix1_2[col])) + \
					abs(double(pix21[col]) - double(pix2_1[col]));

				if (100 * (1 + d1) > 100 * theta*(1 + d2)) {
					double val = attract(double(pix0_3[col]), double(pix0_1[col]), double(pix01[col]), double(pix03[col]), 1.);
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value1[col] = val;
				}
				if (100 * (1 + d2) > 100 * theta*(1 + d1)) {
					double val = attract(double(pix_30[col]), double(pix_10[col]), double(pix10[col]), double(pix30[col]), 1.);
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value1[col] = val;
				}
				if (100 * (1 + d1) <= 100 * theta*(1 + d2) && 100 * (1 + d2) <= 100 * theta*(1 + d1)) {
					double w1 = 1 / (1 + pow(d1, gama));
					double w2 = 1 / (1 + pow(d2, gama));
					double weight1 = w1 / (w1 + w2);
					double weight2 = w2 / (w1 + w2);
					double drval = attract(double(pix0_3[col]), double(pix0_1[col]), double(pix01[col]), double(pix03[col]), 1. / 6.);
					double urval = attract(double(pix_30[col]), double(pix_10[col]), double(pix10[col]), double(pix30[col]), 1. / 6.);
					double val = drval * weight1 + urval * weight2;
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value1[col] = val;
				}
			}
			//down
			for (int col = 0; col < 3; col++) {
				// d1 and d2 strength
				int x = ii2 + 4;
				int y = jj2 + 4;

				Vec3b pix1_2 = R_mirrors.at<Vec3b>(x + 1, y - 2);
				Vec3b pix_1_2 = R_mirrors.at<Vec3b>(x - 1, y - 2);
				Vec3b pix2_1 = R_mirrors.at<Vec3b>(x + 2, y - 1);
				Vec3b pix0_1 = R_mirrors.at<Vec3b>(x, y - 1);
				Vec3b pix_2_1 = R_mirrors.at<Vec3b>(x - 2, y - 1);
				Vec3b pix30 = R_mirrors.at<Vec3b>(x + 3, y);
				Vec3b pix10 = R_mirrors.at<Vec3b>(x + 1, y);
				Vec3b pix_10 = R_mirrors.at<Vec3b>(x - 1, y);
				Vec3b pix_30 = R_mirrors.at<Vec3b>(x - 3, y);
				Vec3b pix21 = R_mirrors.at<Vec3b>(x + 2, y + 1);
				Vec3b pix01 = R_mirrors.at<Vec3b>(x, y + 1);
				Vec3b pix_21 = R_mirrors.at<Vec3b>(x - 2, y + 1);
				Vec3b pix12 = R_mirrors.at<Vec3b>(x + 1, y + 2);
				Vec3b pix_12 = R_mirrors.at<Vec3b>(x - 1, y + 2);
				Vec3b pix03 = R_mirrors.at<Vec3b>(x, y + 3);
				Vec3b pix0_3 = R_mirrors.at<Vec3b>(x, y - 3);

				double d1 = abs(double(pix1_2[col]) - double(pix_1_2[col])) + \
					abs(double(pix2_1[col]) - double(pix0_1[col])) + \
					abs(double(pix0_1[col]) - double(pix_2_1[col])) + \
					abs(double(pix30[col]) - double(pix10[col])) + \
					abs(double(pix10[col]) - double(pix_10[col])) + \
					abs(double(pix_10[col]) - double(pix_30[col])) + \
					abs(double(pix21[col]) - double(pix01[col])) + \
					abs(double(pix01[col]) - double(pix_21[col])) + \
					abs(double(pix12[col]) - double(pix_12[col]));

				double d2 = abs(double(pix_21[col]) - double(pix_2_1[col])) + \
					abs(double(pix_12[col]) - double(pix_10[col])) + \
					abs(double(pix_10[col]) - double(pix_1_2[col])) + \
					abs(double(pix03[col]) - double(pix01[col])) + \
					abs(double(pix01[col]) - double(pix0_1[col])) + \
					abs(double(pix0_1[col]) - double(pix0_3[col])) + \
					abs(double(pix12[col]) - double(pix10[col])) + \
					abs(double(pix10[col]) - double(pix1_2[col])) + \
					abs(double(pix21[col]) - double(pix2_1[col]));

				if (100 * (1 + d1) > 100 * theta*(1 + d2)) {
					double val = attract(double(pix0_3[col]), double(pix0_1[col]), double(pix01[col]), double(pix03[col]), 1.);
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value2[col] = val;
				}
				if (100 * (1 + d2) > 100 * theta*(1 + d1)) {
					double val = attract(double(pix_30[col]), double(pix_10[col]), double(pix10[col]), double(pix30[col]), 1.);
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value2[col] = val;
				}
				if (100 * (1 + d1) <= 100 * theta*(1 + d2) && 100 * (1 + d2) <= 100 * theta*(1 + d1)) {
					double w1 = 1 / (1 + pow(d1, gama));
					double w2 = 1 / (1 + pow(d2, gama));
					double weight1 = w1 / (w1 + w2);
					double weight2 = w2 / (w1 + w2);
					double drval = attract(double(pix0_3[col]), double(pix0_1[col]), double(pix01[col]), double(pix03[col]), 1. / 6.);
					double urval = attract(double(pix_30[col]), double(pix_10[col]), double(pix10[col]), double(pix30[col]), 1. / 6.);
					double val = drval * weight1 + urval * weight2;
					if (val < 0) { val = 0; }
					if (val > 255) { val = 255; }
					value2[col] = val;
				}
			}
			Result.at<Vec3b>(ii1, jj1) = value1;
			Result.at<Vec3b>(ii2, jj2) = value2;
		}
	}
	cout << "test2" << endl;
	//imshow( "sum", Result );
	return Result;
}



bool kGolfPose::blobDetectionIdx(Mat inputImg, int curIdx)
{
#ifdef DebugJJ
	Mat inputTemp = m_results[curIdx].curImg.clone();
	cv::resize(inputTemp, inputTemp, Size(inputTemp.cols * 0.5, inputTemp.rows * 0.5), 1, 1, CV_INTER_LANCZOS4);
	cv::imshow("inputTemp", inputTemp);
	cvWaitKey(1);
#endif


#if 0
	//cout << " .1 - " << curIdx << endl;
	//cvtColor(inputImg, inputImg, COLOR_BGR2GRAY);     //COLOR_BGR2YCrCb
	cv::resize(inputImg, inputImg, Size(inputImg.cols * m_ballImgSizeUpScale, inputImg.rows * m_ballImgSizeUpScale), 1, 1, CV_INTER_LANCZOS4);
	

	//bilateralFilterjj(inputImg, inputImg, 20, 10);   //Y
	//cvtColor(inputImg, inputImg, COLOR_GRAY2BGR);
	Mat inputBlob = blobFromImage(inputImg, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
	kNet.setInput(inputBlob, "data");                   //set the network input
	Mat detectionMat = kNet.forward("detection_out");   //compute output
	//cout << " .2 - " << curIdx << endl;
	float confidenceThreshold = 0.4; // Confidence threshold
	int inpWidth = 416;        // Width of network's input image
	int inpHeight = 416;
	//cout << " .3 - " << curIdx << endl;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);


		size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

		if ((objectClass == 32) && (confidence > confidenceThreshold))
		{
			//cout << " .4 - " << curIdx << endl;
			Rect outRecBallDnn = Rect(0, 0, 0, 0);

			m_dnnBallDetectedCC++;
			float x_center = detectionMat.at<float>(i, 0) * inputImg.cols;
			float y_center = detectionMat.at<float>(i, 1) * inputImg.rows;
			float width = detectionMat.at<float>(i, 2) * inputImg.cols;
			float height = detectionMat.at<float>(i, 3) * inputImg.rows;
			Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
			Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));

			outRecBallDnn = Rect(p1, p2);

#ifdef DebugJJ
			cout << " confidence = " << confidence << endl;
			cout << "dnn ball OK !!!!!!!!! =====================================================    " << m_dnnBallDetectedCC << endl;
			Mat BallROIFinal = inputImg.clone();
			rectangle(BallROIFinal, outRecBallDnn, Scalar(0, 0, 255), 1, 8, 0);
			//cv::resize(BallROIFinal, BallROIFinal, Size(BallROIFinal.cols, BallROIFinal.rows), 1, 1, CV_INTER_LANCZOS4);
			//cv::imshow("inputTemp", inputTemp);
			cv::imshow("dnnBall", BallROIFinal);
			cvWaitKey(1);
#endif

			if (m_dnnBallDetectedCC == 1)
			{
				m_ballPositionMeanPre = outRecBallDnn;
				m_ballPositionSum.x = outRecBallDnn.x;
				m_ballPositionSum.y = outRecBallDnn.y;

				cout << "outRecBallDnn.x = " << outRecBallDnn.x << endl;
				cout << "m_ballPositionSum.x = " << m_ballPositionSum.x << endl;
				cout << "m_ballPositionMeanPre.x = " << m_ballPositionMeanPre.x << endl;
			}
			if (m_dnnBallDetectedCC > 1)
			{
				Rect ballPositionMean = Rect(0, 0, 0, 0);
				m_ballPositionSum.x = m_ballPositionSum.x + outRecBallDnn.x;
				ballPositionMean.x = m_ballPositionSum.x / m_dnnBallDetectedCC;
				m_ballPositionSum.y = m_ballPositionSum.y + outRecBallDnn.y;
				ballPositionMean.y = m_ballPositionSum.y / m_dnnBallDetectedCC;

#ifdef DebugJJ
				cout << "outRecBallDnn.x = " << outRecBallDnn.x << endl;
				cout << "m_ballPositionSum.x = " << m_ballPositionSum.x << endl;
				cout << "m_ballPositionMean.x = " << ballPositionMean.x << endl;
#endif
				float dissFinal = sqrt(((ballPositionMean.x - m_ballPositionMeanPre.x)*(ballPositionMean.x - m_ballPositionMeanPre.x)) + ((ballPositionMean.y - m_ballPositionMeanPre.y)*(ballPositionMean.y - m_ballPositionMeanPre.y)));
				if (dissFinal < 10 && m_dnnBallDetectedCC > 2)
				{
					//m_checkBallPoistion = true;
					outRecBallDnn.x = outRecBallDnn.x / m_ballImgSizeUpScale + m_ballRegion.x;
					outRecBallDnn.y = outRecBallDnn.y / m_ballImgSizeUpScale + m_ballRegion.y;

#ifdef DebugJJ
					cout << "outRecBallDnn.x = " << outRecBallDnn.x << endl;
#endif

					m_finalBallROI = Rect(outRecBallDnn.x - m_windowSizeHalf, outRecBallDnn.y - m_windowSizeHalf,
						m_windowSizeHalf * 4, m_windowSizeHalf * 4);

					m_checkBallPoistion = true;
					return true;
				}
				
				m_ballPositionMeanPre = ballPositionMean;

			}
		}
	}

#endif
	
	//cout << " .5 - " << curIdx << endl;

	//int m_ballImgSizeUpScale = 2;  //m_ballImgSizeUpScale
	int m_meanShiftSp = 5;
	int m_meanShiftSr = 20;
	float m_blobMinDistBetweenBlobs = 40.0f;
	float m_blobMinConvexity = 0.8f;   //jj 0925
	float m_blobMinCircularity = 0.4f;
	float m_blobMaxCircularity = 1.0f;

	// scale= 3 fixed
	int m_blobMinArea = 400;  //400
	int m_blobMaxArea = 2000;
	// scale= 4
	//int m_blobMinArea = 600;
	//int m_blobMaxArea = 2000;
	Mat redImg = m_results[curIdx].curRedImg(m_ballRegion).clone();
	Mat blueImg = m_results[curIdx].curBlueImg(m_ballRegion).clone();
	Mat GreenImg = m_results[curIdx].curGreenImg(m_ballRegion).clone();
	

	//Mat outChannel = yccA[2].clone();   //Cb
	

	
	
	//cv::resize(outChannel, outChannel, Size(outChannel.cols * m_ballImgSizeUpScale, outChannel.rows * m_ballImgSizeUpScale), 1, 1, CV_INTER_LANCZOS4);
	//bilateralFilterjj(outChannel, outChannel, 20, 10);

	//Mat X = interpolate_rgb(outChannel);
	//Mat Y = interpolate_rgb(X);
	//imshow("sum", Y);
	//cvWaitKey(1);

	cv::resize(redImg, redImg, Size(redImg.cols * m_ballImgSizeUpScale, redImg.rows * m_ballImgSizeUpScale), 1, 1, CV_INTER_LANCZOS4);

	
	cv::resize(blueImg, blueImg, Size(blueImg.cols * m_ballImgSizeUpScale, blueImg.rows * m_ballImgSizeUpScale), 1, 1, CV_INTER_LANCZOS4);
	cv::resize(GreenImg, GreenImg, Size(GreenImg.cols * m_ballImgSizeUpScale, GreenImg.rows * m_ballImgSizeUpScale), 1, 1, CV_INTER_LANCZOS4);
	//cv::imshow("meanfilter", re);
	//cvWaitKey(1);
	Mat outChannel;
	//Mat blueTemp = blueImg.clone();
	//bilateralFilterjj(blueTemp, blueTemp, 10, 10);
	//bilateralFilterjj(yccA[0], yccA[0], 10, 10);
//	Mat ycc;
//	cv::cvtColor(inputImg, ycc, COLOR_BGR2YCrCb);     //COLOR_BGR2YCrCb
//	cv::Mat yccA[3];
//	cv::split(ycc, yccA);

	absdiff(redImg, blueImg, outChannel);
	//double alpha = 0.5;
	//double beta = (1.0 - alpha);
	//addWeighted(yccA[0], alpha, redImg, beta, 0.0, outChannel);
	
	//Rect persionROI;
	//Point centerP;
	//cv::imshow("iiiii", inputImg);
	//cvWaitKey(1);
	//detectedPerson(inputImg, 0.5, persionROI, centerP);

	//cv::blur(re, re, Size(3, 3));

	//cv::resize(re, re, Size(re.cols * m_ballImgSizeUpScale, re.rows * m_ballImgSizeUpScale), 1, 1, CV_INTER_LANCZOS4);
	//cv::blur(re, re, Size(3, 3));
	//cv::cvtColor(inputImg, inputImg, CV_BGR2GRAY);
	
	
	//cv::imshow("jjredImg before", redImg);
	//cv::imshow("jjblueImg before", blueImg);
	//cv::imshow("jjGreenImg before", GreenImg);
	//cvWaitKey(1);

	Mat bilateralFilterOutRed, bilateralFilterOutA, bilateralFilterOutB;
	//Mat bilateralFilterOutPre;
	//bilateralFilter(inputImg, bilateralFilterOutPre, 10, 100, 50);
	//bilateralFilter(bilateralFilterOutPre, bilateralFilterOutRed, 10, 100, 50);
	//Mat bilateralFilterOutaaa;
	//bilateralFilter(inputImg, bilateralFilterOutaaa, 10, 100, 50);
	//Mat reee;

	//clock_t start, end;
	//start = clock();
	//Mat asdf;

	//input : 1 ch
	//bilateralFilterjj(inputImg, bilateralFilterOutRed, 10, 10);
	bilateralFilterjj(redImg, bilateralFilterOutRed, 20, 10);   //Y

	//m_results[curIdx].curBlueImg(m_ballRegion).clone();
	Mat bilateralFilterOutBlue;
	bilateralFilterjj(blueImg, bilateralFilterOutBlue, 20, 10);


	//cv::imshow("bilateralFilterOutRed", bilateralFilterOutRed);
	//cv::imshow("bilateralFilterOutBlue", bilateralFilterOutBlue);
	//cvWaitKey(1);

	//bilateralFilterjj(inputImg, bilateralFilterOutRed, 20, 10);
	//bilateralFilterjj(inputImg, bilateralFilterOutA, 50, 10);
	//bilateralFilterjj(inputImg, bilateralFilterOutB, 20, 500);
	//pyrMeanShiftFiltering(bilateralFilterOutRed, bilateralFilterOutRed, m_meanShiftSp, m_meanShiftSr);
	//cv::blur(inputImg, inputImg, Size(3, 3));
	//Mat threshold_output;
	//end = clock(); //시간 측정 끝
	//double resultTime = (double)(end - start);
	//printf("bilateralFilterjj time : %f\n", resultTime);
	//imwrite(".\\BF.jpg", bilateralFilterOutRed);
	//imwrite(".\\BF_before.jpg", inputImg);
	//Mat motion;
	//cv::absdiff(inputImg, bilateralFilterOutRed, motion);


	//src_gray = bilateralFilterOutRed.clone();
	//namedWindow("Canny Edge", WINDOW_AUTOSIZE);
	//createTrackbar("Min Threshold:", "Canny Edge", &lowThreshold, 1000, CannyThreshold);
	//createTrackbar("Max Threshold:", "Canny Edge", &highThreshold, 1000, CannyThreshold);
	//CannyThreshold(0, 0);
	//waitKey(0);



	//vector<KeyPoint> keypointsD;
	//Ptr<FastFeatureDetector> detectorFast = FastFeatureDetector::create();
	//vector<Mat> descriptor;

	//Mat fastOut = bilateralFilterOutRed.clone();
	//detectorFast->detect(bilateralFilterOutRed, keypointsD, Mat());
	//drawKeypoints(fastOut, keypointsD, fastOut);

	//Mat dst;
	//int kernel_size = 3;
	//int scale = 1;
	//int delta = 0;
	//int ddepth = CV_16S;
	//Mat abs_dst;

	//Laplacian(inputImg, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(dst, abs_dst);


	
/*
	vector<Vec3f> circles;
	// 2,   // 누적기 해상도(영상크기/2)
	//50,  // 두 원 간의 최소 거리
	//	200, // 캐니 최대 경계값
	//	100, // 투표 최소 개수
	//	25, 100); // 최소와 최대 반지름
	cv::HoughCircles(outChannel, circles, HOUGH_GRADIENT, 1,
		50,  // 두 원 간의 최소 거리
		50,   // 캐니 최대 경계값
		20,   // 투표 최소 개수 10
		10, 20);  // 최소와 최대 반지름
	Mat houghOut = outChannel.clone();
	cv::cvtColor(houghOut, houghOut, CV_GRAY2BGR);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center(c[0], c[1]);
		int radius = c[2];

		circle(houghOut, center, radius, Scalar(0, 255, 0), 2);
		circle(houghOut, center, 2, Scalar(0, 0, 255), 3);

		//ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].x = c[0];
		//ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].y = c[1];
		//ballCandidates.ballCandidates[ballCandidates.ballCandidatesNum].size = c[2];
		//ballCandidates.ballCandidatesNum++;

	}

	cv::imshow("houghOut", houghOut);
	*/

	//EDCircles testEDCircles = EDCircles(bilateralFilterOutRed);
	//std::vector<mCircle> circlesEDout = testEDCircles.getCircles();
	//Mat circleImg = testEDCircles.drawResult(false, ImageStyle::CIRCLES);
	//cv::imshow("ED", circleImg);
	//mCircle outEDCircles;


	//start = clock();
	//Mat thresholdRe;
	//threshold(outChannel, thresholdRe, 0, 255, THRESH_OTSU);
	//end = clock(); //시간 측정 끝
	//resultTime = (double)(end - start);
	//printf("threshold time : %f\n", resultTime);
	
	//Mat thresholdReThird;
	//threshold(bilateralFilterOutRed, thresholdReThird, 20, 255, THRESH_BINARY_INV);

//	start = clock();
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = m_blobMinDistBetweenBlobs;
	params.filterByConvexity = true;
	params.minConvexity = m_blobMinConvexity;  //1 : very convexity,  0.9;
	params.filterByColor = false;
	params.blobColor = 255;
	params.filterByCircularity = false;
	params.minCircularity = m_blobMinCircularity;
	params.maxCircularity = m_blobMaxCircularity;
	params.filterByArea = true;
	params.minArea = m_blobMinArea;
	params.maxArea = m_blobMaxArea;

	//SimpleBlobDetector::Params paramsSecond;
	//paramsSecond.minDistBetweenBlobs = m_blobMinDistBetweenBlobs;
	//paramsSecond.filterByConvexity = true;
	//paramsSecond.minConvexity = 0.4;  //1 : very convexity,  0.9;
	//paramsSecond.filterByColor = true;
	//paramsSecond.blobColor = 255;
	//paramsSecond.filterByCircularity = false;
	//paramsSecond.minCircularity = m_blobMinCircularity;
	//paramsSecond.maxCircularity = m_blobMaxCircularity;
	//paramsSecond.filterByArea = true;
	//paramsSecond.minArea = m_blobMinArea;
	//paramsSecond.maxArea = m_blobMaxArea;

	std::vector<KeyPoint> firstTmp;
	std::vector<KeyPoint> secondTmp;  //ball
	std::vector<KeyPoint> thirdTmp;

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	//Ptr<SimpleBlobDetector> detectorSecond = SimpleBlobDetector::create(paramsSecond);


	detector->detect(bilateralFilterOutRed, firstTmp);
	detector->detect(outChannel, secondTmp);
	detector->detect(bilateralFilterOutBlue, thirdTmp);
	
	//detectorSecond->detect(thresholdReThird, thirdTmp);
	//detectorSecond->detect(cannyOut, secondTmp);

	//end = clock(); //시간 측정 끝
	//resultTime = (double)(end - start);
	//printf("SimpleBlobDetector time : %f\n", resultTime);

	if (m_onece)
	{
		cout << "m_onece ok "  << endl;
		m_blob = Mat::zeros(GreenImg.rows, GreenImg.cols, CV_8UC3);
		m_blob.setTo(255);

		m_onece = false;
	}
	


	for (const auto& keypoint : firstTmp) {
        candidateBall b;
        b.x=int(keypoint.pt.x/m_ballImgSizeUpScale+m_ballRegion.x);
        b.y=int(keypoint.pt.y/m_ballImgSizeUpScale+m_ballRegion.y);
        b.size=int(keypoint.size/m_ballImgSizeUpScale);b.idx=0;
        m_results[curIdx].paperCandidates.push_back(b);
        // Preserve the bounded legacy buffer only for unused legacy consumers.
        if(m_results[curIdx].ballCandidatesNum<candidatesBall_Num_Max)
            m_results[curIdx].ballCandidates[m_results[curIdx].ballCandidatesNum++]=b;
    }

	//jj 0924
	for (const auto& keypoint : secondTmp) {
        candidateBall b;
        b.x=int(keypoint.pt.x/m_ballImgSizeUpScale+m_ballRegion.x);
        b.y=int(keypoint.pt.y/m_ballImgSizeUpScale+m_ballRegion.y);
        b.size=int(keypoint.size/m_ballImgSizeUpScale);b.idx=0;
        m_results[curIdx].paperCandidates.push_back(b);
        // Preserve the bounded legacy buffer only for unused legacy consumers.
        if(m_results[curIdx].ballCandidatesNum<candidatesBall_Num_Max)
            m_results[curIdx].ballCandidates[m_results[curIdx].ballCandidatesNum++]=b;
    }

	//for (size_t s = 0; s < circlesEDout.size(); s++)
	//{
	//	m_results[curIdx].ballCandidates[m_results[curIdx].ballCandidatesNum].x = circlesEDout[s].center.x / m_ballImgSizeUpScale + m_ballRegion.x;
	//	m_results[curIdx].ballCandidates[m_results[curIdx].ballCandidatesNum].y = circlesEDout[s].center.y / m_ballImgSizeUpScale + m_ballRegion.y;
	//	m_results[curIdx].ballCandidates[m_results[curIdx].ballCandidatesNum].size = circlesEDout[s].r / m_ballImgSizeUpScale;;
	//	m_results[curIdx].ballCandidates[m_results[curIdx].ballCandidatesNum].idx = 1;
	//	m_results[curIdx].ballCandidatesNum++;
	//	if (m_results[curIdx].ballCandidatesNum > candidatesBall_Num_Max)
	//		m_results[curIdx].ballCandidatesNum = candidatesBall_Num_Max - 2;
	//}

	for (const auto& keypoint : thirdTmp) {
        candidateBall b;
        b.x=int(keypoint.pt.x/m_ballImgSizeUpScale+m_ballRegion.x);
        b.y=int(keypoint.pt.y/m_ballImgSizeUpScale+m_ballRegion.y);
        b.size=int(keypoint.size/m_ballImgSizeUpScale);b.idx=0;
        m_results[curIdx].paperCandidates.push_back(b);
        // Preserve the bounded legacy buffer only for unused legacy consumers.
        if(m_results[curIdx].ballCandidatesNum<candidatesBall_Num_Max)
            m_results[curIdx].ballCandidates[m_results[curIdx].ballCandidatesNum++]=b;
    }

	

	
#ifdef DebugJJ
	cout << " " << m_results[curIdx].ballCandidatesNum << "----------------" << endl;
	//cv::imshow("thresholdRe", thresholdRe);
	cv::imshow("inputImg_green", outChannel);
	cv::imshow("bilateralFilterOutRed", bilateralFilterOutRed);
	cv::imshow("bilateralFilterOutBlue", bilateralFilterOutBlue);
	Mat im_with_keypoints = bilateralFilterOutRed.clone();
	cv::cvtColor(im_with_keypoints, im_with_keypoints, CV_GRAY2BGR);
	for (size_t t = 0; t < firstTmp.size(); t++)
		circle(im_with_keypoints, CvPoint(firstTmp[t].pt.x, firstTmp[t].pt.y), firstTmp[t].size*0.5, Scalar(0, 0, 255), 1, -1);
	for (size_t at = 0; at < secondTmp.size(); at++)
		circle(im_with_keypoints, CvPoint(secondTmp[at].pt.x, secondTmp[at].pt.y), secondTmp[at].size*0.5, Scalar(0, 255, 0), 1, -1);
	//for (size_t at = 0; at < circlesEDout.size(); at++)
	//	circle(im_with_keypoints, CvPoint(circlesEDout[at].center.x, circlesEDout[at].center.y), circlesEDout[at].r, Scalar(0, 255, 0), 1, -1);
	for (size_t at = 0; at < thirdTmp.size(); at++)
		circle(im_with_keypoints, CvPoint(thirdTmp[at].pt.x, thirdTmp[at].pt.y), thirdTmp[at].size*0.5, Scalar(255, 0, 0), 1, -1);
	//cv::imwrite(".\\BallROI.jpg", im_with_keypoints);
	cv::imshow("im_with_keypoints", im_with_keypoints);
	cvWaitKey(0);
#endif
	//circlesEDout.~vector< mCircle>();
	//firstTmp.~vector< KeyPoint>();
	//secondTmp.~vector< KeyPoint>();

	return true;
}


void kGolfPose::kFlowEstimation(analysisFactor& flowBody)
{
	int m_flowBlurKernelSize = 3;
	int m_flowWindowSize = 10;
	//float sumE = 0;
	Mat curImg = m_results[flowBody.idx].curGrayImg(m_flowRegion).clone();
	Mat preImg = m_results[flowBody.idx].preGrayImg(m_flowRegion).clone();

	cv::blur(curImg, curImg, Size(m_flowBlurKernelSize, m_flowBlurKernelSize));
	cv::blur(preImg, preImg, Size(m_flowBlurKernelSize, m_flowBlurKernelSize));

	

	Mat flowUmat, flow;
	calcOpticalFlowFarneback(preImg, curImg, flowUmat, 0.5, 1, 40, 2, 8, 1.2, 0);
	// copy Umat container to standard Mat
	flowUmat.copyTo(flow);

	//Mat flowRe = inputColorCResize(flowROI).clone();
	float sumFlowEnergy = 0;
	for (int y = 0; y < curImg.rows; y += m_flowWindowSize)
	{
		for (int x = 0; x < curImg.cols; x += m_flowWindowSize)
		{
			// get the flow from y, x position * 10 for better visibility
			const Point2f flowatxy = flow.at<Point2f>(y, x) * m_flowWindowSize;
			// draw line at flow direction
			int ax = cvRound(x + flowatxy.x);
			int ay = cvRound(y + flowatxy.y);
			float len = sqrt(((x - ax)*(x - ax)) + ((y - ay)*(y - ay)));
			
			//line(flowRe, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(0, 0, 255));
			//circle(flowRe, Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), 2, Scalar(255, 0, 0), -1);

			//float gradient = atan2f(ay - y, ax - x)*(180 / CV_PI);
			//if (gradient < 0)
			//	gradient += 359;
			sumFlowEnergy += len;
		}
	}
	flowBody.flowEnergySum = sumFlowEnergy;

	//Mat m_inputBlueCTemp = m_inputBlueC.clone();
	//cv::rectangle(m_inputBlueCTemp, m_flowRegion, Scalar(0, 0, 0), 2);
	//cv::imwrite(".\\personFlow.jpg", m_inputBlueCTemp);
	//cv::imshow("curImgPROI", m_inputBlueCTemp);
	//cvWaitKey(1);
}

void kGolfPose::kFlowEstimationIdx(int curIdx)
{
	int m_flowBlurKernelSize = 3;
	int m_flowWindowSize = 10;
	//float sumE = 0;
	Mat curImg = m_results[curIdx].curGrayImg(m_flowRegion).clone();
	Mat preImg = m_results[curIdx].preGrayImg(m_flowRegion).clone();

	cv::blur(curImg, curImg, Size(m_flowBlurKernelSize, m_flowBlurKernelSize));
	cv::blur(preImg, preImg, Size(m_flowBlurKernelSize, m_flowBlurKernelSize));


	
	Mat flowUmat, flow;
	calcOpticalFlowFarneback(preImg, curImg, flowUmat, 0.5, 1, 40, 2, 8, 1.2, 0);
	// copy Umat container to standard Mat
	
	
	flowUmat.copyTo(flow);
#ifdef DebugJJ
	Mat flowRe = m_results[curIdx].curImg(m_flowRegion).clone();
#endif
	float sumFlowEnergy = 0;
	for (int y = 0; y < curImg.rows; y += m_flowWindowSize)
	{
		for (int x = 0; x < curImg.cols; x += m_flowWindowSize)
		{
			// get the flow from y, x position * 10 for better visibility
			const Point2f flowatxy = flow.at<Point2f>(y, x) * m_flowWindowSize;
			// draw line at flow direction
			int ax = cvRound(x + flowatxy.x);
			int ay = cvRound(y + flowatxy.y);
			float len = sqrt(((x - ax)*(x - ax)) + ((y - ay)*(y - ay)));
#ifdef DebugJJ
			line(flowRe, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(0, 0, 255));
			circle(flowRe, Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), 1, Scalar(255, 0, 0), -1);
#endif
			//float gradient = atan2f(ay - y, ax - x)*(180 / CV_PI);
			//if (gradient < 0)
			//	gradient += 359;
			sumFlowEnergy += len;
		}
	}
	m_results[curIdx].flowEnergySum = sumFlowEnergy;

	//Mat m_inputBlueCTemp = m_inputBlueC.clone();
	//cv::rectangle(m_inputBlueCTemp, m_flowRegion, Scalar(0, 0, 0), 2);
	//cv::imwrite(".\\personFlow.jpg", m_inputBlueCTemp);
#ifdef DebugJJ
	//cv::imwrite(".\\personFlow.jpg", flowRe);
	cout << "idx = " << curIdx << ", " << "energy = " << sumFlowEnergy << endl;
	cv::imshow("flowRe", flowRe);
	cvWaitKey(1);
#endif
}

void kGolfPose::kGolfBallDetection(analysisFactor& analy)
{
	//int m_ballImgSizeUpScale = 2;  //m_ballImgSizeUpScale
	//analysisFactor tmp;

	//balance_white(m_inputC);

	//Mat BallROIInput = m_inputBlueC(m_ballRegion).clone();
	//Mat BallROIInput = m_inputBlueC(m_ballRegion).clone();
	Mat BallROIInput = m_results[analy.idx].curBlueImg(m_ballRegion).clone();


	//Mat BallROIout = BallROIInput.clone();

	//cv::cvtColor(BallROIInput, BallROIInput, CV_GRAY2BGR);

	//clock_t start, end;
	//start = clock();
	//input=color, out = 3ch
	//imgEnhance(BallROIInput, BallROIout);


	

	//cv::Mat inputCC[3];
	//cv::split(BallROIInput, inputCC);
	//Mat out = inputCC[0].clone();


	//end = clock(); //시간 측정 끝
	//double resultTime = (double)(end - start);
	//printf("imgEnhance : %f\n", resultTime);

	//Mat equalizedBallROI;
	//equalizeHist(BallROI, equalizedBallROI);
	//cv::imshow("equalizedBallROI", equalizedBallROI);

	
	//candidateBall outBallCandidates;
	//cv::KeyPoint tempk;
	//for (std::size_t i = 0; i < 20; ++i)
	//	outBallCandidates.push_back(tempk);
	//start = clock();
	blobDetection(BallROIInput, analy);
	//end = clock(); //시간 측정 끝
	//resultTime = (double)(end - start);
	//printf("blobDetection : %f\n", resultTime);
	for (int b = 0; b < analy.ballCandidatesNum; b++)
	{
		analy.ballCandidates[b].x = analy.ballCandidates[b].x / m_ballImgSizeUpScale + m_ballRegion.x;
		analy.ballCandidates[b].y = analy.ballCandidates[b].y / m_ballImgSizeUpScale + m_ballRegion.y;
		analy.ballCandidates[b].size = analy.ballCandidates[b].size / m_ballImgSizeUpScale;
	}
}

void kGolfPose::kGolfBallDetectionIdx(int curIdx)
{
	//int m_ballImgSizeUpScale = 2;  //m_ballImgSizeUpScale
	//analysisFactor tmp;

	//balance_white(m_inputC);

	//Mat BallROIInput = m_inputBlueC(m_ballRegion).clone();
	//Mat BallROIInput = m_inputBlueC(m_ballRegion).clone();
	//Mat BallROIInput = m_results[curIdx].curBlueImg(m_ballRegion).clone();


	//Mat BallROIout = BallROIInput.clone();

	//cv::cvtColor(BallROIInput, BallROIInput, CV_GRAY2BGR);

	//clock_t start, end;
	//start = clock();
	//input=color, out = 3ch
	
	Mat BallROIout= m_results[curIdx].curImg(m_ballRegion).clone();

	


#ifdef DebugJJ
	Mat beforeBallROIoutEnhance;
	imshow("before-BallEnh", BallROIout);
	cvWaitKey(1);
#endif


	imgEnhance(BallROIout, BallROIout);
	
#ifdef DebugJJ
	Mat afterBallROIoutEnhance;
	imshow("after-BallEnh", BallROIout);
	cvWaitKey(1);
#endif
	
	
	
	//Mat yy = yccA[1].clone();


	//cv::Mat inputCC[3];
	//cv::split(BallROIInput, inputCC);
	//Mat out = inputCC[0].clone();


	//end = clock(); //시간 측정 끝
	//double resultTime = (double)(end - start);
	//printf("imgEnhance : %f\n", resultTime);

	//Mat equalizedBallROI;
	//equalizeHist(BallROI, equalizedBallROI);
	//cv::imshow("equalizedBallROI", equalizedBallROI);


	//candidateBall outBallCandidates;
	//cv::KeyPoint tempk;
	//for (std::size_t i = 0; i < 20; ++i)
	//	outBallCandidates.push_back(tempk);
	//start = clock();
	
	
	//blobDetection(BallROIInput, analy);

 
	bool ck = blobDetectionIdx(BallROIout, curIdx);

	//end = clock(); //시간 측정 끝
	//resultTime = (double)(end - start);
	//printf("blobDetection : %f\n", resultTime);
	


	//for (int b = 0; b < m_results[curIdx].ballCandidatesNum; b++)
	//{
	//	m_results[curIdx].ballCandidates[b].x = analy.ballCandidates[b].x / m_ballImgSizeUpScale + m_ballRegion.x;
	//	m_results[curIdx].ballCandidates[b].y = analy.ballCandidates[b].y / m_ballImgSizeUpScale + m_ballRegion.y;
	//	m_results[curIdx].ballCandidates[b].size = analy.ballCandidates[b].size / m_ballImgSizeUpScale;
	//}
}


void kGolfPose::analysisFactorInit(analysisFactor& tmp)
{
	tmp.ballCandidatesNum = 0;
	tmp.idx = m_result_Max;
	tmp.flowEnergySum = -1;
	tmp.localImpactBGSCount = 0;

	for (size_t j = 0; j < candidatesBall_Num_Max; j++)
	{
		tmp.ballCandidates[j].x = -1;
		tmp.ballCandidates[j].y = -1;
		tmp.ballCandidates[j].size = -1;
	}
}

int kGolfPose::estimationImpactSide()
{
    Rect gate(m_ballPosition.x-20,m_ballPosition.y-20,40,40);
    Rect bounds(0,0,m_results[1].curGrayImg.cols,m_results[1].curGrayImg.rows);
    if((gate & bounds)!=gate || (m_finalBallROI & bounds)!=m_finalBallROI) {
        m_failureReason="invalid_verification_roi";return -1;
    }
    Mat erosion(3,3,CV_8U,Scalar(1)),empty,out;
    int gateMax=0;
    for(int i=1;i<=m_resultsNum;++i) {
        if(m_results[i].preGrayImg.empty() || m_results[i].curGrayImg.empty()) continue;
        kBGS(m_results[i].preGrayImg,m_results[i].curGrayImg,gate,0,0.1f,erosion,empty,out);
        gateMax=std::max(gateMax,countNonZero(out>200));
    }
    cout<<"AUDIT gate_max "<<gateMax<<endl;
    if(gateMax<5) {m_failureReason="practice_gate";return -1;}
    vector<int> change(m_resultsNum+1,-1);
    vector<double> motion(m_resultsNum+1,-1);
    for(int i=5;i<=m_resultsNum-6;++i) {
        if(m_results[i].preGrayImg.empty() || m_results[i].curGrayImg.empty()) {
            m_failureReason="unavailable_change_score";return -1;
        }
        kBGS(m_results[i].preGrayImg,m_results[i].curGrayImg,m_finalBallROI,0,0.1f,erosion,empty,out);
        change[i]=countNonZero(out>200);
        motion[i]=m_results[i].flowEnergySum;
        m_results[i].localImpactBGSCount=change[i];
    }
    auto decision=hit::selectCandidate(change,motion,5,m_resultsNum-6);
    cout<<"AUDIT joint_candidate "<<decision<<endl;
    if(decision<0) {m_failureReason="no_joint_candidate";return -1;}
    int before=hit::stableFrame(change,decision,-1,5,m_resultsNum-6);
    int after=hit::stableFrame(change,decision,1,5,m_resultsNum-6);
    cout<<"AUDIT stable_neighbors "<<before<<" "<<after<<endl;
    if(before<0 || after<0) {m_failureReason="missing_stable_neighbor";return -1;}
    kBGS(m_results[before].curGrayImg,m_results[after].curGrayImg,m_finalBallROI,0,0.1f,erosion,empty,out);
    cout<<"AUDIT stable_change "<<countNonZero(out>200)<<endl;
    if(countNonZero(out>200)<=13) {m_failureReason="stable_verification";return -1;}
    m_impactIdx=decision;return decision;
}


int kGolfPose::estimationImpact(Mat initGrayImg)
{
	Mat out;
	Mat eleEmpty;
	Mat eleT(3, 3, CV_8U, cv::Scalar(1));
	int m_impactInitFrameMargin = 5; // 10;  //jj 0924
	int m_impactDecisionThreshold = 500; // 5; // 50 total 20*20 = 400 pixel  

	for (size_t i = 0; i < m_resultsNum; i++)
	{
		kBGS(initGrayImg, m_results[i].curBlueImg, m_finalBallROI, 0, 0.1, eleT, eleEmpty, out);
		int ccount = 0;
		for (int yy = 0; yy < out.rows; yy++)
		{
			for (int xx = 0; xx < out.cols; xx++)
			{
				if (out.at<uchar>(yy, xx) > 200)
					ccount++;
			}
		}
		cout << ccount << " / " << out.cols  * out.rows << endl;
		Mat outTmp = out.clone();
		cv::resize(outTmp, outTmp, Size(outTmp.cols * 3, outTmp.rows * 3), 1, 1, CV_INTER_LANCZOS4);
		//cv::resize(m_results[i].curBlueImg, m_results[i].curBlueImg, Size(outTmp.cols * 0.4, outTmp.rows * 0.4), 1, 1, CV_INTER_LANCZOS4);
		//imshow("input", m_results[i].curBlueImg);
		imshow("bgs", outTmp);
		cv::waitKey(1);

		if ((ccount > m_impactDecisionThreshold) && (i > m_impactInitFrameMargin))
		{
			//jj 0924
			if (m_results[i].idx == 500)
				return 0;
			else
				return m_results[i].idx;   // -1th image of the detected impact 
		}

		//if (true)
		//{

		//}
		//cv::waitKey(1);
	}
	return 0;
}

//min first
bool comparatorAnalysisFactorFlowEnergy(const analysisFactor& s1, const analysisFactor& s2) 
{
	return s1.flowEnergySum < s2.flowEnergySum;
}
//min first
bool comparatorAnalysisFactorIdx(const analysisFactor& s1, const analysisFactor& s2)
{
	return s1.idx < s2.idx;
}


int kGolfPose::estimationTop(int impactIdx)
{
	m_backIdx = 0;
	int ss = impactIdx - 1 - m_searchFrameRange_Max;
	if (ss < 0)
	{
		return 0;
	}
	else
	{
		analysisFactor back[m_searchFrameRange_Max+5];

		for (size_t i = 0; i < m_searchFrameRange_Max+5; i++)
		{
			back[i].idx = -1;
			back[i].flowEnergySum = FLT_MAX;
			back[i].ballCandidatesNum = 0;
		}
		int end = impactIdx - 1;
		if (end-ss > m_searchFrameRange_Max)
		{
			return 0;
		}

		int cc = 0;

		for (int r = ss; r < end; r++)
		{
			back[cc].idx = m_results[r].idx;
			back[cc].flowEnergySum = m_results[r].flowEnergySum;
			cc++;
		}

		sort(back, back+m_searchFrameRange_Max, resultSortFlow);

		int backIdx = back[0].idx;


		if (m_results[backIdx].flowEnergySum*2 < m_results[m_impactIdx].flowEnergySum)
		{
			m_backIdx = backIdx;
			return backIdx;
		}
		else
			return 0;



	}
}

int kGolfPose::estimationFollow(int impactIdx)
{
	//int m_searchFrameRange = 40;  //m_searchFrameRange = 40
	int ee = impactIdx - 1 + m_searchFrameRange_Max;
	
	if (ee < 0 || ee > m_resultsNum)
		return 0;
	
	if ((ee - impactIdx) > m_searchFrameRange_Max)
		return 0;

	
	//vector<analysisFactor> follow;
	analysisFactor follow[m_searchFrameRange_Max+5];
	for (size_t i = 0; i < m_searchFrameRange_Max+5; i++)
	{
		follow[i].idx = -1;
		follow[i].flowEnergySum = FLT_MAX;
		follow[i].ballCandidatesNum = 0;
	}
	int cc = 0;
	for (int w = impactIdx; w < ee; w++)
	{
		//follow.push_back(m_results[w]);
		follow[cc].idx = m_results[w].idx;
		follow[cc].flowEnergySum = m_results[w].flowEnergySum;
		cc++;
	}
	//std::sort(follow.begin(), follow.end(), comparatorAnalysisFactorFlowEnergy);
	//std::qsort(follow, m_searchFrameRange_Max + 10, sizeof(analysisFactor),	resultSortFlow);
	sort(follow, follow + m_searchFrameRange_Max+5, resultSortFlow);

	int imIdx = follow[0].idx;

	//return imIdx;
	cout << "m_results[imIdx].flowEnergySum * 2 = "  << m_results[imIdx].flowEnergySum * 2  << " " << m_results[m_impactIdx].flowEnergySum << endl;
	
	if (m_results[imIdx].flowEnergySum < m_results[m_impactIdx].flowEnergySum)
	{
		//check
		Mat eleEmpty;
		Mat eleT(3, 3, CV_8U, cv::Scalar(1));
		Mat out;
		Rect tempFlowRect;
		tempFlowRect.x = m_finalBallROI.x - (m_finalBallROI.width);
		tempFlowRect.y = m_finalBallROI.y - (m_finalBallROI.height);
		tempFlowRect.width = m_finalBallROI.width*3;
		tempFlowRect.height = m_finalBallROI.height*3;

#ifdef DebugJJ
		Mat BallROIFinal = m_results[50].curImg.clone();
		cv::rectangle(BallROIFinal, tempFlowRect, Scalar(0, 0, 255), 1, 8, 0);
		cv::resize(BallROIFinal, BallROIFinal, Size(BallROIFinal.cols*0.5, BallROIFinal.rows*0.5), 1, 1, CV_INTER_LANCZOS4);
		cv::imshow("tempFlowRect", BallROIFinal);
		cvWaitKey(1);
#endif
		kBGS(m_results[imIdx].preGrayImg, m_results[m_backIdx].curGrayImg, m_finalBallROI, 0, 0.1, eleT, eleEmpty, out);


		int ccount = 0;
		for (int yy = 0; yy < out.rows; yy++)
		{
			for (int xx = 0; xx < out.cols; xx++)
			{
				if (out.at<uchar>(yy, xx) > 200)
					ccount++;
			}
		}
		cout << " imIdx = " << imIdx << "  " << "back-follow bgs ccount = " << ccount << endl;

#ifdef DebugJJ
		Mat outTemp = out.clone();
		cv::resize(outTemp, outTemp, Size(outTemp.cols * 3, outTemp.rows * 3), 1, 1, CV_INTER_LANCZOS4);
		cv::imshow("outTemp", outTemp);
		cvWaitKey(1);
#endif


		//if (ccount > 13)  //2pixel*2pixel*3.14 = 13  // 3pixel*3pixel*3.14 = 29
		//{
		//	return imIdx;
		//}
		//else
		//	return 0;
		return imIdx;
	}
	else
		return 0;
}


void kGolfPose::sortIdx()
{
	//std::sort(m_results.begin(), m_results.end(), comparatorAnalysisFactorIdx);
	//std::qsort(m_results, m_resultsNum, sizeof(analysisFactor), resultSortIdx);
	sort(m_results, m_results + m_result_Max, resultSortIdx);
}


bool kGolfPose::kGolfPreProcessing(Mat firstImg)
{
	//cv::cvtColor(firstImg, firstImg, CV_GRAY2BGR);
	bool re = setROIs(firstImg, m_detectedPersonRect, m_ballRegion, m_flowRegion);
	//m_resultsNum++;
	return re;
}
bool kGolfPose::kGolfPreProcessing(int initImgIdx)
{
	//cv::cvtColor(firstImg, firstImg, CV_GRAY2BGR);
	Mat firstImg = m_results[initImgIdx].curImg;
	bool re = setROIs(firstImg, m_detectedPersonRect, m_ballRegion, m_flowRegion);
	//m_resultsNum++;
	//cout << "rererer == " << re << endl;
	return re;
}


void kGolfPose::kGolfPoseDetection(Mat curRedImg, Mat curBlueImg, int curImgIndex)
{
	analysisFactor analysisFactorTemp;
	analysisFactorInit(analysisFactorTemp);
	analysisFactorTemp.idx = curImgIndex;

	//m_inputC = curBlueImg.clone();
	//cv::Mat inputC[3];
	//cv::split(m_inputC, inputC);
	m_inputBlueC = curBlueImg.clone();

	//m_inputP = curRedImg.clone();
	//cv::Mat inputP[3];
	//cv::split(m_inputP, inputP);
	m_inputBlueP = curRedImg.clone();

	//cv::cvtColor(m_inputC, firstImg, CV_GRAY2BGR);
	//cv::cvtColor(firstImg, firstImg, CV_GRAY2BGR);

	//m_inputBlueP = curRedImg.clone();
	//m_inputBlueC = curBlueImg.clone();
	analysisFactorTemp.curBlueImg = curBlueImg.clone();
	
	//clock_t start, end;
	//start = clock();
	kGolfBallDetection(analysisFactorTemp);
	//end = clock(); //시간 측정 끝
	//double resultTime = (double)(end - start);
	//printf("kGolfBallDetection total : %f\n", resultTime);
	//start = clock();
	kFlowEstimation(analysisFactorTemp);
	//end = clock(); //시간 측정 끝
	//resultTime = (double)(end - start);
	//printf("kFlowEstimation total : %f\n", resultTime);

	m_results[curImgIndex] = analysisFactorTemp;
	m_resultsNum++;
}

void kGolfPose::kGolfPoseDetectionIdx(int curImgIndex)
{
	//analysisFactor analysisFactorTemp;
	//analysisFactorInit(analysisFactorTemp);
	
	//m_results[curImgIndex].idx = curImgIndex;

	//m_inputC = curBlueImg.clone();
	//cv::Mat inputC[3];
	//cv::split(m_inputC, inputC);
//	m_inputBlueC = curBlueImg.clone();

	//m_inputP = curRedImg.clone();
	//cv::Mat inputP[3];
	//cv::split(m_inputP, inputP);
	//m_inputBlueP = curRedImg.clone();

	//cv::cvtColor(m_inputC, firstImg, CV_GRAY2BGR);
	//cv::cvtColor(firstImg, firstImg, CV_GRAY2BGR);

	//m_inputBlueP = curRedImg.clone();
	//m_inputBlueC = curBlueImg.clone();
	//analysisFactorTemp.curBlueImg = curBlueImg.clone();

	//clock_t start, end;
	//start = clock();
	//kGolfBallDetection(analysisFactorTemp);



	

	if (curImgIndex <= 120)
		kGolfBallDetectionIdx(curImgIndex);




	//end = clock(); //시간 측정 끝
	//double resultTime = (double)(end - start);
	//printf("kGolfBallDetection total : %f\n", resultTime);
	//start = clock();
	//kFlowEstimation(analysisFactorTemp);
	kFlowEstimationIdx(curImgIndex);


	//end = clock(); //시간 측정 끝
	//resultTime = (double)(end - start);
	//printf("kFlowEstimation total : %f\n", resultTime);

	//m_results[curImgIndex].flowEnergySum = analysisFactorTemp.flowEnergySum;
	//m_results[curImgIndex].ballCandidates = analysisFactorTemp.ballCandidates;

	// **여기에서 지표를 기록**
	recordMetrics(curImgIndex);


	m_resultsNum++;
}

#if 1
bool kGolfPose::calBGS()
{
	Mat fiveKernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	Mat sevenKernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	Mat fifthKernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));


	for (int u = 1; u < totalImgNumMember; u++)
	{
		m_results[u].curBGSImg = Mat(m_ballRegionEx.height, m_ballRegionEx.width, CV_8UC1, Scalar(0));

		m_results[u].m_preGrayBlurROIImg = m_results[u].preGrayImg(m_ballRegionEx).clone();
		cv::blur(m_results[u].m_preGrayBlurROIImg, m_results[u].m_preGrayBlurROIImg, Size(5, 5));
		m_results[u].m_curGrayBlurROIImg = m_results[u].curGrayImg(m_ballRegionEx).clone();
		cv::blur(m_results[u].m_curGrayBlurROIImg, m_results[u].m_curGrayBlurROIImg, Size(5, 5));
		m_results[u].m_fuGrayBlurROIImg = m_results[u].fuGrayImg(m_ballRegionEx).clone();
		cv::blur(m_results[u].m_fuGrayBlurROIImg, m_results[u].m_fuGrayBlurROIImg, Size(5, 5));
	}
	//imshow("tssete", m_results[1].preGrayImg(m_persionExROI));
	//waitKey(0);


	cout << "ckeck point 2" << endl;

	int a;
	for (a = 1; a < totalImgNumMember; a++)
	{
		//cv::imshow("ori ", m_results[a].m_curGrayBlurROIImg);
		Mat preBGS;
		Ptr<BackgroundSubtractorMOG2> BGSPre;
		BGSPre = createBackgroundSubtractorMOG2(); //MOG2 approach
		BGSPre->setDetectShadows(0);
		BGSPre->setBackgroundRatio(1);

		//pre를 기준으로 cur 빼?
		BGSPre->apply(m_results[a].m_preGrayBlurROIImg, preBGS); //처음 pre 넣어줌
		BGSPre->apply(m_results[a].m_curGrayBlurROIImg, preBGS);
		//cv::imshow("b-PrCu", preBGS);
		//cvWaitKey(1);
		cv::morphologyEx(preBGS, preBGS, cv::MORPH_OPEN, sevenKernel);   //  fifthKernel   sevenKernel
		//cv::imshow("a-PrCu", preBGS);
		//cvWaitKey(1);

		Mat fuBGS;
		Ptr<BackgroundSubtractorMOG2> BGSFu;
		BGSFu = createBackgroundSubtractorMOG2(); //MOG2 approach
		BGSFu->setDetectShadows(0);
		BGSFu->setBackgroundRatio(1);

		BGSFu->apply(m_results[a].m_fuGrayBlurROIImg, fuBGS);
		BGSFu->apply(m_results[a].m_curGrayBlurROIImg, fuBGS);
		//cv::imshow("b-FuCu", fuBGS);
		//cvWaitKey(1);
		cv::morphologyEx(fuBGS, fuBGS, cv::MORPH_OPEN, sevenKernel);   //  fifthKernel   sevenKernel
		//cv::imshow("a-FuCu", fuBGS);
		//cvWaitKey(1);


		for (int y = 0; y < preBGS.rows; y++)
		{
			for (int x = 0; x < preBGS.cols; x++)
			{
				if (preBGS.at<uchar>(y, x) > 200 && fuBGS.at<uchar>(y, x) > 200)
					m_results[a].curBGSImg.at<uchar>(y, x) = 255;
			}
		}
	
		//cv::imshow("b", m_results[a].curBGSImg);
		//cvWaitKey(1);

		cv::morphologyEx(m_results[a].curBGSImg, m_results[a].curBGSImg, cv::MORPH_DILATE, sevenKernel);   //  fifthKernel   sevenKernel

		//cv::imshow("a", m_results[a].curBGSImg);
		//cvWaitKey(0);
	}

	//Mat fuBGSOut;
	//Ptr<BackgroundSubtractorMOG2> BGSFuOut;
	//BGSFuOut = createBackgroundSubtractorMOG2(); //MOG2 approach
	//BGSFuOut->setDetectShadows(0);
	//BGSFuOut->setBackgroundRatio(1);

	//for (int s = 1; s < totalImgNumMember; s++)
	//{
	//	BGSFuOut->apply(m_results[s].m_fuGrayBlurROIImg, fuBGSOut);
	//	BGSFuOut->apply(m_results[s].m_curGrayBlurROIImg, fuBGSOut);
	//	cv::morphologyEx(fuBGSOut, fuBGSOut, cv::MORPH_DILATE, sevenKernel);
	//	//cv::morphologyEx(m_results[s].curBGSImg, m_results[s].curBGSImg, cv::MORPH_DILATE, fiveKernel);
	//	Mat temp = fuBGSOut.clone();
	//	cv::imshow("a", m_results[s].curBGSImg);
	//	for (int y = 0; y < fuBGSOut.rows; y++)
	//	{
	//		for (int x = 0; x < fuBGSOut.cols; x++)
	//		{
	//			if (fuBGSOut.at<uchar>(y, x) > 200 && m_results[s].curBGSImg.at<uchar>(y, x) > 200)
	//				m_results[s].curBGSImg.at<uchar>(y, x) = 0;

	//			if (x > m_persionExROI.width)
	//				m_results[s].curBGSImg.at<uchar>(y, x) = 0;
	//		}
	//	}
	//	//cout << s << endl;
	//	//cv::imshow("m_results[s].cur", m_results[s].m_curGrayBlurROIImg);
	//	//cv::imshow("m_results[s].curBGSImg", m_results[s].curBGSImg);
	//	//cvWaitKey(0);
	//}



	return 0;
}

//jj
bool compare_cciIdx(const cciIdx& x, const cciIdx& y)
{
	return x.cc > y.cc;   // > : 큰 값이 처음에 
}

bool kGolfPose::clustering()
{
	//int total = m_filePathsSize - 1;

	for (int i = 1; i < totalImgNumMember; i++)
	{
		//imshow("clustering_before", m_curImg[i].curBGSImg);
		cv::Mat labels, stats, centroids;
		int nlabels = cv::connectedComponentsWithStats(m_results[i].curBGSImg, labels, stats, centroids, 8, CV_32S);


		vector<cciIdx> countLabel;
		int maxLabel = 0;
		int maxLabelcci = 0;

		//find label index max 2
		for (int b = 1; b < nlabels; b++)
		{
			cciIdx temp;
			int cci = 0;
			for (int yyy = 0; yyy < labels.rows; yyy++)
			{
				for (int xxx = 0; xxx < labels.cols; xxx++)
				{
					if (labels.at<int>(yyy, xxx) == b)
						cci++;
				}
			}
			temp.cc = cci;  //area
			temp.idx = b;
			countLabel.push_back(temp);
		}

		//area가 큰게 먼저
		std::sort(countLabel.begin(), countLabel.end(), compare_cciIdx);

		//cal distance
		Point maxAreaEndP = Point(0, 0);
		Point maxAreaSecondEndP = Point(0, 0);
		float diTemp = 0;

		//area가 제일 큰거
		if (countLabel.size() > 0)
		{
			int maxX = 0;
			int maxY = 0;
			for (int yyy = 0; yyy < labels.rows; yyy++)
			{
				for (int xxx = 0; xxx < labels.cols; xxx++)
				{
					if (labels.at<int>(yyy, xxx) == countLabel[0].idx && countLabel[0].idx != 0)
					{
						float dist = sqrt((xxx - m_personCenter.x)*(xxx - m_personCenter.x) + (yyy - m_personCenter.y)*(yyy - m_personCenter.y));

						if (diTemp < dist)
						{
							diTemp = dist;
							maxX = xxx;
							maxY = yyy;
						}
					}
				}
			}
			maxAreaEndP.x = maxX;
			maxAreaEndP.y = maxY;
		}

		float diTempSecond = 0;

		//area가 두번째로 큰거
		if (countLabel.size() > 1)
		{
			int maxX = 0;
			int maxY = 0;

			for (int yyy = 0; yyy < labels.rows; yyy++)
			{
				for (int xxx = 0; xxx < labels.cols; xxx++)
				{
					if (labels.at<int>(yyy, xxx) == countLabel[1].idx && countLabel[1].idx != 0)
					{
						float dist = sqrt((xxx - m_personCenter.x)*(xxx - m_personCenter.x) + (yyy - m_personCenter.y)*(yyy - m_personCenter.y));

						if (diTempSecond < dist)
						{
							diTempSecond = dist;
							maxX = xxx;
							maxY = yyy;
						}
					}
				}
			}
			maxAreaSecondEndP.x = maxX;
			maxAreaSecondEndP.y = maxY;
		}

		if (diTemp == 0 && diTempSecond == 0)
		{
			m_results[i].ckSkipEndPoint = 1;
			continue;
		}

		int finalLabelIndex = 0;

		//2개가 나오면, 중심에서 제일 먼 걸로 end point select
		if (countLabel.size() > 0)
		{
			if (diTemp >= diTempSecond)
			{
				m_results[i].endP = maxAreaEndP;
				finalLabelIndex = countLabel[0].idx;
				if (maxAreaEndP.x == 0 && maxAreaEndP.x == 0)
					m_results[i].ckSkipEndPoint = 1;
				else
					m_results[i].ckSkipEndPoint = 0;
			}
			else
			{
				m_results[i].endP = maxAreaSecondEndP;
				finalLabelIndex = countLabel[1].idx;
				if (maxAreaSecondEndP.x == 0 && maxAreaSecondEndP.x == 0)
					m_results[i].ckSkipEndPoint = 1;
				else
					m_results[i].ckSkipEndPoint = 0;
			}
		}
		else
		{
			m_results[i].endP = Point(0, 0);
			m_results[i].ckSkipEndPoint = 1;
		}

		m_results[i].state = 0;
		m_results[i].accel = 0;
		m_results[i].vel = 0;

		if (finalLabelIndex != 0)
		{
			m_results[i].curBGSImg.setTo(0);
			for (int yyy = 0; yyy < labels.rows; yyy++)
			{
				for (int xxx = 0; xxx < labels.cols; xxx++)
				{
					if (labels.at<int>(yyy, xxx) == finalLabelIndex)
						m_results[i].curBGSImg.at<uchar>(yyy, xxx) = 255;
				}
			}
		}
		else
		{
			m_results[i].ckSkipEndPoint = 1;
			continue;
		}

		//imshow("cur", m_results[i].m_curGrayBlurROIImg);
		//imshow("connectedComponentsWithStats", m_results[i].curBGSImg);
		//waitKey(0);

	}

	return 1;
}

void kGolfPose::recordMetrics(int idx)
{
	// 1) optical flow energy (이미 k_results[idx].flowEnergySum에 계산되어 있음)
	float flowEnergy = m_results[idx].flowEnergySum;
	const Mat& bgsMask = m_results[idx].curBGSImg;
	int bgsCount = countNonZero(bgsMask);

	m_csvFile << idx << "," << flowEnergy << "," << bgsCount << "\n";
	m_csvFile.flush();
}

bool kGolfPose::clusteringFixed()
{
    for(int i=1;i<totalImgNumMember;++i) {
        auto& r=m_results[i]; r.ckSkipEndPoint=true;r.endP=Point(-1,-1);
        if(r.curBGSImg.empty() || r.preGrayImg.empty() || r.fuGrayImg.empty()) continue;
        Mat labels,stats,centroids;
        int n=connectedComponentsWithStats(r.curBGSImg,labels,stats,centroids,8,CV_32S);
        if(n<=1) continue;
        int largest=1;
        for(int k=2;k<n;++k) if(stats.at<int>(k,CC_STAT_AREA)>stats.at<int>(largest,CC_STAT_AREA)) largest=k;
        for(int y=0;y<labels.rows;++y) for(int x=0;x<labels.cols;++x)
            if(labels.at<int>(y,x)==largest && y>r.endP.y) r.endP=Point(x,y);
        r.ckSkipEndPoint=false;
    }
    return true;
}


#endif