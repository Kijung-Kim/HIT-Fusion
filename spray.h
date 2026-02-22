#pragma once

#include <iostream>
#include <vector>
#include <algorithm> 


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

void Filter64F(cv::Mat img, cv::Mat &result, int k);
cv::Point2i **CreateSprays(int spraysCount, int n, int R);
void DeleteSprays(cv::Point2i **sprays, int spraysCount);
void PerformRandomSpraysRetinex(cv::Mat source, cv::Mat &destination, int N, int n, double upperBound, int rowsStep, int colsStep);
void PerformLightRandomSpraysRetinex(cv::Mat source, cv::Mat &destination, int N, int n, int inputKernelSize, double inputSigma, int intensityChangeKernelSize, double intensityChangeSigma, int rowsStep, int colsStep, bool normalizeIntensityChange, double upperBound);


