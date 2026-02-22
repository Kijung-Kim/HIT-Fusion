#pragma once

//#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>

#include <iostream>
#include <vector>
#include <algorithm> 

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/video/background_segm.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include "dirent.h"
#include <direct.h>
#include <stdlib.h>
#include <string.h>

#include "kGolfPose.h"
#include "spray.h"




using namespace cv;
using namespace std;
using namespace cv::dnn;




int kGolf(Mat* inputArray, Mat* inputArrayBlue, int num, int& impactIdx, int& backIdx, int& followIdx);
int ktGolf(std::string inputRGB, int& impactIdx, int& backIdx, int& followIdx, int& newImpact);


