#include "golfBall.h"

int m_dnnBallDetectedCC;
Rect m_ballPositionMeanPre;
Rect m_ballPositionSum;
bool m_checkBallPoistion;


std::vector<string> loadPath(string m_path, string m_ext)
{

	std::vector<string> m_TrainImgFullPath;

	if (m_path[0] == '.') {
		long pwdsize;
		char *pwdbuf = NULL;
		char *pwdptr = NULL;

		//pwdsize = pathconf(".", _PC_PATH_MAX);
		pwdsize = 1000;  //ju

		if ((pwdbuf = (char *)malloc((size_t)pwdsize)) != NULL)
			pwdptr = _getcwd(pwdbuf, (size_t)pwdsize);

		string fullpath(pwdptr);
		fullpath.append(string(&m_path.c_str()[1]));
		////cout<< fullpath <<endl;
		m_path = fullpath;
	}


	DIR* dir = opendir(m_path.c_str());
    if(!dir) return m_TrainImgFullPath;
	dirent* ptr;

	while ((ptr = readdir(dir)) != NULL)
	{
		if (ptr->d_name[0] == '.')
			continue;

		size_t len = strlen(ptr->d_name);
		if (len >= 4) {
			string extension;
			extension.push_back('.');
			extension.append(m_ext);

			if (extension.compare(&(ptr->d_name[len - 4])) == 0) {
				string imgfullpath, matfullpath;
				imgfullpath.append(m_path.c_str());
				imgfullpath.append((string)ptr->d_name);
				m_TrainImgFullPath.push_back(imgfullpath);

			}
		}

	}
	closedir(dir);
	return m_TrainImgFullPath;
}

void test()
{
	string cfg = ".\\yolo.cfg";
	string weight = ".\\yolo.weights";
	kGolfPose kPose(cfg, weight);
	kPose.initResult();
}

//image correction
#if 1   


// White balancing using the method in Gimp https://docs.gimp.org/en/gimp-layer-white-balance.html
Mat white_balance(Mat img, float perc = 0.05)
{
	CV_Assert(img.data);

	// Accept only char  colour type matrices
	CV_Assert(img.depth() != sizeof(uchar));
	// CV_Assert(img.channels() != 3);


	// Compute the histograms of the three colour channels
	int histSize = 256;

	// Set the range
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat r_hist, g_hist, b_hist;

	Mat bgr[3];   //destination array
	split(img, bgr);//split source  

	// Compute the histogram
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	Mat r_norm_hist = r_hist / bgr[2].total();
	Mat g_norm_hist = g_hist / bgr[1].total();
	Mat b_norm_hist = b_hist / bgr[0].total();




	// Now get the CDF

	//std::vector<float> r_cdf;
	//std::vector<float> g_cdf;
	//std::vector<float> b_cdf;

	float r_cdf[256];
	float g_cdf[256];
	float b_cdf[256];

	for (int i = 0; i < 256; i++)
	{
		if (i == 0)
		{
			/*r_cdf.push_back((r_norm_hist.at<float>(i, 0)));
			g_cdf.push_back((g_norm_hist.at<float>(i, 0)));
			b_cdf.push_back((b_norm_hist.at<float>(i, 0)));*/
			r_cdf[i] = r_norm_hist.at<float>(i, 0);
			g_cdf[i] = g_norm_hist.at<float>(i, 0);
			b_cdf[i] = b_norm_hist.at<float>(i, 0);

		}
		else
		{
			/*r_cdf.push_back((r_norm_hist.at<float>(i, 0) + r_cdf[i - 1]));
			g_cdf.push_back((g_norm_hist.at<float>(i, 0) + g_cdf[i - 1]));
			b_cdf.push_back((b_norm_hist.at<float>(i, 0) + b_cdf[i - 1]));*/

			r_cdf[i] = r_norm_hist.at<float>(i, 0) + r_cdf[i - 1];
			g_cdf[i] = g_norm_hist.at<float>(i, 0) + g_cdf[i - 1];
			b_cdf[i] = b_norm_hist.at<float>(i, 0) + b_cdf[i - 1];
		}

	}




	// Get the min and max pixel values indicating the 5% of pixels at the ends of the histograms
	float r_min_val = -1, g_min_val = -1, b_min_val = -1; // Dummy value
	float r_max_val = -1, g_max_val = -1, b_max_val = -1; // Dummy value

	for (int i = 0; i < 256; i++)
	{
		if (r_cdf[i] >= perc && r_min_val == -1)
		{
			r_min_val = (float)i;
		}

		if (r_cdf[i] >= (1.0 - perc) && r_max_val == -1)
		{
			cout << r_cdf[i] << endl;
			r_max_val = (float)i;
		}

		if (g_cdf[i] >= perc && g_min_val == -1)
		{
			g_min_val = (float)i;
		}

		if (g_cdf[i] >= (1.0 - perc) && g_max_val == -1)
		{
			g_max_val = (float)i;
		}

		if (b_cdf[i] >= perc && b_min_val == -1)
		{
			b_min_val = (float)i;
		}

		if (b_cdf[i] >= (1.0 - perc) && b_max_val == -1)
		{
			b_max_val = (float)i;
		}
	}


	// Build look up table
	unsigned char r_lut[256], g_lut[256], b_lut[256];
	for (int i = 0; i < 256; i++)
	{
		float index = (float)i;
		r_lut[i] = saturate_cast<uchar>(255.0 * ((index - r_min_val) / (r_max_val - r_min_val)));
		g_lut[i] = saturate_cast<uchar>(255.0 * ((index - g_min_val) / (g_max_val - g_min_val)));
		b_lut[i] = saturate_cast<uchar>(255.0 * ((index - b_min_val) / (b_max_val - b_min_val)));
	}



	Mat out = img.clone();
	MatIterator_<Vec3b> it, end;
	for (it = out.begin<Vec3b>(), end = out.end<Vec3b>(); it != end; it++)
	{

		(*it)[2] = r_lut[(*it)[2]];
		(*it)[1] = g_lut[(*it)[1]];
		(*it)[0] = b_lut[(*it)[0]];
	}


	return out;

}

float mean_pixel(Mat img)
{
	if (img.channels() > 2)
	{
		cvtColor(img.clone(), img, CV_BGR2GRAY);
		return mean(img)[0];
	}
	else
	{
		return mean(img)[0];
	}
}

float auto_gamma_value(Mat img)
{
	float max_pixel = 255;
	float middle_pixel = 128;
	float pixel_range = 256;
	float mean_l = mean_pixel(img);

	float gamma = log(middle_pixel / pixel_range) / log(mean_l / pixel_range); // Formula from ImageJ

	return gamma;

}


Mat gamma_correction(Mat img, float gamma = 0)
{
	CV_Assert(img.data);

	// Accept only char type matrices
	CV_Assert(img.depth() != sizeof(uchar));


	if (gamma == 0) { gamma = auto_gamma_value(img); }


	// Build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
	}

	Mat out = img.clone();
	const int channels = out.channels();
	switch (channels)
	{
	case 1:
	{
		MatIterator_<uchar> it, end;
		for (it = out.begin<uchar>(), end = out.end<uchar>(); it != end; it++)
			*it = lut[(*it)];

		break;
	}
	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = out.begin<Vec3b>(), end = out.end<Vec3b>(); it != end; it++)
		{

			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}

	return out;
}


/*
	Huang, S. C., Cheng, F. C., & Chiu, Y. S. (2013).
	Efficient contrast enhancement using adaptive gamma correction with weighting distribution.
	IEEE Transactions on Image Processing, 22(3), 1032-1041.
*/
Mat adaptive_gamma_correction(Mat img, float alpha = 0)
{

	CV_Assert(img.data);

	// Accept only char type matrices
	CV_Assert(img.depth() != sizeof(uchar));

	// Automatically compute the alpha value
	if (alpha == 0) { alpha = auto_gamma_value(img); }

	// Get the image probability density function using the histogram
	Mat gray_img;
	if (img.channels() > 2)
	{
		cvtColor(img.clone(), gray_img, CV_BGR2GRAY);
	}


	// Establish the number of bins
	int histSize = 256;

	// Set the range
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;

	// Compute the histogram
	calcHist(&gray_img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	Mat norm_hist = hist / gray_img.total();
	double pdf_min = 0;
	double pdf_max = 0;
	minMaxLoc(norm_hist, &pdf_min, &pdf_max);

	std::vector<float> pdf_weights;
	for (int i = 0; i < 256; i++)
	{
		pdf_weights.push_back(pdf_max * (pow((norm_hist.at<float>(i, 0) - pdf_min) / (pdf_max - pdf_min), alpha)));
	}

	std::vector<float> cdf_weights;
	float pdf_weights_total = sum(pdf_weights)[0];
	for (int i = 0; i < 256; i++)
	{
		if (i == 0)
		{
			cdf_weights.push_back((pdf_weights[i]));
		}
		else
		{
			cdf_weights.push_back((pdf_weights[i] + cdf_weights[i - 1]));
		}

	}

	// Build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		float gamma = 1 - cdf_weights[i];
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
	}

	Mat out = img.clone();
	const int num_channels = out.channels();
	switch (num_channels)
	{
	case 1:
	{
		MatIterator_<uchar> it, end;
		for (it = out.begin<uchar>(), end = out.end<uchar>(); it != end; it++)
			*it = lut[(*it)];

		break;
	}
	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = out.begin<Vec3b>(), end = out.end<Vec3b>(); it != end; it++)
		{

			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}

	return out;

}

void SimplestCB(Mat& in, Mat& out, float percent) {
	assert(in.channels() == 3);
	assert(percent > 0 && percent < 100);

	float half_percent = percent / 200.0f;

	vector<Mat> tmpsplit; split(in, tmpsplit);
	for (int i = 0; i < 3; i++) {
		//find the low and high precentile values (based on the input percentile)
		Mat flat; tmpsplit[i].reshape(1, 1).copyTo(flat);
		cv::sort(flat, flat, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
		cout << lowval << " " << highval << endl;

		//saturate below the low percentile and above the high percentile
		tmpsplit[i].setTo(lowval, tmpsplit[i] < lowval);
		tmpsplit[i].setTo(highval, tmpsplit[i] > highval);

		//scale the channel
		normalize(tmpsplit[i], tmpsplit[i], 0, 255, NORM_MINMAX);
	}
	merge(tmpsplit, out);
}



void balance_white(cv::Mat mat) {
	double discard_ratio = 0.05;
	int hists[3][256];
	memset(hists, 0, 3 * 256 * sizeof(int));

	for (int y = 0; y < mat.rows; ++y) {
		uchar* ptr = mat.ptr<uchar>(y);
		for (int x = 0; x < mat.cols; ++x) {
			for (int j = 0; j < 3; ++j) {
				hists[j][ptr[x * 3 + j]] += 1;
			}
		}
	}

	// cumulative hist
	int total = mat.cols*mat.rows;
	int vmin[3], vmax[3];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 255; ++j) {
			hists[i][j + 1] += hists[i][j];
		}
		vmin[i] = 0;
		vmax[i] = 255;
		while (hists[i][vmin[i]] < discard_ratio * total)
			vmin[i] += 1;
		while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
			vmax[i] -= 1;
		if (vmax[i] < 255 - 1)
			vmax[i] += 1;
	}


	for (int y = 0; y < mat.rows; ++y) {
		uchar* ptr = mat.ptr<uchar>(y);
		for (int x = 0; x < mat.cols; ++x) {
			for (int j = 0; j < 3; ++j) {
				int val = ptr[x * 3 + j];
				if (val < vmin[j])
					val = vmin[j];
				if (val > vmax[j])
					val = vmax[j];
				ptr[x * 3 + j] = static_cast<uchar>((vmax[j] == vmin[j] ? val : (val - vmin[j]) * 255.0 / (vmax[j] - vmin[j])));
			}
		}
	}
}



#endif



int kGolf(Mat* inputArray, Mat* inputArrayBlue, int num, int& impactIdx, int& backIdx, int& followIdx)
{

	string cfg = ".\\yolo.cfg";
	string weight = ".\\yolo.weights";
	kGolfPose kPose(cfg, weight);
	//end = clock(); //시간 측정 끝
	//double resultTime = (double)(end - start);
	//printf("kGolfPose : %f\n", resultTime);

	kPose.initResult();

	//for (int m = 0; m < tempFile.size(); m++)
	//{
		//clock_t start, end;
		//start = clock();

		//bool rewr = kPose.kGolfPreProcessing(inputImgPool[m]);

		//end = clock(); //시간 측정 끝
		//double resultTime = (double)(end - start);
		//printf("dnn : %f\n", resultTime);



		//Mat rer = inputImgPool[m].clone();
		//cv::rectangle(rer, kPose.m_detectedPersonRect, Scalar(0, 0, 0), 2);
		//cv::imshow("rer", rer);
		//cvWaitKey(1);
	//}
	int firstImgIdx = 65;
	int pushIdx = 0;
	bool rewr = kPose.kGolfPreProcessing(inputArray[pushIdx]);

	if (!rewr)
	{
		cout << "Person detection error !" << endl;
		return 0;
	}

	Mat inputBlueInit = inputArrayBlue[pushIdx].clone();


	//int wcc = 0;



	int ff = 1;
	for (int ii = ff; ii < num; ii++)
	{
		clock_t start, end;
		start = clock();

		// ii :image index
		kPose.kGolfPoseDetection(inputArrayBlue[ii - 1], inputArrayBlue[ii], ii);
		//kPose.kGolfPoseDetection(imread(tempFile[ii - 1].c_str(), 0), imread(tempFile[ii].c_str(), 0), ii);


		cout << ii << " / " << num << endl;

		//printf("%d, %f %% \n", wcc, (((float)ii) / (float)tempFile.size()) * 100);
		//cv::waitKey(1);
		//if (ii == tempFile.size()-1)
		//{
		//	kPose.deletekGolfPose();
		//	kPose.init();
		//	ii = ff;
		//	wcc++;
		//}

		end = clock(); //시간 측정 끝
		double resultTime = (double)(end - start);
		printf("kGolfPose : %f\n", resultTime);

		//if (ii == tempFile.size()-1)
		//{
		//	kPose.deletekGolfPose();
		//	kPose.initResult();
		//	ii = 1;
		//}

		cvWaitKey(0);

	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//sorting for the index
	//kPose.sortIdx();

	//kPose.findBallPosition(m_ballPosition);
	bool ck = kPose.findBallPosition();

	if (ck)
	{
		Mat BallROIFinal = inputBlueInit.clone();
		rectangle(BallROIFinal, kPose.m_finalBallROI, Scalar(0, 0, 0), 2, 8, 0);
		cv::imwrite("C:\\Users\\KijungKim\\Google 드라이브\\골프과제\\golfBall_v4_8_1_paper(4)_New_20240607_23h29m14s\\final_ball_position.jpg", BallROIFinal);

		// inputBlueInit : init image with an only ball
		// impact, follow and back indice ==> sorted indice 
		int impactIdxR = kPose.estimationImpactSide();
		cout << "impactIdx = " << impactIdxR << endl;
		if (impactIdxR > 0)
		{
			//Mat inputGray = kPose.m_results[impactIdx].curBlueImg;
			//Mat inputGray = imread(tempFile[impactIdx].c_str(), 1);
			//Mat inputGray = InputArray[];
			//cv::resize(inputGray, inputGray, Size(inputGray.cols*0.3, inputGray.rows*0.3), 1, 1, CV_INTER_LANCZOS4);
			//cv::imshow("impact gray", inputGray);
			//moveWindow("Blob", 1000, 50);
			//cv::waitKey(1);

			cv::imwrite(".\\impact.jpg", inputArray[impactIdxR]);
			impactIdx = impactIdxR;

			int backIdxR = kPose.estimationTop(impactIdxR);

			cout << "backIdx = " << backIdxR << endl;
			if (backIdxR > 0)
			{
				//Mat inputGray = kPose.m_results[backIdx].curBlueImg;
				//Mat inputGray = imread(tempFile[backIdx].c_str(), 1);
				//cv::resize(inputGray, inputGray, Size(inputGray.cols*0.3, inputGray.rows*0.3), 1, 1, CV_INTER_LANCZOS4);
				//cv::imshow("back gray", inputGray);
				//cv::waitKey(1);
				cv::imwrite(".\\back.jpg", inputArray[backIdxR]);
				backIdx = backIdxR;
			}
			else
				return 0;


			int followIdxR = kPose.estimationFollow(impactIdx);
			cout << "followIdx = " << followIdxR << endl;
			if (followIdxR > 0)
			{
				//Mat inputGray = kPose.m_results[followIdx].curBlueImg;
				//Mat inputGray = imread(tempFile[followIdx].c_str(), 1);
				//cv::resize(inputGray, inputGray, Size(inputGray.cols*0.3, inputGray.rows*0.3), 1, 1, CV_INTER_LANCZOS4);
				//cv::imshow("follow gray", inputGray);
				//cv::waitKey(1);
				cv::imwrite(".\\follow.jpg", inputArray[followIdxR]);
				followIdx = followIdxR;
			}
			else
				return 0;
		}
		else
			return 0;
	}
	else
		return 0;

	return 1;

}

//void SplitFilename(const std::string& str)
//{
//	//std::cout << "Splitting: " << str << '\n';
//	std::size_t found = str.find_last_of("/\\");
//	std::cout << " path: " << str.substr(0, found) << '\n';
//	std::cout << " file: " << str.substr(found + 1) << '\n';
//}

void splitImageToPatches(const Mat& image, int rows, int cols, vector<Mat>& patches, vector<Rect>& patchesRect) {
	// 패치의 크기 계산
	int patchWidth = image.cols / cols;
	int patchHeight = image.rows / rows;

	// 이미지 패치를 분할하여 벡터에 저장
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			Rect roi(c * patchWidth, r * patchHeight, patchWidth, patchHeight);
			patchesRect.push_back(roi);
			patches.push_back(image(roi));
		}
	}
}
#define BLOCK 5
double blurDetection(Mat frame)
{

	int cx = frame.cols / 2;
	int cy = frame.rows / 2;

	// Go float
	Mat fImage;
	frame.convertTo(fImage, CV_32F);

	// FFT
	cout << "Direct transform...\n";
	Mat fourierTransform;
	dft(fImage, fourierTransform, DFT_SCALE | DFT_COMPLEX_OUTPUT);

	//center low frequencies in the middle
	//by shuffling the quadrants.
	Mat q0(fourierTransform, Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
	Mat q1(fourierTransform, Rect(cx, 0, cx, cy));      // Top-Right
	Mat q2(fourierTransform, Rect(0, cy, cx, cy));      // Bottom-Left
	Mat q3(fourierTransform, Rect(cx, cy, cx, cy));     // Bottom-Right

	Mat tmp;                                            // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                                     // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// Block the low frequencies
	// #define BLOCK could also be a argument on the command line of course
	fourierTransform(Rect(cx - BLOCK, cy - BLOCK, 2 * BLOCK, 2 * BLOCK)).setTo(0);

	//shuffle the quadrants to their original position
	Mat orgFFT;
	fourierTransform.copyTo(orgFFT);
	Mat p0(orgFFT, Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
	Mat p1(orgFFT, Rect(cx, 0, cx, cy));      // Top-Right
	Mat p2(orgFFT, Rect(0, cy, cx, cy));      // Bottom-Left
	Mat p3(orgFFT, Rect(cx, cy, cx, cy));     // Bottom-Right

	p0.copyTo(tmp);
	p3.copyTo(p0);
	tmp.copyTo(p3);

	p1.copyTo(tmp);                                     // swap quadrant (Top-Right with Bottom-Left)
	p2.copyTo(p1);
	tmp.copyTo(p2);

	// IFFT
	cout << "Inverse transform...\n";
	Mat invFFT;
	Mat logFFT;
	double minVal, maxVal;

	dft(orgFFT, invFFT, DFT_INVERSE | DFT_REAL_OUTPUT);

	//img_fft = 20*numpy.log(numpy.abs(img_fft))
	invFFT = cv::abs(invFFT);
	cv::minMaxLoc(invFFT, &minVal, &maxVal, NULL, NULL);

	//check for impossible values
	if (maxVal <= 0.0) {
		cerr << "No information, complete black image!\n";
		return 1;
	}

	cv::log(invFFT, logFFT);
	logFFT *= 20;

	//result = numpy.mean(img_fft)
	cv::Scalar result = cv::mean(logFFT);
	//cout << "Result : " << result.val[0] << endl;

	// show if you like
	//Mat finalImage;
	//logFFT.convertTo(finalImage, CV_8U);    // Back to 8-bits
	//imshow("Input", frame);
	//imshow("Result", finalImage);
	//cv::waitKey();
	// end show if you like

	return result.val[0];

}


int ktGolf(const char* inputRGBpath, int& impactIdx, int& backIdx, int& followIdx, int &NewImpact)
{

	string cfg = ".\\yolo.cfg";
	string weight = ".\\yolo.weights";
	kGolfPose kPose(cfg, weight);
	kPose.initResult();


	vector<string> tempFile = loadPath(inputRGBpath, "jpg");
	int totalImgNum = tempFile.size();
	if (totalImgNum < 160 || totalImgNum > 280)
	{
		cout << "out of range (the number of images) " << endl;
		return 0;
	}

	//Mat inputImgPool[m_result_Max];
	//Mat inputImgPoolBlue[m_result_Max];

	std::size_t foundstart = tempFile[0].find_last_of("\\");
	std::size_t foundend = tempFile[0].find_last_of(".");
	string name = tempFile[0].substr(foundstart + 1, 5);
	kPose.m_realFirstIdx = std::stoi(name);




	int ttotal = totalImgNum - 1;
	kPose.totalImgNumMember = ttotal;

	//read and save
#pragma omp parallel for
	for (int u = 1; u < ttotal; u++)
	{
		kPose.m_results[u].curImg = imread(tempFile[u].c_str(), 1);
		cvtColor(kPose.m_results[u].curImg, kPose.m_results[u].curGrayImg, CV_BGR2GRAY);

		

		kPose.m_results[u].idx = u;
		kPose.m_results[u].preGrayImg = imread(tempFile[u - 1].c_str(), 0);
		kPose.m_results[u].fuGrayImg = imread(tempFile[u + 1].c_str(), 0);
		//kPose.m_results[u].curGrayImg = imread(tempFile[u].c_str(), 0);


		Mat tempColorAWBc = kPose.m_results[u].curImg.clone();
		//Mat tempColorAWBp = imread(tempFile[u - 1].c_str(), 1).clone();
		balance_white(tempColorAWBc);
		//balance_white(tempColorAWBp);
		cv::Mat bgrC[3];
		cv::split(tempColorAWBc, bgrC);
		//cv::Mat bgrP[3];
		//cv::split(tempColorAWBp, bgrP);
		
		kPose.m_results[u].curBlueImg = bgrC[0].clone();  //[0]:blue
		kPose.m_results[u].curRedImg = bgrC[2].clone();
		kPose.m_results[u].curGreenImg = bgrC[1].clone();

		//cv::imshow("kPose.m_results[u].curBlueImg", kPose.m_results[u].curBlueImg);
		//cv::imshow("kPose.m_results[u].curRedImg", kPose.m_results[u].curRedImg);
		//cv::imshow("kPose.m_results[u].curGreenImg", kPose.m_results[u].curGreenImg);
		//cvWaitKey(0);
	}

	kPose.m_results[ttotal].curImg = imread(tempFile[ttotal].c_str(), 1).clone();
	cvtColor(kPose.m_results[ttotal].curImg, kPose.m_results[ttotal].curGrayImg, CV_BGR2GRAY);
	kPose.m_results[ttotal].idx = ttotal;
	kPose.m_results[ttotal].preGrayImg = imread(tempFile[ttotal - 1].c_str(), 0);
	//kPose.m_results[u].curGrayImg = imread(tempFile[u].c_str(), 0).clone();


	Mat tempColorAWBcTT = kPose.m_results[ttotal].curImg.clone();
	//Mat tempColorAWBp = imread(tempFile[u - 1].c_str(), 1).clone();
	balance_white(tempColorAWBcTT);
	//balance_white(tempColorAWBp);
	cv::Mat bgrCTT[3];
	cv::split(tempColorAWBcTT, bgrCTT);
	//cv::Mat bgrP[3];
	//cv::split(tempColorAWBp, bgrP);

	kPose.m_results[ttotal].curBlueImg = bgrCTT[0].clone();  //[0]:blue
	kPose.m_results[ttotal].curRedImg = bgrCTT[2].clone();
	kPose.m_results[ttotal].curGreenImg = bgrCTT[1].clone();

	cout << "File Read Done, size = " << tempFile.size() << endl;
	


	


	//int pushIdx = totalImgNum*0.5;  // center of the imagePool

			
	//persion detection , set to m_persionExROI
	if (!kPose.kGolfPreProcessing(10))
	{
		cout << "Person detection error !" << endl;
		return 0;
	}
	//return 0;

	m_dnnBallDetectedCC = 0;
	m_ballPositionMeanPre = Rect(0, 0, 0, 0);
	m_ballPositionSum = Rect(0, 0, 0, 0);
	m_checkBallPoistion = false;

	int ff = 1;


	bool ckj = kPose.calBGS();
	//ckj = kPose.clustering();
	ckj = kPose.clusteringFixed();






	//blob 찾기
//#pragma omp parallel for private (m_ballPositionSum, m_dnnBallDetectedCC, m_ballPositionMeanPre) 
//#pragma omp parallel for
	for (int ii = ff; ii < totalImgNum; ii++)
	{
		
		//kPose.kGolfPoseDetection(inputImgPoolBlue[ii - 1], inputImgPoolBlue[ii], ii);
		kPose.kGolfPoseDetectionIdx(ii);
		//cout << ii << "   check 3" << endl;
		
		//cvWaitKey(0);
	}
	cout << "kGolfPoseDetectionIdx Done" << endl;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef DebugJJ
	cout << "end for" << endl;
#endif

	//blob voting 방법 적용 ==> ball 찾기 
	bool ck = false;
	if (!m_checkBallPoistion)
		ck = kPose.findBallPosition();
	else
		ck = true;

	cout << "findBallPosition Done" << endl;
	
	impactIdx = 0;
	
	//ball을 찾았으면...
	if (ck)
	{
		Mat BallROIFinal = imread(tempFile[100].c_str(), 1).clone();
		//rectangle(BallROIFinal, kPose.m_finalBallROI, Scalar(0, 0, 255), 2, 8, 0);
		//
		//std::size_t foundstartball = tempFile[0].find("sq");
		//std::size_t foundstartballs = tempFile[0].find("\\", foundstartball);
		//std::size_t foundstartballe = tempFile[0].find("\\", foundstartball + 3);
		//int sixx = foundstartballe - foundstartballs - 1;
		//string nameT = tempFile[0].substr(foundstartballs + 1, sixx);
		//string fin = ".\\ball\\" + nameT + ".bmp";
		//
		//cv::imwrite(fin, BallROIFinal);
		//nameT.clear();
		//fin.clear();

		//Mat BallROIFinal = imread(tempFile[pushIdx-20].c_str(), 1).clone();
		rectangle(BallROIFinal, kPose.m_finalBallROI, Scalar(0, 0, 255), 2, 8, 0);
		cv::imwrite(".\\final_ball_position.jpg", BallROIFinal);
		//cv::resize(BallROIFinal, BallROIFinal, Size(BallROIFinal.cols*0.5, BallROIFinal.rows*0.5), 1, 1, CV_INTER_LANCZOS4);
		//cv::imshow("BallROIFinal", BallROIFinal);
		//cvWaitKey(1);
		// inputBlueInit : init image with an only ball
		// impact, follow and back indice ==> sorted indice 
		int impactIdxR = kPose.estimationImpactSide();
		cout << "estimationImpactSide Done impactIdxR = " << impactIdxR << endl;
		
		int fineIdxImpact = 0;

		if (impactIdxR <= 0 || impactIdxR > 400)
			return -1;
		else
		{
			
			//Rect m_ballRegionEx = kPose.m_ballRegion;
			//ballRegionEx.y = ballRegionEx.y - ballRegionEx.height;
			//ballRegionEx.height = ballRegionEx.height * 2;
			int st = kPose.m_impactIdx - 4;
			int en = kPose.m_impactIdx + 4;
			int lowY = 0;
			
			for (int ik = st; ik < en; ik++)
			{
				if (kPose.m_results[ik].endP.y > lowY)
				{
					lowY = kPose.m_results[ik].endP.y;
					fineIdxImpact = ik;
				}		
				cv::imshow("kPose.curBGSImg[i]", kPose.m_results[ik].curBGSImg);
				cv::imshow("kPose.m_curGrayBlurROIImg[i]", kPose.m_results[ik].m_curGrayBlurROIImg);
				cout << "kPose.m_results[i].endP.y = " << kPose.m_results[ik].endP.y << endl;
				cout << "lowY = " << lowY << endl;
				cout << "i = " << ik << "fineIdxImpact  ---------------- ::: " << fineIdxImpact << endl;
				cvWaitKey(0);
			}
		}
		//다르면 +1 같으면 그대로, impactIdxR == fineIdxImpact
		cout << "impactIdxR = " << impactIdxR << ", fineIdxImpact = " << fineIdxImpact << endl;
		if (fineIdxImpact > 0)
		{
			if (fineIdxImpact != impactIdxR)
			{
				kPose.m_impactIdxNew = fineIdxImpact + kPose.m_realFirstIdx + 1;
				NewImpact = kPose.m_impactIdxNew;
				impactIdx = kPose.m_impactIdxNew;
			}
			else
			{
				kPose.m_impactIdxNew = impactIdxR + kPose.m_realFirstIdx;
				NewImpact = kPose.m_impactIdxNew;
				impactIdx = kPose.m_impactIdxNew;
			}
		}
		else
		{
			if (impactIdxR > 10)
			{
				kPose.m_impactIdxNew = impactIdxR + kPose.m_realFirstIdx;
				impactIdx = kPose.m_impactIdxNew;
				NewImpact = impactIdxR + kPose.m_realFirstIdx;
			}
			else
			{
				kPose.m_impactIdxNew = 0;
				impactIdx = impactIdxR;
				NewImpact = kPose.m_impactIdxNew;
			}
			

		}
		



		cout << "impactIdx New ----------------:: = " << kPose.m_impactIdxNew << endl;
		//impactIdx = kPose.m_impactIdxNew;
		cout << "kPose.m_realFirstIdx = " << kPose.m_realFirstIdx << endl;
		cout << "impactIdx = " << impactIdx << endl;






		

//blur detection
#if 0
		/// blure detection ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		Rect ballRegionEx = kPose.m_ballRegion;
		ballRegionEx.y = ballRegionEx.y - ballRegionEx.height;
		ballRegionEx.height = ballRegionEx.height * 2;
		int st = kPose.m_impactIdx - 5;
		int en = kPose.m_impactIdx + 5;
		for (int ik = st; ik < en; ik++)
		{
			kPose.m_results[ik].curImgROI = imread(tempFile[ik].c_str(), 0)(ballRegionEx).clone();

			vector<Mat> patches;
			vector<Rect> patchesRect;
			splitImageToPatches(kPose.m_results[ik].curImgROI, 8, 6, patches, patchesRect);

			imshow("kPose.m_results[ik].curImgROI", kPose.m_results[ik].curImgROI);

			for (int i = 0; i < 48; i++)
			{
				double ss = blurDetection(patches[i]);
				cout << i << " , blur score = " << ss << endl;
				Mat ccc = kPose.m_results[ik].curImgROI.clone();
				rectangle(ccc, patchesRect[i], Scalar(0, 0, 255));
				imshow("ccc", ccc);
				waitKey(0);
			}


			patches.~vector<Mat>();
			patchesRect.~vector<Rect>();
			waitKey(0);

		}
		/// blure detection ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

		//else
		//{
		//	//check fine impact idx
		//	int st = kPose.m_impactIdx - 5;
		//	int en = kPose.m_impactIdx;
		//	int fineIdxImpact = 0;
		//	int lowY = 0;
		//	for (int i = st; i < en; i++)
		//	{


		//		if (kPose.m_results[i].endP.y > lowY)
		//		{
		//			lowY = kPose.m_results[i].endP.y;
		//			fineIdxImpact = i;
		//		}

		//		cv::imshow("kPose.curBGSImg[i]", kPose.m_results[i].curBGSImg);
		//		cv::imshow("kPose.m_curGrayBlurROIImg[i]", kPose.m_results[i].m_curGrayBlurROIImg);
		//		cout << "kPose.m_results[i].endP.y = " << kPose.m_results[i].endP.y << endl;
		//		cout << "lowY = " << lowY << endl;
		//		cout << "i = " << i << "fineIdxImpact  ---------------- ::: " << fineIdxImpact << endl;
		//		cvWaitKey(0);

		//	}

		//	if (fineIdxImpact>0)
		//		kPose.m_impactIdxNew = fineIdxImpact + kPose.m_realFirstIdx + 1;
		//	else
		//		kPose.m_impactIdxNew = 0;
		//	
		//	cout << "impactIdx New ----------------:: = " << kPose.m_impactIdxNew << endl;
		//	

		//}
		
#if 1
		if (impactIdxR > 0)
		{
			
			//cv::imwrite(".\\impact.jpg", kPose.m_results[impactIdxR].curImg);
			
			//impactIdx = impactIdxR + kPose.m_realFirstIdx;
			
#ifdef DebugJJ
			cout << "impactIdx = " << impactIdx << endl;
			//cv::imwrite(".\\impact.jpg", kPose.m_results[impactIdxR].curImg);
			//cv::imshow("impact", kPose.m_results[impactIdxR].curImg);
			Mat tempimpact = kPose.m_results[impactIdxR].curImg.clone();
			cv::resize(tempimpact, tempimpact, Size(tempimpact.cols*0.5, tempimpact.rows*0.5), 1, 1, CV_INTER_LANCZOS4);
			cv::imshow("impact", tempimpact);
			cv::waitKey(1);
#endif

			int backIdxR = kPose.estimationTop(impactIdxR);
			if (backIdxR > 0)
			{
				
				//cv::imwrite(".\\back.jpg", kPose.m_results[backIdxR].curImg);
				

				backIdx = backIdxR + kPose.m_realFirstIdx;
				cout << "backIdx = " << backIdx << endl;

#ifdef DebugJJ
				//cv::imwrite(".\\back.jpg", kPose.m_results[backIdxR].curImg);
				//cv::imshow("back", kPose.m_results[backIdxR].curImg);
				Mat tempback = kPose.m_results[backIdxR].curImg.clone();
				cv::resize(tempback, tempback, Size(tempback.cols*0.5, tempback.rows*0.5), 1, 1, CV_INTER_LANCZOS4);
				cv::imshow("back", tempback);
				cv::waitKey(1);
#endif

			}
			else
				return 0;

			int followIdxR = kPose.estimationFollow(impactIdxR);
			if (followIdxR > 0)
			{
				//checkFollow
				cv::imwrite(".\\follow.jpg", kPose.m_results[followIdxR].curImg);

				followIdx = followIdxR + kPose.m_realFirstIdx;
				cout << "followIdx = " << followIdx << endl;

#ifdef DebugJJ
				//cv::imwrite(".\\follow.jpg", kPose.m_results[followIdxR].curImg);
				Mat tempfollow = kPose.m_results[followIdxR].curImg.clone();
				cv::resize(tempfollow, tempfollow, Size(tempfollow.cols*0.5, tempfollow.rows*0.5), 1, 1, CV_INTER_LANCZOS4);
				cv::imshow("follow", tempfollow);
				cv::waitKey(1);
#endif

				return 1;
			}
			else
				return 0;
		}
		else
			return 0;
#endif
	}
	else
		return 0;

	
	
}

void ktPersonDetection(Mat input)
{
	string cfg = ".\\yolo.cfg";
	string weight = ".\\yolo.weights";

	Net ktNet = readNetFromDarknet(cfg, weight);
	ktNet.setPreferableBackend(DNN_BACKEND_DEFAULT);
	ktNet.setPreferableTarget(DNN_TARGET_CPU);

	Mat inputBlob = blobFromImage(input, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
	ktNet.setInput(inputBlob, "data");                   //set the network input
	Mat detectionMat = ktNet.forward("detection_out");   //compute output

	float confidenceThreshold = 0.5; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;        // Width of network's input image
	int inpHeight = 416;
	Rect outRec = Rect(Point(0, 0), Point(0, 0));  //max
	//Rect outRec = Rect(Point(0, 0), Point(99990, 99990));   //min
	//Point personCenter = Point(0, 0);
	bool detected = false;

	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

		//0:person
		size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

		if ((objectClass == 0) && (confidence > confidenceThreshold))
		{
			//std::cout << "detectionMat = " << objectClass << "confedence = " << confidence << endl;

			float x_center = detectionMat.at<float>(i, 0) * input.cols;
			float y_center = detectionMat.at<float>(i, 1) * input.rows;
			float width = detectionMat.at<float>(i, 2) * input.cols;
			float height = detectionMat.at<float>(i, 3) * input.rows;
			Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
			Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));

			Rect outRecTemp = Rect(p1, p2);

			//max height
			if (outRecTemp.height > outRec.height)
			{
				outRec = outRecTemp;
				//personCenter.x = outRec.x + (outRec.width*0.5);
				//personCenter.y = outRec.y + (outRec.height*0.5);
			}
			detected = true;
		}
	}

	FILE *fp;
	fopen_s(&fp, ".\\bdp.result", "w");
	fprintf(fp, "{\"x\": %d, \"y\": %d, \"width\": %d, \"height\": %d, \"result\": %d}", outRec.x, outRec.y, outRec.width, outRec.height, detected);
	fclose(fp);

}

void ktGolfMain(const char* inputRGBpath)
{
	
	//first image index = 0 ==>  first image neme index + impactIdx
	int impactIdx = -1;
	int backIdx = -1;
	int followIdx = -1;

	int re = ktGolf(inputRGBpath, impactIdx, backIdx, followIdx, followIdx);



	FILE *fp;
	fopen_s(&fp, ".\\bd.result", "w");

	fprintf(fp, "{\"impact\": %d, \"follow\": %d, \"back\": %d, \"result\": %d}", impactIdx, followIdx, backIdx, re);
	fclose(fp);

//#ifdef DebugJJ
	if (re)
	{
		cout << "Ok" << endl;
		//cout << "impact = " << impactIdx << ", back = " << backIdx << ", follow = " << followIdx << endl;
	}
	else
		cout << "Fail" << endl;
//#endif
	//cv::waitKey(0);
}

void ktGolfMain(string* inputRGBpath)
{
	//for (int i = 0; i < 82; i++)
	for (int i = 70; i < 71; i++)   // 3
	{
		clock_t start, end;
		start = clock();
		//first image index = 0 ==>  first image neme index + impactIdx
		int impactIdx = -1;
		int backIdx = -1;
		int followIdx = -1;
		int newImpact = -1;
		char path[200];
		strcpy_s(path, inputRGBpath[i].c_str());
		cout << path << endl;

		int re = ktGolf(path, impactIdx, backIdx, followIdx, newImpact);

#if 0
		//FILE *fp;
		//fopen_s(&fp, ".\\bd.result.txt", "w");

		//fprintf(fp, "{\"impact\": %d, \"follow\": %d, \"back\": %d, \"result\": %d}", newImpact, followIdx, backIdx, re);
		//fclose(fp);

		FILE *fp;
		fopen_s(&fp, ".\\bd.resultNew_0607.txt", "a");

		fprintf(fp, "%d %d %d \n", backIdx, impactIdx, followIdx);
		fclose(fp);


#endif

		


		//#ifdef DebugJJ
		if (re)
		{
			cout << "Ok" << endl;
			//cout << "impact = " << impactIdx << ", back = " << backIdx << ", follow = " << followIdx << endl;
			cout << "kPose.m_impactIdxNew = " << newImpact << endl;
		}
		else
			cout << "Fail Fail  Fail Fail Fail Fail" << endl;


		end = clock(); //시간 측정 끝
		double resultTime = (double)(end - start);
		printf_s("kGolfPose Time : %f\n", resultTime);
		//cout << "final Impact idx " << m_impactIdx
		cout << " ================================================================== " << endl;
		cv::waitKey(1);
	}
}

int main(void)
{
	string filename[82];
	//ktPersonDetection(imread("C:\\kt\\golfZone\\data\\mat\\field\\p01\\506cbd\\00023.jpg", 1));
	//string test = "00125.jpg";
	//string name = getFileName(test);
	//int nummmm = atoi(test.c_str());

	//const char* filename = "D:\\golfZone\\golfdemo\\front\\b\\gray1\\";
	//const char* filename = "D:\\golfZone\\golfdemo\\front\\h\\gray\\old\\";
	
	//const char* filename = "D:\\golfZone\\golfdemo\\front\\i\\";
	//const char* filename = "D:\\golfZone\\golfdemo\\front\\j_120hz_10pixel\\gray\\";
	//const char* filename = "D:\\golfZone\\golfdemo\\front\\l_120hz_18pixel\\";
	//const char* filename = "D:\\golfZone\\golfdemo\\front\\m_5pixel\\gray\\";
	//const char* filename = "D:\\golfZone\\golfdemo\\front\\GH010163_8steps\\";
	//const char* filename = "D:\\golfZone\\golfdemo\\front\\GH010169_8steps\\";
	//const char* filename = "D:\\golfZone\\golfdemo\\front\\GH010174_9steps\\gray\\";


	//const char* filename = "C:\\kt\\data\\real_a\\gray\\";
	//const char* filename = "C:\\kt\\data\\real_b\\gray\\";
	//const char* filename = "C:\\kt\\data\\real_c\\gray\\";
	//const char* filename = "C:\\kt\\golfZone\\data\\real_c\\";
	//const char* filename = "C:\\kt\\data\\real_c1\\";

	//const char* filename = "C:\\kt\\data\\fixed\\a\\";
	
	//const char* filename = "C:\\kt\\golfZone\\data\\mat\\506cbb\\a\\";
	//const char* filename = "C:\\kt\\golfZone\\data\\mat\\cen-s\\";
	//const char* filename = "C:\\kt\\golfZone\\data\\mat\\a-s\\";
	//const char* filename = "C:\\kt\\golfZone\\data\\mat\\b-s\\";
	//const char* filename = "C:\\kt\\golfZone\\data\\mat\\c-s\\";
	//const char* filename = "C:\\kt\\golfZone\\data\\mat\\d-s\\";
	//00023.jpg

	filename[0]  = "C:\\researches\\golfSwing\\sq\\p01\\506cbd\\";
	filename[1]  = "C:\\researches\\golfSwing\\sq\\p02\\506cbd\\a\\";
	filename[2]  = "C:\\researches\\golfSwing\\sq\\p03\\506cbd\\";
	filename[3]  = "C:\\researches\\golfSwing\\sq\\p04\\506cbd\\";
	filename[4]  = "C:\\researches\\golfSwing\\sq\\p05\\506cbd\\";             //
	filename[5]  = "C:\\researches\\golfSwing\\sq\\p06\\506cbd\\";             //
	filename[6]  = "C:\\researches\\golfSwing\\sq\\p07\\506cbd\\";             //
	filename[7]  = "C:\\researches\\golfSwing\\sq\\p08\\506cbd\\";           
	filename[8]  = "C:\\researches\\golfSwing\\sq\\p09\\506cbd\\";           
	filename[9]  = "C:\\researches\\golfSwing\\sq\\p10\\506cbd\\";           
	filename[10] = "C:\\researches\\golfSwing\\sq\\p11\\506cbd\\";           
	filename[11] = "C:\\researches\\golfSwing\\sq\\p12\\506cbd\\";            // negative
	filename[12] = "C:\\researches\\golfSwing\\sq\\p13\\506cbd\\";            //
	filename[13] = "C:\\researches\\golfSwing\\sq\\p14\\506cbd\\";            //
	filename[14] = "C:\\researches\\golfSwing\\sq\\p15\\506cbd\\";            //
	filename[15] = "C:\\researches\\golfSwing\\sq\\p16\\506cbd\\";            //
	filename[16] = "C:\\researches\\golfSwing\\sq\\p16\\temp_short\\";
	filename[17] = "C:\\researches\\golfSwing\\sq\\p17\\506cbd\\";
	filename[18] = "C:\\researches\\golfSwing\\sq\\p18\\506cbd\\";
	filename[19] = "C:\\researches\\golfSwing\\sq\\p19\\506cbd\\";
	filename[20] = "C:\\researches\\golfSwing\\sq\\p20\\506cbd\\";
	filename[21] = "C:\\researches\\golfSwing\\sq\\p21\\506cbd\\";
	filename[22] = "C:\\researches\\golfSwing\\sq\\p22\\506cbd\\";
	filename[23] = "C:\\researches\\golfSwing\\sq\\p23\\506cbd\\";
	filename[24] = "C:\\researches\\golfSwing\\sq\\p24\\506cbd\\";
	filename[25] = "C:\\researches\\golfSwing\\sq\\p25\\506cbd\\";
	filename[26] = "C:\\researches\\golfSwing\\sq\\p26_nohit\\";
	filename[27] = "C:\\researches\\golfSwing\\sq\\p27_nohit\\";
	filename[28] = "C:\\researches\\golfSwing\\sq\\p28\\";
	filename[29] = "C:\\researches\\golfSwing\\sq\\p29\\";                       //+1
	filename[30] = "C:\\researches\\golfSwing\\sq\\p30_1frame\\";
	filename[31] = "C:\\researches\\golfSwing\\sq\\p31\\";
	filename[32] = "C:\\researches\\golfSwing\\sq\\p32_afterHitDone\\";
	filename[33] = "C:\\researches\\golfSwing\\sq\\p33_occlusion\\";
	filename[34] = "C:\\researches\\golfSwing\\sq\\p34_nohit\\";
	filename[35] = "C:\\researches\\golfSwing\\sq\\p35\\";
	filename[36] = "C:\\researches\\golfSwing\\sq\\p36_missshot_rain\\";
	filename[37] = "C:\\researches\\golfSwing\\sq\\p37_rain\\";
	filename[38] = "C:\\researches\\golfSwing\\sq\\p38_rain\\";
	filename[39] = "C:\\researches\\golfSwing\\sq\\p39_rain\\";
	filename[40] = "C:\\researches\\golfSwing\\sq\\p40_leaf\\";
	filename[41] = "C:\\researches\\golfSwing\\sq\\p41_green\\";
	filename[42] = "C:\\researches\\golfSwing\\sq\\p42_green\\";
	filename[43] = "C:\\researches\\golfSwing\\sq\\p43\\";
	filename[44] = "C:\\researches\\golfSwing\\sq\\p44\\";
	filename[45] = "C:\\researches\\golfSwing\\sq\\p45\\";
	filename[46] = "C:\\researches\\golfSwing\\sq\\p46\\";
	filename[47] = "C:\\researches\\golfSwing\\sq\\p47\\";
	filename[48] = "C:\\researches\\golfSwing\\sq\\p48\\";
	filename[49] = "C:\\researches\\golfSwing\\sq\\p49\\";
	filename[50] = "C:\\researches\\golfSwing\\sq\\p50\\";
	filename[51] = "C:\\researches\\golfSwing\\sq\\p51\\";
	filename[52] = "C:\\researches\\golfSwing\\sq\\p52\\";
	filename[53] = "C:\\researches\\golfSwing\\sq\\p53\\";
	filename[54] = "C:\\researches\\golfSwing\\sq\\p54\\";
	filename[55] = "C:\\researches\\golfSwing\\sq\\p55\\";

	filename[56] = "C:\\researches\\golfSwing\\sq\\M20191014151428299\\";
	filename[57] = "C:\\researches\\golfSwing\\sq\\M20191014151514142\\";
	filename[58] = "C:\\researches\\golfSwing\\sq\\M20191014151548416\\";
	filename[59] = "C:\\researches\\golfSwing\\sq\\M20191014153413805\\";
	filename[60] = "C:\\researches\\golfSwing\\sq\\M20191014153821811\\";
	filename[61] = "C:\\researches\\golfSwing\\sq\\M20191014153937782\\";
	filename[62] = "C:\\researches\\golfSwing\\sq\\M20191014154720596\\";
	filename[63] = "C:\\researches\\golfSwing\\sq\\M20191014154822522\\";
	filename[64] = "C:\\researches\\golfSwing\\sq\\M20191014160641664\\";
	filename[65] = "C:\\researches\\golfSwing\\sq\\M20191014160740950\\";
	filename[66] = "C:\\researches\\golfSwing\\sq\\M20191015153416278\\";
	filename[67] = "C:\\researches\\golfSwing\\sq\\M20191015164256784\\";
	filename[68] = "C:\\researches\\golfSwing\\sq\\M20191015164340091\\";
	filename[69] = "C:\\researches\\golfSwing\\sq\\M20191015164414988\\";
	filename[70] = "C:\\researches\\golfSwing\\sq\\M20191015164450204\\";
	filename[71] = "C:\\researches\\golfSwing\\sq\\M20191015164526538\\";
	filename[72] = "C:\\researches\\golfSwing\\sq\\M20191015175249355\\";
	filename[73] = "C:\\researches\\golfSwing\\sq\\M20191015175326709\\";
	filename[74] = "C:\\researches\\golfSwing\\sq\\M20191016104951261\\";
	filename[75] = "C:\\researches\\golfSwing\\sq\\M20191016105024734\\";
	filename[76] = "C:\\researches\\golfSwing\\sq\\M20191016122611875\\";
	filename[77] = "C:\\researches\\golfSwing\\sq\\M20191016122654746\\";
	filename[78] = "C:\\researches\\golfSwing\\sq\\M20191016122725247\\";
	filename[79] = "C:\\researches\\golfSwing\\sq\\M20191016155658385\\";
	filename[80] = "C:\\researches\\golfSwing\\sq\\M20191016155822876\\";
	filename[81] = "C:\\researches\\golfSwing\\sq\\M20191016162104097\\";
	
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p01\\506cbd\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p02\\506cbd\\a\\";   //tee에 악세사리 white X  , small GS 
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p03\\506cbd\\";   //red  XX  tee occlusion
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p04\\506cbd\\";   //red  up Ball

	// 2 person
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p05\\506cbd\\";  //yellow  left
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p06\\506cbd\\";  //white



	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p07\\506cbd\\";  //white  right
	// 2 person
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p08\\506cbd\\";    //white left  

	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p09\\506cbd\\";    //white  (+ 1 frame) , bgs and flow energy 상호 확인 ok
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p10\\506cbd\\";    //white
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p11\\506cbd\\";    //white

	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p12\\506cbd\\";   //yellow  //no impact   negative
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p13\\506cbd\\";   //white  +1
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p14\\506cbd\\";   //white center
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p15\\506cbd\\";   //white
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p16\\506cbd\\";   //  //red  +1 frame
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p16\\temp_short\\";   //white  //red  //짧은 영상 217 frames ,2개 scene 섞은것
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p17\\506cbd\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p18\\506cbd\\";
	const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p19\\506cbd\\";    //yellow  , 골프채에 ball roi 잡힘.
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p20\\506cbd\\";       //짧음, 164장  negative
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p21\\506cbd\\";             //ball 윗쪽
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p22\\506cbd\\";         ////짧음, 200장  2개 scene 섞음
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p23\\506cbd\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p24\\506cbd\\";         // white   +1 frame
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p25\\506cbd\\";


	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p26_nohit\\";   //negative
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p27_nohit\\";     //negative
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p28\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p29\\";        //+1
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p30_1frame\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p31\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p32_afterHitDone\\";   //negative
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p33_occlusion\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p34_nohit\\";   //negative
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p35\\";

	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p36_missshot_rain\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p37_rain\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p38_rain\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p39_rain\\";

	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p40_leaf\\";        //공 근처 나뭇잎이 공 비슷, 권 상무님

	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p41_green\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p42_green\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p43\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\p55\\";

	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\a_temp\\M20191010174145155\\";
	//const char* filenames = "C:\\kt\\golfZone\\data\\mat\\field\\M20191014154720596\\";


#if 0

	vector<string> tempFile = loadPath(filename, "jpg");

#if 0
	char buf[500];
	int index = 0;
	for (int ww = 0; ww < tempFile.size(); ww++)
	{
		Mat gray = imread(tempFile[ww].c_str(), 0);
		sprintf_s(buf, "C:\\kt\\data\\real_c1\\gray\\%05d.jpg", index);
		imwrite(buf, gray);
		index++;
		cout << ww << " / " << tempFile.size() << endl;
	}
#endif

#if 0
	cv::Mat bgr[3];
	Mat rr, gg, bb;
	Mat inputTemp = imread(tempFile[0].c_str(), 1);
	cv::split(inputTemp, bgr);
	rr = bgr[2];
	gg = bgr[1];
	bb = bgr[0];

	Mat sfdssobel = (rr) + (bb);
	Mat ggray = imread(tempFile[0].c_str(), 0);

	Mat out;

	


	//cv::imshow("Red Channel", bgr[2]);  // caution : the red channel index is 2!
	//cv::imshow("Green Channel", bgr[1]);
	//cv::imshow("Blue Channel", bgr[0]);
	//cvWaitKey(0);
	//Mat ycbcr;
	//cv::cvtColor(inputTemp, ycbcr, CV_BGR2HSV);
	//cv::Mat hsv[3];
	//Mat hh, ss, vv;
	//cv::split(inputTemp, hsv);
	//vv = hsv[2];
	//ss = hsv[1];
	//yy = hsv[0];
#endif

#pragma omp parallel for
	for (int u = 0; u < tempFile.size(); u++)
	{
		Mat tempColor = imread(tempFile[u].c_str(), 1);
		Mat tempColorAWB = tempColor.clone();
		balance_white(tempColorAWB);
		inputImgPool[u] = tempColor.clone();
		//Mat out, outc;
	//	out = white_balance(tempColor.clone());
	//	outc = adaptive_gamma_correction(tempColor.clone(), 0.005);

		//Mat BW_out = tempColor.clone();
		//balance_white(BW_out);
		//outc = adaptive_gamma_correction(BW_out.clone());
		cv::Mat bgrw[3];
		cv::split(tempColor, bgrw);
		inputImgPoolBlue[u] = bgrw[0].clone();

		//cv::Mat bgr[3];
		//Mat rr, gg, bb;
		//Mat inputTemp = imread(tempFile[0].c_str(), 1);
		//cv::split(tempColor, bgr);
		//inputImgPool[u] = bgr[0].clone();
		//Mat tempII = bgr[0].clone();

		//Mat SimplestCB_Out;
		//SimplestCB(tempColor, SimplestCB_Out, 1);

		
		//int jkl = 0;

	//	imshow("tsetst", inputImgPool[u]);
		//cvWaitKey(0);
		//inputImgPool[u] = imread(tempFile[u].c_str(), 0);
		cout << u << " / " << tempFile.size() << endl;
	}

	//imshow("tsetst", inputImgPool[3]);
	//cvWaitKey(0);
	//define
	//bool m_firstCheckROI = false;
	//Rect m_detectedPersonRect;
	//Rect setLeftRegion, setRightRegion, m_ballRegion, m_upRegion, m_flowRegion;
	//Mat m_inputBlueC, m_inputBlueP;
	//person detection

	//test();
	int impactIdx = -1;
	int backIdx = -1;
	int followIdx = -1;

	int re = kGolf(inputImgPool, inputImgPoolBlue, tempFile.size(), impactIdx, backIdx, followIdx);
#endif

	

	clock_t start, end;
	start = clock();

	//int re = ktGolf(filename, impactIdx, backIdx, followIdx);
	
	
	ktGolfMain(filename);

	end = clock(); //시간 측정 끝
	double resultTime = (double)(end - start);
	printf_s("kGolfPose Time : %f\n", resultTime);

	
	//if (re)
	//{
	//	cout << "OK" << endl;
	//}
	//else
	//	cout << "FAIL" << endl;
#ifdef DebugJJ
	cvWaitKey(0);
#endif

	return 1014;
}




