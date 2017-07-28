#include "stdafx.h"  
#include "fftw3.h"
#include "Header.h" 
#include <iostream>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>


#ifndef __SMOOTH_ROI__
#define __SMOOTH_ROI__

#define LIMIT_SMOOTH_COUNT 7 //設定要smooth幾個frame
#endif

#define length 128

using namespace dlib;
using namespace cv;
using namespace std;

CvPoint SmoothROI(CvPoint2D32f newPt);

#define CHKRGN(pos) pos<0?0:pos
float CalculateROIAverage(Mat, int);
void SaveData(void);
void normalize();

float Origin_g[length * 11 + 10], Origin_r[length * 11 + 10]; //原本長度
float MoveAverage[length * 11], MoveAverage_r[length * 11]; //移動平均
							    //float Standard[length];
float Detrended[length * 11], Detrended_r[length * 11]; //Origin - Average

int frameNumber = 0; //目前影格
int record = 0;

Mat img;

int main(int argc, const char** argv)
{
	try //如果沒找到輪廓不會當機
	{
		double fps;
		char string[10];
		double times = 0;

		int MRoiX = 0; //額頭Roi中心 (x, )
		int MRoiY = 0; //額頭Roi中心 ( ,y)

		int RRoiX = 0; //右臉Roi中心 (x, )
		int RRoiY = 0; //右臉Roi中心 ( ,y)

		int LRoiX = 0; //左臉Roi中心 (x, )
		int LRoiY = 0; //左臉Roi中心 ( ,y)

		CvPoint avgPtM;

		VideoCapture cap(0);

		int totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 640); //設定原始鏡頭寬
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480); //設定原始鏡頭高

		// 載入學習檔
		frontal_face_detector detector = get_frontal_face_detecto44144r();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		bool _isInit = false;
		
		Mat _PPos;
		Rect _Pos;

		bool paused = false;

		while (true)
		{
			times = (double)getTickCount();

				cap >> img;

				cv_image<bgr_pixel> cimg;

				if (_isInit == false)
				{
					//給定整張影像
					cimg = cv_image<bgr_pixel>(img);

					//偵測人臉
					std::vector<dlib::rectangle> faces = detector(cimg);

					// Find the pose of each face.
					full_object_detection shape;
					for (unsigned short i = 0; i < faces.size(); ++i)
					{
						shape = pose_model(cimg, faces[i]);
						//SHAPE 代表找到的人臉位置資訊

						point pt39 = shape.part(39); //右內眼角
						point pt42 = shape.part(42); //左內眼角
						point pt36 = shape.part(36); //右外眼角
						point pt36 = shape.part(45); //左外眼角
						
						double EyeL = sqrt((pt42.x()-pt45.x())*(pt42.x()-pt45.x())+(pt42.y()-pt45.y())*(pt42.y()-pt45.y())); //左眼寬
						double EyeR = sqrt((pt36.x()-pt39.x())*(pt36.x()-pt39.x())+(pt36.y()-pt39.y())*(pt36.y()-pt39.y())); //右眼寬
						double EyeAVG = (EyeL + EyeR) /2; //眼睛寬度
						
						point pt27 = shape.part(27); //鼻樑
						point pt33 = shape.part(33); //鼻頭

						double Nose = sqrt((pt27.x()-pt33.x())*(pt37.x()-pt33.x())+(pt27.y()-pt33.y())*(pt27.y()-pt33.y())); //鼻子長
						
						double CenterX = ((pt42.x() + pt39.x())/2 ; //印堂x點
						double CenterY = ((pt42.y() + pt39.y())/2 ; //印堂y點
							       
						//===========
						CvPoint2D32f newPtM = Point(CenterX, CenterY);
						avgPtM = SmoothROI(newPtM);

						if (avgPtM.x == 0 && avgPtM.y == 0) continue;
					}

					//將 _Pos 放入人臉位置資訊 (左上x,y , 中心)
					_Pos = Rect(CHKRGN(avgPtM.x - (EyeAVG)*3.5)+ _Pos.x, CHKRGN(avgPtM.y - Nose*2.5)+ _Pos.y, EyeAVG*7, Nose*5);

					_isInit = true;
				}
				else
				{
					img = img(_Pos);
					Mat roiImg;

					img.copyTo(roiImg);

					cimg = cv_image<bgr_pixel>(roiImg);

					// Detect faces 
					std::vector<dlib::rectangle> faces = detector(cimg);

					// Find the pose of each face.
					full_object_detection shape;
					for (unsigned short i = 0; i < faces.size(); ++i)
					{
						shape = pose_model(cimg, faces[i]);

						point pt39 = shape.part(39); //右內眼角
						point pt42 = shape.part(42); //左內眼角
						point pt36 = shape.part(36); //右外眼角
						point pt36 = shape.part(45); //左外眼角
						
						double EyeL = sqrt((pt42.x()-pt45.x())*(pt42.x()-pt45.x())+(pt42.y()-pt45.y())*(pt42.y()-pt45.y())); //左眼寬
						double EyeR = sqrt((pt36.x()-pt39.x())*(pt36.x()-pt39.x())+(pt36.y()-pt39.y())*(pt36.y()-pt39.y())); //右眼寬
						double EyeAVG = (EyeL + EyeR) /2; //眼睛寬度
						
						point pt27 = shape.part(27); //鼻樑
						point pt33 = shape.part(33); //鼻頭

						double Nose = sqrt((pt27.x()-pt33.x())*(pt37.x()-pt33.x())+(pt27.y()-pt33.y())*(pt27.y()-pt33.y())); //鼻子長
						
						double CenterX = ((pt42.x() + pt39.x())/2 ; //印堂x點
						double CenterY = ((pt42.y() + pt39.y())/2 ; //印堂y點
						
						point pt2 = shape.part(2); //右臉顴骨
						point pt14 = shape.part(14); //左臉顴骨
						point pt30 = shape.part(30); //鼻尖
						
						double FaceR = sqrt((pt2.x()-pt30.x())*(pt2.x()-pt30.x())+(pt2.y()-pt30.y())*(pt2.y()-pt30.y()));//右臉寬度
						double FaceL = sqrt((pt30.x()-pt14.x())*(pt30.x()-pt14.x())+(pt30.y()-pt14.y())*(pt30.y()-pt14.y()));//左臉寬度							  
								  
						//===========
						CvPoint2D32f newPtM = Point(CenterX, CenterY);
						avgPtM = SmoothROI(newPtM);

						if (avgPtM.x == 0 && avgPtM.y == 0) continue;

						if (FaceL <= EyeAVG*3)
						{
							RRoiX = pt36.x();
							RRoiY = pt33.y();
							Rect R_region_of_interest = Rect(RRoiX-5, RRoiY-5, 10, 10);
							for()//抓資料
							{
								
							}
							cv::rectangle(img, R_region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}
						else if (FaceL <= EyeAVG*3)
						{
							RRoiX = pt45.x();
							RRoiY = pt33.y();
							Rect L_region_of_interest = Rect(RRoiX-5, RRoiY-5, 10, 10);
							for()//抓資料
							{
								
							}
							cv::rectangle(img, L_region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}
						else
						{
							MRoiX = Nose; //中間ROI中心x
							MRoiY = CenterY - (Nose/2); //中間ROI中心y
							Rect region_of_interest = Rect(CHKRGN(avgPtM.x - 5), CHKRGN(avgPtM.y - 5), 10, 10);
							for()//抓資料
							{
								
							}
							cv::rectangle(img, region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}

						times = ((double)getTickCount() - times) / getTickFrequency();
						fps = 1.0 / times;
						sprintf(string, "%.2f", fps);
						std::string fpsString("FPS:");
						fpsString += string;
						putText(img, fpsString, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
					}
					//將 _Pos 放入人臉位置資訊 (左上x,y , 中心等)，但要記得再位置上做shift(人臉位置加上 _Pos.LeftTop Position)

					_Pos = Rect(CHKRGN(avgPtM.x - 100)+_Pos.x, CHKRGN(avgPtM.y - 80)+_Pos.y, 200, 250);


					//_Pos = img(Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250));
				}

			imshow("Demo", img);

			char c = (char)waitKey(10);
			if (c == 27)
				break;
		}
	}

	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}

CvPoint SmoothROI(CvPoint2D32f newPt)
{
	static std::queue<CvPoint2D32f> _queuePt;
	static CvPoint2D32f _sumPt = cvPoint2D32f(0.f, 0.f);
	_queuePt.push(newPt);
	_sumPt.x += newPt.x;
	_sumPt.y += newPt.y;
	if (_queuePt.size() <= LIMIT_SMOOTH_COUNT) return cvPoint(0, 0);
	CvPoint avgPt = cvPoint(0, 0);
	CvPoint2D32f firstPt = _queuePt.front();
	_sumPt.x -= firstPt.x;
	_sumPt.y -= firstPt.y;
	_queuePt.pop();
	avgPt.x = (int)(_sumPt.x / LIMIT_SMOOTH_COUNT);
	avgPt.y = (int)(_sumPt.y / LIMIT_SMOOTH_COUNT);
	return avgPt;
}

oid SaveData(void)
{
	fstream  dataFile;
	//開啟.txt檔
	//記錄G
	dataFile.open("data_g.txt", ios::app);
	if (dataFile)
		for (int i = 0; i < frameNumber - 1; i++)
		{
			if (i < length * 11)
				dataFile << Origin_g[i] << "\t" << MoveAverage[i] << "\t" << Detrended[i] << "\n";
			else
				dataFile << Origin_g[i] << "\n";
		}
	dataFile.close();
}
								  

float CalculateROIAverage(Mat roi, int channel)
{
	float avg = 0;

	if (roi.empty()) return 0;
	try
	{
		for (int x = 0; x < 10; x++)
			for (int y = 0; y < 10; y++)
				avg += (int)roi.at<Vec3b>(x, y)[channel];
		avg /= 100;

		return avg;
	}
	catch (exception& e)
	{
		cout << "Calculate Fail" << endl;
	}
}
								  
void normalize(void)
{
	float average = 0, sd = 0;
	for (int i = 0; i < length * 2; i++)
		average += Detrended[i];
	average /= length * 2;

	for (int i = 0; i < length * 2; i++)
		sd += pow(Detrended[i] - average, 2);
	sd /= (length * 2 - 1);
	sd = sqrt(sd);

	for (int i = 0; i < length * 2; i++)
		Detrended[i] = (Detrended[i] - average) / sd;
}
