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

#define LIMIT_SMOOTH_COUNT 7 //³]©w­nsmooth´X­Óframe
#endif

#define length 512

using namespace dlib;
using namespace cv;
using namespace std;

CvPoint SmoothROI(CvPoint2D32f newPt);

#define CHKRGN(pos) pos<0?0:pos

Mat img;

int main(int argc, const char** argv)
{
	try //¦pªG¨S§ä¨ì½ü¹ø¤£·|·í¾÷
	{
		double fps;
		char string[10];
		double times = 0;

		int MRoiX = 0; //ÃBÀYRoi¤¤¤ß (x, )
		int MRoiY = 0; //ÃBÀYRoi¤¤¤ß ( ,y)

		int RRoiX = 0; //¥kÁyRoi¤¤¤ß (x, )
		int RRoiY = 0; //¥kÁyRoi¤¤¤ß ( ,y)

		int LRoiX = 0; //¥ªÁyRoi¤¤¤ß (x, )
		int LRoiY = 0; //¥ªÁyRoi¤¤¤ß ( ,y)

		CvPoint avgPtM;

		VideoCapture cap(0);

		int totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 640); //³]©w¼e
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480); //³]©w°ª

		// ¸ü¤J¾Ç²ßÀÉ
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		bool _isInit = false;
//		Mat _Pos;

		Mat _PPos;
		Rect _Pos;

		bool paused = false;

		while (true)
		{
			times = (double)getTickCount();

				cap >> img;

				//Mat img_roi(320, 430, img.type());
				//int nChannels = img.channels();
				//int nRows = img.rows;
				//int nCols = img.cols;

				//for (int j = 60; j < nRows - 170; j++) {
				//	uchar* frameData = img.ptr<uchar>(j);
				//	uchar* roiData = img_roi.ptr<uchar>(j);
				//	for (int i = 220; i < nCols - 220; i++) {
				//		roiData[nChannels*i + 2] = frameData[nChannels*i + 2];
				//		roiData[nChannels*i + 1] = frameData[nChannels*i + 1];
				//		roiData[nChannels*i + 0] = frameData[nChannels*i + 0];
				//	}
				//}
				cv_image<bgr_pixel> cimg;

				if (_isInit == false)
				{
					//µ¹©w¾ã±i¼v¹³
					cimg = cv_image<bgr_pixel>(img);

					//°»´ú¤HÁy
					std::vector<dlib::rectangle> faces = detector(cimg);

					// Find the pose of each face.
					full_object_detection shape;
					for (unsigned short i = 0; i < faces.size(); ++i)
					{
						shape = pose_model(cimg, faces[i]);
						//SHAPE ¥Nªí§ä¨ìªº¤HÁy¦ì¸m¸ê°T

						point pt39 = shape.part(39); //¤º²´¨¤
						point pt42 = shape.part(42); //¤º²´¨¤
						point pt33 = shape.part(33); //»óÀY

						int CenterX = (pt42.x() + pt39.x()) ; //¦L°ó
						int CenterY = (pt42.y() + pt39.y()) ; //¦L°ó

						MRoiX = CenterX - pt33.x();
						MRoiY = CenterY - pt33.y();

						//===========
						CvPoint2D32f newPtM = Point(MRoiX, MRoiY);
						avgPtM = SmoothROI(newPtM);

						if (avgPtM.x == 0 && avgPtM.y == 0) continue;

					}

					//------±N _Pos ©ñ¤J¤HÁy¦ì¸m¸ê°T (¥ª¤Wx,y , ¤¤¤ßµ¥)
//					_PPos = img(Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250));
					_Pos = Rect(CHKRGN(avgPtM.x - 100)+ _Pos.x, CHKRGN(avgPtM.y - 80)+ _Pos.y, 200, 250);

					_isInit = true;
				}
				else
				{
					//-------±N ¾ã±i¼v¹³(img) ³]©w¦bROI(_Pos+?????)ªº½d³ò
					//-------±N³]©w¦nROIªºimg copy ¨ì roiImg¤º

					//Mat test(_Pos.rows, _Pos.cols, img.type());
					//int nChannels = _Pos.channels();
					//int nRows = _Pos.rows;
					//int nCols = _Pos.cols;

					//for (int j = 0; j < nRows; j++) {
					//	uchar* srcData = _Pos.ptr<uchar>(j);
					//	uchar* dstData = test.ptr<uchar>(j);
					//	for (int i = 0; i < nCols; i++) {
					//		*dstData++ = *srcData++;
					//	}
					//}

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

						point pt39 = shape.part(39); //¤º²´¨¤
						point pt42 = shape.part(42); //¤º²´¨¤
						point pt33 = shape.part(33); //»óÀY

						//ÀYÂà¨¤«×§PÂ_
						point pt2 = shape.part(2); //¥kÁyù¯°©
						point pt36 = shape.part(36); //¥k¥~²´¨¤
						point pt14 = shape.part(14); //¥ªÁyù¯°©
						point pt45 = shape.part(45); //¥ª¥~²´¨¤
						point pt30 = shape.part(30); //»ó¦y

						int CenterX = (pt42.x() + pt39.x()) ; //¦L°ó
						int CenterY = (pt42.y() + pt39.y()) ; //¦L°ó

						MRoiX = CenterX - pt33.x();
						MRoiY = CenterY - pt33.y();

						//===========
						CvPoint2D32f newPtM = Point(MRoiX, MRoiY);
						avgPtM = SmoothROI(newPtM);

						if (avgPtM.x == 0 && avgPtM.y == 0) continue;

						if (pt14.x() - pt30.x() <= 40)
						{
							CvPoint2D32f newPtR = Point(RRoiX, RRoiY);
							CvPoint avgPtR = SmoothROI(newPtR);
							Rect R_region_of_interest = Rect(CHKRGN(avgPtR.x - 5), CHKRGN(avgPtR.y - 5), 10, 10);
							cv::rectangle(img, R_region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}
						else if (pt30.x() - pt2.x() <= 40)
						{
							CvPoint2D32f newPtL = Point(LRoiX, LRoiY + 70);
							CvPoint avgPtL = SmoothROI(newPtL);
							Rect L_region_of_interest = Rect(CHKRGN(avgPtL.x - 5), CHKRGN(avgPtL.y - 5), 10, 10);
							cv::rectangle(img, L_region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}
						else
						{
							Rect region_of_interest = Rect(CHKRGN(avgPtM.x - 5), CHKRGN(avgPtM.y - 5), 10, 10);
							cv::rectangle(img, region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}

						times = ((double)getTickCount() - times) / getTickFrequency();
						fps = 1.0 / times;
						sprintf(string, "%.2f", fps);
						std::string fpsString("FPS:");
						fpsString += string;
						putText(img, fpsString, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
					}
					//±N _Pos ©ñ¤J¤HÁy¦ì¸m¸ê°T (¥ª¤Wx,y , ¤¤¤ßµ¥)¡A¦ý­n°O±o¦A¦ì¸m¤W°µshift(¤HÁy¦ì¸m¥[¤W _Pos.LeftTop Position)

					_Pos = Rect(CHKRGN(avgPtM.x - 100)+_Pos.x, CHKRGN(avgPtM.y - 80)+_Pos.y, 200, 250);


					//_Pos = img(Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250));
				}
					//imshow("1", img);
				
//				imshow("123", img_roi);

			imshow("Demo", img);

			char c = (char)waitKey(10);
			if (c == 27)
				break;

			fix(c, paused);
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
