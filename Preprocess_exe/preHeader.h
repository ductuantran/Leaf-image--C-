#include <stdio.h>
#include <conio.h>
#include <io.h>
#include <stdio.h>  
#include <Windows.h>
#include <direct.h>
#include "dirent.h"
#include <time.h>
#include <fstream>
#include <math.h>
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

//structure for a frame
typedef struct frame
{
	int right; int left; int up; int down;
};

//segmention by Watershed
class WatershedSegmenter{                   
private:
    Mat markers;
public:
    void setMarkers(Mat& markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }
	
    Mat process(Mat &image)
    {
        watershed(image, markers);
        markers.convertTo(markers,CV_8U);
        return markers;
    }
};

//apply a mask on an original image, keep the region inside the mask, change the region outside into white
//Input  : Mar ori  - original image
//         Mat mask - mask
//Output : Mat      - image after masking
Mat segment_img (Mat ori, Mat mask)                       
{
	Mat final; int i, j;
	final = ori.clone();
	for (i=0; i<final.rows; i++)
		for (j=0; j<final.cols; j++)
			{
				if ((int)mask.at<UCHAR>(i,j)!=255) 
					{ final.at<Vec3b>(i,j)[0] = ori.at<Vec3b>(i,j)[0]; 
				      final.at<Vec3b>(i,j)[1] = ori.at<Vec3b>(i,j)[1]; 
					  final.at<Vec3b>(i,j)[2] = ori.at<Vec3b>(i,j)[2];}
				else
					{ final.at<Vec3b>(i,j)[0] = 255; 
				      final.at<Vec3b>(i,j)[1] = 255; 
					  final.at<Vec3b>(i,j)[2] = 255;}
			}
	return final;
}

//find the outer frame of the image
//Input  : Mat img - input image
//Output : frame   - outer border of the image
frame find_outframe (Mat img)              
{
	frame outfr;
	outfr.down = img.rows-1; outfr.up = 0; outfr.left = 0; outfr.right = img.cols-1;
	int cnt1, cnt2;
	long int count=0;
	double sum;
	
	for (cnt1=0; cnt1 < img.rows; cnt1++)
		{
			sum = 0; count =0;
			for (cnt2=0; cnt2 < img.cols; cnt2++)
				{ if ((int)img.at<UCHAR>(cnt1,cnt2)==255) count++;}
			if (count>(int)(0.9*img.cols)) { outfr.up = cnt1; break; }
			if ((cnt1==2*img.rows/3)&&(count<=(int)(0.9*img.cols))) { outfr.up = 0; break; }
		}
	count = 0;
	for (cnt1=img.rows-1; cnt1 >= 0; cnt1--)
		{
			sum = 0; count =0;
			for (cnt2=0; cnt2 < img.cols; cnt2++)
				{if ((int)img.at<UCHAR>(cnt1,cnt2)==255) count++;}
			if (count>(int)(0.96*img.cols)) { outfr.down = cnt1; break; }
			if ((cnt1==2*(img.rows/3))&&(count<=(int)(0.96*img.cols))) { outfr.down = img.rows-1; break; }
		}
	count = 0;
	for (cnt1=0; cnt1 < img.cols; cnt1++)
		{
			sum = 0; count =0;
			for (cnt2=0; cnt2 < img.rows; cnt2++)
				{if ((int)img.at<UCHAR>(cnt2,cnt1)==255) count++;}
			if (count>(int)(0.96*img.rows)) { outfr.left = cnt1; break; }
			if ((cnt1==2*img.cols/3)&&(count<=(int)(0.96*img.cols))) { outfr.left = 0; break; }
		}
	count = 0;
	for (cnt1=img.cols-1; cnt1 >= 0; cnt1--)
		{
			sum = 0; count =0;
			for (cnt2=0; cnt2 < img.rows; cnt2++)
				{if ((int)img.at<UCHAR>(cnt2,cnt1)==255) count++;}
			if (count>(int)(0.96*img.rows)) { outfr.right = cnt1; break; }
			if ((cnt1==2*(img.cols/3))&&(count<=(int)(0.96*img.cols))) { outfr.right = img.cols-1; break; }
		}
	return outfr;
}

//find the inner frame of the image
//Input  : Mat img - input image
//Output : frame   - inner border of the image
frame find_inframe (Mat img, frame outfr)   
{
	frame infr;
	infr = outfr;
	int cnt1, cnt2;
	bool sign;
	for (cnt1 = outfr.up; cnt1 <= outfr.down; cnt1++)
		{
			sign=false;
			int cntu=0;
			for (cnt2 = outfr.left; cnt2 < outfr.right; cnt2++)
				if ((int)img.at<UCHAR>(cnt1,cnt2)==255) cntu++;
			if (cntu<0.96*(outfr.right-outfr.left)) { infr.up = cnt1-20; break;}
		}
	for (cnt1 = outfr.down; cnt1 >= outfr.up; cnt1--)
		{
			sign=false;
			for (cnt2 = outfr.left; cnt2 < outfr.right; cnt2++)
				if ((int)img.at<UCHAR>(cnt1,cnt2)!=255) { sign=true; break;}
			if (sign==true) { infr.down = cnt1; break;}
		}
	for (cnt1 = outfr.left; cnt1 <= outfr.right; cnt1++)
		{
			sign=false;
			int cntl = 0;
			for (cnt2 = outfr.up; cnt2 < outfr.down; cnt2++)
				if ((int)img.at<UCHAR>(cnt2,cnt1)==255) cntl++;
		    if (cntl<0.96*(outfr.down-outfr.up)) { infr.left = cnt1; break;}
		}
	for (cnt1 = outfr.right; cnt1 >= outfr.left; cnt1--)
		{
			sign=false;
			for (cnt2 = outfr.up; cnt2 < outfr.down; cnt2++)
				if ((int)img.at<UCHAR>(cnt2,cnt1)!=255) { sign=true; break;}
		    if (sign==true) { infr.right = cnt1; break;}
		}

	if (infr.down+30<img.rows) infr.down = infr.down+30;
	else if (infr.down+15<img.rows) infr.down = infr.down+15;
	else infr.down=infr.down;

	if (infr.up>15) infr.up = infr.up-15;
	else if (infr.up>5) infr.up = infr.up-5;
	else infr.up=infr.up;

	if (infr.left>15) infr.left = infr.left-15;
	else if (infr.left>5) infr.left = infr.left-5;
	else infr.left=infr.left;
	
	if (infr.right+15<img.cols) infr.right = infr.right+15;
	else if (infr.right+5<img.cols) infr.right = infr.right+5;
	else infr.right=infr.right;

	if (infr.left<=outfr.left)   infr.left  = outfr.left+2;
	if (infr.right>=outfr.right) infr.right = outfr.right-2;
	if (infr.up<=outfr.up)       infr.up    = outfr.up+2;
	if (infr.down>=outfr.down)   infr.down  = outfr.down-2;

	return infr;
}

//cut image by a frame
//Input  : Mat img    - input image
//         frame infr - frame
//Output : Mat        - output image
Mat cut_image (Mat img, frame infr)                             
{
	Mat newimg(infr.down-infr.up+1,infr.right-infr.left+1,CV_8UC3);
	int cnt1, cnt2;
	for (cnt1=infr.up; cnt1<=infr.down; cnt1++)
		{
			for (cnt2=infr.left; cnt2<=infr.right; cnt2++)
			{
					newimg.at<Vec3b>(cnt1-infr.up,cnt2-infr.left)[0] = (int)img.at<Vec3b>(cnt1,cnt2)[0];
					newimg.at<Vec3b>(cnt1-infr.up,cnt2-infr.left)[1] = (int)img.at<Vec3b>(cnt1,cnt2)[1];
					newimg.at<Vec3b>(cnt1-infr.up,cnt2-infr.left)[2] = (int)img.at<Vec3b>(cnt1,cnt2)[2];
			}
		}
	return newimg;
}

//cut the petioles of the leaf
//Input  : Mat img_ori - original image
//Output : Mat         - image after petiole cutting
Mat cut_tag (Mat img_ori)     
{
	int cnt1, cnt2;
	long int sum;
	frame fr;
	Mat img = img_ori.clone();
	
	cvtColor(img, img, CV_BGR2GRAY);
	threshold(img, img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	
	for (cnt1 = img.rows-1; cnt1 >= 0; cnt1--)
		{
			sum=0;
			for (cnt2 = 0; cnt2 < img.cols; cnt2++)
				{   if (img.at<UCHAR>(cnt1,cnt2)==255) sum++;}
			if (sum<0.8*img.cols) { fr.down = cnt1 + 15; if (fr.down>img.rows) fr.down=img.rows-1; break; }
			if ((cnt1==0)) fr.down = img.rows -1 ;
		}
	for (cnt1 = 0; cnt1 < img.rows; cnt1++)
		{
			sum=0;
			for (cnt2 = 0; cnt2 < img.cols; cnt2++)
				{ if ((int)img.at<UCHAR>(cnt1,cnt2)==255) sum++;}
			if (sum<0.96*img.cols) 
				{ if (cnt1>20) fr.up = cnt1 - 20;
				  else fr.up = cnt1;
					  break; }
			
			if ((cnt1==(img.rows-1))) fr.up = 1;
		}
	fr.left = 1; fr.right = img.cols-1;

	if (fr.down+20<img.rows) fr.down = fr.down + 20;
	//fr.down = fr.down -20;
	if (fr.up>20) fr.up = fr.up - 20;
	//printf("\n up=%d, down=%d, right=%d, left=%d",fr.up,fr.down,fr.right,fr.left);
	//printf("\n Kich thuoc anh (cat cuong) = %d x %d",img.rows,img.cols);
	//if (fr.up-10>0) fr.up = fr.up - 10;
	Mat newimg = cut_image(img_ori,fr);
	return newimg;
}

//rotate leaf in image from its dominant direction to vertical one
//Input  : char* imgpath - image path
//Output : Mat           - output image after rotation
Mat img_rotate (char *imgpath)                
{
	IplImage* src = cvLoadImage(imgpath, 1);   
	IplImage* dst = cvCloneImage( src );
	IplImage* gray;
	double m00, m20, m02, m11, m01, m10;
	CvMoments moments;
	int height,width,step,channels;
	uchar *data;
	CvScalar s;
	int delta = 1;
	double factor;
	gray = cvCreateImage(cvSize(src->width, src->height), src->depth,1);
	cvCvtColor(src, gray, CV_RGB2GRAY);
	height = gray->height; width = gray->width; step = gray->widthStep; channels = gray->nChannels; data = (uchar*)gray->imageData;

	// invert the image
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
			for(int k=0;k<channels;k++)
				if (data[i*step+j*channels+k]<200)
					data[i*step+j*channels+k]=255;
				else data[i*step+j*channels+k]=0;
			    if ((src)&&(gray))
						{
							cvMoments(gray, &moments,0);					 
							m00 = cvGetSpatialMoment(&moments, 0,0);
							m01 = cvGetSpatialMoment(&moments, 0,1);
							m10 = cvGetSpatialMoment(&moments, 1,0);
							m20 = cvGetSpatialMoment(&moments, 2,0);
							m02 = cvGetSpatialMoment(&moments, 0,2);
							m11 = cvGetSpatialMoment(&moments, 1,1);		 
						}
				double angle;
				angle = 180/CV_PI*atan(2*(m11*m00-m01*m10)/((m20-m02)*m00-(m10*m10-m01*m01)));
				if (angle>90)
						{
							angle=180-angle;
						}
				angle=angle/2;
				cout << " , angle = " << angle;
						
				float m[6];
	            int w = src->width; int h = src->height;  
				//factor = (cos(angle*CV_PI/180.) + 1.05) * 2;
				factor = 1;
	
				m[0] = (float)(factor*cos(-angle*CV_PI/180.));
				m[1] = (float)(factor*sin(-angle*CV_PI/180.));
				m[3] = -m[1];
				m[4] = m[0];
				CvMat M = cvMat(2, 3, CV_32F, m);
				m[2] = m10/m00;  
				m[5] = m01/m00;
				cvGetQuadrangleSubPix(src, dst, &M);
				Mat rotated(dst);
				//threshold(rotated, rotated, 0, 255, THRESH_BINARY_INV); 
	return rotated;
}

//detect contours of a grayscale image
//Input  : Mat img - input image
//Output : Mat     - output image with contours
Mat fContours(Mat img)  
{
		int thresh=100;
		RNG rng(12345);
		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		// Detect edges using canny
		Canny(img,canny_output,thresh,thresh*2,3);
		// Find contours
		findContours(canny_output,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0, 0));

		// Draw contours
		Mat drawing = Mat::zeros(canny_output.size(),CV_8UC3);
		for(int i = 0; i< contours.size(); i++)
			{
				//Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
				Scalar color = Scalar(255,255,255);
				drawContours(drawing,contours,i,color,2,8,hierarchy,0,Point());
			}
		return drawing;
}

//preprocess image by all the previous processings
//Input  : char* imgpath     - path of original image
//         char* saveimgpath - path where saving output
//Output : void
void preprocess(char* imgpath, char* saveimgpath)
{
	Mat ori = imread(imgpath,1);
	Mat image = ori;
	Mat binary;
	cvtColor(image, binary, CV_BGR2GRAY);
	Mat ct=fContours(binary);
	threshold(binary, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
									
	//foreground detection
	Mat fg;
	erode(binary,fg,Mat(),Point(-1,-1),1);

	//background detection
	Mat bg;
	dilate(binary,bg,Mat(),Point(-1,-1),1);
	threshold(bg,bg,1, 128,THRESH_BINARY);
	Mat markers(binary.size(),CV_8U,Scalar(0));
	markers= fg+bg;

	//segmentation by watershed
	WatershedSegmenter segmenter;
	segmenter.setMarkers(markers);
	Mat result = segmenter.process(image);
	result.convertTo(result,CV_8U);
	threshold(result, result, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Mat result1 = segment_img(image, result);
	frame out = find_outframe(result);
	frame in = find_inframe(result,out);
	Mat segmented;
	if (result1.empty()==0) 
		segmented = cut_image(result1,out); 
	else segmented = ori;
	imwrite("segmented.jpg",segmented);
	
	//petiole cutting
	Mat cutimg;	
	if (segmented.empty()==0) 
		cutimg = cut_tag(segmented);
	else cutimg = ori;
	imwrite("cut.jpg",cutimg);
	
	//orientation normalization
	if (cutimg.empty()==0) 
		imwrite(saveimgpath,cutimg);
	else imwrite(saveimgpath,ori);
	Sleep(500);
	Mat rotated = img_rotate(saveimgpath);
	imwrite("rotated.jpg",rotated);
									
	//sobel filter						
	cvtColor(rotated,rotated,CV_RGB2GRAY);
	medianBlur(rotated,rotated,3);									
	Mat sobelx, sobely, absx, absy, sobelxy;
	Sobel(rotated,sobelx,CV_16S,1,0,3,1,0,BORDER_DEFAULT);
	convertScaleAbs(sobelx,absx);
	Sobel(rotated,sobely,CV_16S,0,1,3,1,0,BORDER_DEFAULT);
	convertScaleAbs(sobely,absy);
	addWeighted(absx,0.5,absy,0.5,0,sobelxy);
	imwrite(saveimgpath,sobelxy);
}