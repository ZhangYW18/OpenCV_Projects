//OpenCV 4.0.0 Alpha
#include <core/core.hpp>
#include <opencv.hpp>
#include <highgui.hpp>
#include <imgproc/imgproc.hpp>

using namespace cv;

int main() {
	Mat mio = imread("F:/faceimage.jpg"),fltdMio;
	imshow("mio", mio);
	waitKey(1000);
	blur(mio, fltdMio, Size(3, 3));
	imshow("Filtered Mio", fltdMio);
	imwrite("F:/FltFaceimage.jpg", fltdMio);
	waitKey(0);
	return 0;
}