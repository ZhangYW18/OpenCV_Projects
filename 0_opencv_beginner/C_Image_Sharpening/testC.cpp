//OpenCV 4.0.0 Alpha
#include <core/core.hpp>
#include <opencv.hpp>
#include <highgui.hpp>
#include <imgproc/imgproc.hpp>

using namespace cv;

int main() {
	Mat mio = imread("F:/20150609191222_awxGC.jpg"),fltMio;
	imshow("mio", mio);
	waitKey(1000);
	Mat kernel(3, 3, CV_32F, Scalar(-1));
	kernel.at<float>(1, 1) = 9.0;
	filter2D(mio, fltMio, mio.depth(), kernel);
	imshow("Filtered Mio", fltMio);
	imwrite("F:/fltMio.jpg", fltMio);
	waitKey(0);
	return 0;
}