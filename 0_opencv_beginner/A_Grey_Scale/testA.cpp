//OpenCV 4.0.0 Alpha
#include <core/core.hpp>
#include <opencv.hpp>
#include <highgui.hpp>

using namespace cv;

int main() {
	namedWindow("test opencv setup",WINDOW_AUTOSIZE);
	Mat mio = imread("F:/20150609191222_awxGC.jpg"),greyMio;
	imshow("test opencv setup", mio);
	waitKey(1000);
	cvtColor(mio,greyMio,COLOR_BGR2GRAY);
	imshow("test opencv setup", greyMio);
	imwrite("F:/greyMio.jpg", greyMio);
	waitKey(0);
	return 0;
}