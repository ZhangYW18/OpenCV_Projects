//OpenCV 4.0.0 Alpha
#include <core/core.hpp>
#include <opencv.hpp>
#include <highgui.hpp>
#include <imgproc/imgproc.hpp>

using namespace cv;

int main() {
	namedWindow("test opencv setup",WINDOW_AUTOSIZE);
	Mat mio = imread("F:/20150609191222_awxGC.jpg"),greyMio;
	imshow("mio", mio);
	waitKey(1000);
	cvtColor(mio,greyMio,COLOR_BGR2GRAY);
	Mat equMio;
	equalizeHist(greyMio, equMio);
	imshow("equalized Mio", equMio);
	imwrite("F:/equMio.jpg", equMio);
	waitKey(0);
	return 0;
}