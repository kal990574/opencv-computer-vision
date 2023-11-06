#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    Mat image = imread("/Users/mambasa/Desktop/picture/coins0.jpg");

    Mat gray_image;
    Mat blur_image;
    Mat canny_image;
    vector<Vec3f> result;
    int cnt =0;
    
    namedWindow("Image");
    imshow("Image", image);
    
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    // Gaussian blur param 조작, 9x9 고정, sigma 값
    GaussianBlur(gray_image, blur_image, Size(7, 7), 1.6, 1.6);
    // thresholding param 조작 최소값, 최대값
    Canny(blur_image, canny_image, 60, 150);
    // Circle param 조작 dp, mindist, 높은 임계값, 크기가 작을수록 잘못된 원 더 감지, 최소원반경, 최대원반경
    HoughCircles(canny_image, result, HOUGH_GRADIENT, 1.2, 50, 150, 40, 20, 80);

    for (auto& c : result) {
        circle(image, Point(c[0], c[1]), c[2], Scalar(255, 0, 255), 3, LINE_AA);
        cnt++;
    }
    
    cout << cnt;
    namedWindow("Gray");
    imshow("Gray", gray_image);
    namedWindow("Blur");
    imshow("Blur", blur_image);
    namedWindow("Canny");
    imshow("Canny", canny_image);
    namedWindow("Circles");
    imshow("Circles", image);

    waitKey(0);
}
