#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 이미지 load
    Mat img1 = imread("image_1.jpeg");
    Mat img2 = imread("image_2.jpeg");
    
    Mat grayimg1, grayimg2;
    cvtColor(img1, grayimg1, COLOR_RGB2GRAY);
    cvtColor(img2, grayimg2, COLOR_RGB2GRAY);


    // Key point 변수 생성 및 FAST 기반 특징점 추출
    vector<KeyPoint> keypoints1, keypoints2;
    FAST(grayimg1, keypoints1, 70, true);
    FAST(grayimg2, keypoints2, 70, true);

    // ORB 활용 Descriptor 추출
    Ptr<ORB> orb = ORB::create();
    Mat descriptors1, descriptors2;
    orb->compute(grayimg1, keypoints1, descriptors1);
    orb->compute(grayimg2, keypoints2, descriptors2);

    // BFMatcher 활용 특징점 매칭
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    // drawMatches 함수
    // Mat drawing;
    // drawMatches(img1, keypoints1, img2, keypoints2, matches, drawing);
    // imshow("matches", drawing);
    // imshow("s", grayimg1);

    // 대응점 집합 계산
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < matches.size(); i++) {
        pts1.push_back(keypoints1[matches[i].queryIdx].pt);
        pts2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    
    // Homography 활용 transform 계산 및 RANSAC 활용 Outlier 제거
    Mat H = findHomography(pts2, pts1, RANSAC);

    // image 합성
    Mat result;
    warpPerspective(img2, result, H, Size(img2.cols * 2, img2.rows));
    Mat half(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);
    
    // 결과 출력
    imshow("Result", result);
    waitKey(0);

    return 0;
}
