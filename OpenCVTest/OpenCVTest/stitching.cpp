#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 이미지 load
    Mat img1 = imread("image1.jpeg", IMREAD_GRAYSCALE);
    Mat img2 = imread("image2.jpeg", IMREAD_GRAYSCALE);

    // Key point 변수 생성 및 FAST 기반 특징점 추출
    vector<KeyPoint> keypoints1, keypoints2;
    FAST(img1, keypoints1, 30, true);
    FAST(img2, keypoints2, 30, true);

    // ORB 활용 Descriptor 추출
    Ptr<ORB> orb = ORB::create();
    Mat descriptors1, descriptors2;
    orb->compute(img1, keypoints1, descriptors1);
    orb->compute(img2, keypoints2, descriptors2);

    // 특징점 매칭 (BruteForce Matcher 사용)
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 대응점 집합 계산 (RANSAC 사용하여 outliers 제거)
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < matches.size(); i++) {
        pts1.push_back(keypoints1[matches[i].queryIdx].pt);
        pts2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    Mat mask;
    Mat H = findHomography(pts1, pts2, RANSAC, 5, mask);

    // 이미지 합성 및 변환
    Mat result;
    warpPerspective(img1, result, H, Size(img1.cols + img2.cols, img1.rows));
    Mat half(result, Rect(0, 0, img2.cols, img2.rows));
    img2.copyTo(half);

    // 결과 표시
    imshow("Result", result);
    waitKey(0);

    return 0;
}
