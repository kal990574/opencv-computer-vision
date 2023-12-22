#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 이미지 load
    Mat img1 = imread("image1.jpeg", IMREAD_COLOR);
    Mat img2 = imread("image2.jpeg", IMREAD_COLOR);

    // Key point 변수 생성 및 FAST 기반 특징점 추출
    vector<KeyPoint> keypoints1, keypoints2;
    FAST(img1, keypoints1, 30, true);
    FAST(img2, keypoints2, 30, true);

    // ORB 활용 Descriptor 추출
    Ptr<ORB> orb = ORB::create();
    Mat descriptors1, descriptors2;
    orb->compute(img1, keypoints1, descriptors1);
    orb->compute(img2, keypoints2, descriptors2);

    // BFMatcher 활용 특징점 매칭
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    // drawMatches 함수
    Mat drawing;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, drawing);
    imshow("matches", drawing);

    // 대응점 집합 계산
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < matches.size(); i++) {
        pts1.push_back(keypoints1[matches[i].queryIdx].pt);
        pts2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    
    // Homography 활용 transform 계산 및 RANSAC 활용 Outlier 제거
    Mat H = findHomography(pts1, pts2, RANSAC);

    // image 합성
    Mat result;
    warpPerspective(img2, result, H, Size(2*img2.cols, img2.rows), INTER_CUBIC);
    Mat mam;
    mam = result.clone();
    Mat half(mam, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);

    // 결과 출력
    imshow("Result", result);
    waitKey(0);

    return 0;
}
