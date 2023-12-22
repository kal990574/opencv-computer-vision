#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    // Step 0: 두 장의 이미지 촬영
    Mat image1 = imread("image1.jpeg");
    Mat image2 = imread("image2.jpeg");

    // Step 1: 특징점 추출 및 매칭
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    Ptr<DescriptorExtractor> descriptor = BRISK::create();

    std::vector<KeyPoint> kp1, kp2;
    Mat des1, des2;

    detector->detect(image1, kp1);
    detector->detect(image2, kp2);

    descriptor->compute(image1, kp1, des1);
    descriptor->compute(image2, kp2, des2);

    BFMatcher matcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    matcher.match(des1, des2, matches);

    // Step 2: 대응점 집합 계산
    std::sort(matches.begin(), matches.end());
    std::vector<DMatch> good_matches(matches.begin(), matches.begin() + static_cast<int>(matches.size() * 0.25));

    // Step 3: 이미지 간의 transform 계산 (Homography)
    std::vector<Point2f> pts1, pts2;

    for (const auto& match : good_matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    Mat H = findHomography(pts1, pts2, RANSAC);

    // Step 4: 이미지 이어붙이기
    Mat result;
    warpPerspective(image2, result, H, Size(image1.cols + image2.cols, image2.rows));
    image1.copyTo(result(Rect(0, 0, image1.cols, image1.rows)));

    // 결과 이미지 저장
    imshow("result", result);
    return 0;
}
