## OpenCV 활용 이미지 연결 기능 구현

### 1. Fast 알고리즘 및 ORB를 이용한 특징점 추출
Fast 알고리즘과 ORB(Oriented FAST and Rotated BRIEF)를 사용하여 이미지에서 특징점을 추출합니다. Fast 알고리즘은 빠르고 효율적인 코너 검출 알고리즘이며, ORB는 이를 기반으로 회전 불변성 및 스케일 불변성을 가진 특징점 디스크립터를 생성합니다.

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

int main() {
    // 이미지 로드
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    // ORB 특징점 검출기 생성
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // 특징점 검출 및 기술자 계산
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    // 결과 시각화
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);
    cv::imshow("ORB Keypoints", output);
    cv::waitKey(0);

    return 0;
}
```

### 2. BFMatcher와 vector를 활용하여 특징점 매칭 및 계산
BFMatcher(Brute Force Matcher)를 사용하여 ORB 디스크립터를 기반으로 특징점을 매칭합니다. `std::vector`를 사용하여 매칭된 결과를 저장하고, 매칭 품질을 평가합니다.

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

int main() {
    // 이미지 로드
    cv::Mat image1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Images not found!" << std::endl;
        return -1;
    }

    // ORB 특징점 검출기 생성
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // BFMatcher 생성 및 특징점 매칭
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 매칭 결과 시각화
    cv::Mat output;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, output);
    cv::imshow("Matches", output);
    cv::waitKey(0);

    return 0;
}
```

### 3. Homography와 RANSAC 알고리즘을 통해 이미지 변환 계산
Homography와 RANSAC(Random Sample Consensus) 알고리즘을 사용하여 두 이미지 간의 변환 행렬을 계산합니다. 이 알고리즘은 두 이미지 간의 매칭된 특징점을 기반으로 변환 행렬을 추정하여 이미지의 기하학적 변형을 보정합니다.

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

int main() {
    // 이미지 로드
    cv::Mat image1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Images not found!" << std::endl;
        return -1;
    }

    // ORB 특징점 검출기 생성
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // BFMatcher 생성 및 특징점 매칭
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 매칭된 포인트 추출
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Homography 계산 및 RANSAC 적용
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);

    // 결과 시각화
    cv::Mat result;
    cv::warpPerspective(image1, result, H, image2.size());
    cv::imshow("Warped Image", result);
    cv::waitKey(0);

    return 0;
}
```
