// ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// OpenCV dep
#include <opencv2/cvconfig.h>

#include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>

using namespace sl;
using namespace std;
using namespace cv;

/**
 * @brief Method type to find the center of ROI in the image for auto ajust of Exposure/Gain
 */
enum class findCenter{
    CONTOURS = 1 ,
    MOMENTS = 2
};


#define refreshRate 1 
#define GRID true

#define focus_center_threshold 150


cv::Mat slMat2cvMat(sl::Mat &input); 
int getOCVtype(sl::MAT_TYPE type);

/**
 * @brief Draw a grid in dst with dimension n x n
 * @param src src image
 * @param dst  dst image
 * @param n size of the grid n*n
 * @param width src image width
 * @param height src image height
 */
void grid(cv::Mat src, cv::Mat &dst, int n, int width, int height);
/**
 * @brief Find the contours of the image and calculate the central point, based on the median of all centers for each contour, 
 * of the window for ROI in ZED AEC_AGC
 * @param src_gray grayscale src image
 * @param contour return image with all contours highlighted
 * @param thresh threshold for the edge detector -> Canny's algorithm 
 * @param center return of the median center 
 */
void thresh_callback_contours(cv::Mat src_gray, cv::Mat &contour, int thresh, cv::Point2f &center);
/**
 * @brief Find the moment of the image and calculate the central point (moment center) of the window for ROI in ZED AEC_AGC
 * @param src_gray grayscale src image
 * @param contour return image with all moments highlighted
 * @param thresh threshold for the edge detector -> Canny's algorithm 
 * @param center return of the image moment center (based only in the main frame)
 */
void thresh_callback_moments(cv::Mat src_gray, cv::Mat &contour, int thresh, cv::Point2f &center);

/**
 * @brief Returns x and y vector ordered based os the vector<Point2f> array
 * @param array Input Point2f array to be ordered
 * @param x return x vector ordered
 * @param y return y vector ordered
 */
void sortXY(vector<cv::Point2f> array, vector<float> &x, vector<float> &y);
/**
 * @brief Median of a data set
 * @param data 
 * @return float 
 */
float median(vector<float> data);
/**
 * @brief Calculate the histogram of the src image
 * @param src src image
 * @param dst histogram for imshow()
 */
void plotHistogram(cv::Mat src, cv::Mat &dst);

/**
 * @brief Depending of the type, find the pixel that will be the center of the ROI to focus 
 * @param src src image
 * @param focus_center return center point
 * @param grayscale return grayscale image
 * @param blur return blur image
 * @param histogram return histogram of RBG channels
 * @param contour return contour image with the ROI highlighted
 * @param type findCenter type MOMENTS or CONTOUR
 * @return sl::Rect 
 */
sl::Rect focus_center(cv::Mat src, Point2f &focus_center, cv::Mat &grayscale, cv::Mat &blur, cv::Mat &histogram, cv::Mat &contour, findCenter type);
/**
 * @brief adjust the window size in case of the window is overlap the edges of the image
 * @param src src image
 * @param max_size wanted size for the square (could be smaller)
 * @param center focus center
 * @return int 
 */
int adjust_focus_window(cv::Mat src, int max_size, Point center);