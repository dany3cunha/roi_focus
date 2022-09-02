#include "include/headers.hpp"

cv::RNG rng(12345);

// For testing the size of ROI
uint window_size = 100;
//

cv::Mat slMat2cvMat(sl::Mat &input)
{
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type)
{
    int cv_type = -1;
    switch (type)
    {
    case sl::MAT_TYPE::F32_C1:
        cv_type = CV_32FC1;
        break;
    case sl::MAT_TYPE::F32_C2:
        cv_type = CV_32FC2;
        break;
    case sl::MAT_TYPE::F32_C3:
        cv_type = CV_32FC3;
        break;
    case sl::MAT_TYPE::F32_C4:
        cv_type = CV_32FC4;
        break;
    case sl::MAT_TYPE::U8_C1:
        cv_type = CV_8UC1;
        break;
    case sl::MAT_TYPE::U8_C2:
        cv_type = CV_8UC2;
        break;
    case sl::MAT_TYPE::U8_C3:
        cv_type = CV_8UC3;
        break;
    case sl::MAT_TYPE::U8_C4:
        cv_type = CV_8UC4;
        break;
    default:
        break;
    }
    return cv_type;
}

void grid(cv::Mat src, cv::Mat &dst, int n, int width, int height)
{
    // Grid n x n
    int horizontal_lines = width / n;
    int vertical_lines = height / n;

    // Horizontal
    for (int i = 1; i <= (n / 2) + 2; i++)
    {
        cv::Point pointA(0, vertical_lines * i);
        cv::Point pointB(width - 1, vertical_lines * i);

        line(dst, pointA, pointB, cv::Scalar(255, 0, 0), 3, 8, 0);
    }

    // Vertical
    for (int i = 1; i <= (n / 2) + 2; i++)
    {
        cv::Point pointA(horizontal_lines * i, 0);
        cv::Point pointB(horizontal_lines * i, height - 1);

        line(dst, pointA, pointB, cv::Scalar(255, 0, 0), 3, 8, 0);
    }
}

void thresh_callback_contours(cv::Mat src_gray, cv::Mat &contour, int thresh, cv::Point2f &center)
{
    cv::Mat canny_output;
    cv::Canny(src_gray, canny_output, thresh, thresh * 2);

    vector<vector<cv::Point>> contours;

    findContours(canny_output, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    vector<cv::Rect> boundRect(contours.size());
    vector<vector<cv::Point>> contours_poly(contours.size());
    vector<cv::Point2f> centers(contours.size());
    vector<float> radius(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], contours_poly[i], 5, true);
        boundRect[i] = boundingRect(contours_poly[i]);
        minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
    }

    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC4);

    if (centers.size() > 0)
    {
        vector<float> x, y;

        sortXY(centers, x, y);

        center.x = median(x);
        center.y = median(y);

        cv::circle(drawing, center, 1, cv::Scalar(255, 255, 255), -1);

        // Fazer filtro para media e mediana em x e y separado para ter o centro, e tb para o raio(distancia do pixel central aos outros e mediana. pimba!)

        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            drawContours(drawing, contours_poly, (int)i, color);
            // rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
            // circle(drawing, centers[i], (int)radius[i], color, 2);
        }

        // circle(drawing, center, 4, Scalar(0, 0, 255), -1);
    }

    contour = drawing;
}

void thresh_callback_moments(cv::Mat src_gray, cv::Mat &contour, int thresh, cv::Point2f &center)
{
    cv::Mat canny_output;
    Canny(src_gray, canny_output, thresh, thresh * 2, 3, true);
    vector<vector<Point>> contours;
    findContours(canny_output, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    vector<Moments> mu(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i]);
    }
    Moments moment = moments(src_gray, true);

    vector<Point2f> mc(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        // add 1e-5 to avoid division by zero
        mc[i] = cv::Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
                            static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
        // cout << "mc[" << i << "]=" << mc[i] << endl;
    }
    Point2f my_mc;
    my_mc = cv::Point2f(static_cast<float>(moment.m10 / (moment.m00 + 1e-5)),
                        static_cast<float>(moment.m01 / (moment.m00 + 1e-5)));
    if (my_mc.x > 0 || my_mc.y > 0)
    {
        center = my_mc;
    }

    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC4);

    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        // drawContours(drawing, contours, (int)i, color, 2);
        // circle(drawing, mc[i], 4, color, -1);
    }

    // circle(drawing, my_mc, 4, Scalar(255, 0, 0), -1);

    // imshow( "Contours", drawing );
    /*
    cout << "\t Info: Area and Contour Length \n";
    for (size_t i = 0; i < contours.size(); i++)
    {
        cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
             << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength(contours[i], true) << endl;
    }
    */
    contour = drawing;
}

void sortXY(vector<cv::Point2f> array, vector<float> &x, vector<float> &y)
{

    for (int i = 0; i < array.size(); i++)
    {
        x.push_back(array.at(i).x);
        y.push_back(array.at(i).y);
    }

    sort(x.begin(), x.end());
    sort(y.begin(), y.end());
}

float median(vector<float> data)
{
    int half = data.size() / 2;

    if (data.size() % 2 == 0)
    {

        return (data.at(half - 1) + data.at(half)) / 2;
    }
    else
    {
        return data.at(half);
    }
}

void plotHistogram(cv::Mat src, cv::Mat &dst)
{
    /// Separate the image in 3 places ( B, G and R )
    vector<cv::Mat> bgr_planes;
    split(src, bgr_planes);

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float *histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // Draw the histograms for B, G and R
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    /// Draw for each channel
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
             cv::Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
             cv::Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
             cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    dst = histImage;

    /// Display
    /*
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
    imshow("calcHist Demo", histImage);
    */
}

sl::Rect focus_center(cv::Mat src, Point2f &focus_center, cv::Mat &grayscale, cv::Mat &blur, cv::Mat &histogram, cv::Mat &contour, findCenter type)
{

    cv::Point2f center;
    switch (type)
    {
    case findCenter::CONTOURS:
        thresh_callback_contours(blur, contour, focus_center_threshold, center);
        circle(contour, center, 4, Scalar(0, 0, 255), -1);
        break;
    case findCenter::MOMENTS:
        thresh_callback_moments(blur, contour, focus_center_threshold, center);
        circle(contour, center, 4, Scalar(0, 0, 255), -1);
        break;
    default:
        break;
    }
    focus_center = center;
    plotHistogram(src, histogram);

    // cout << "Window size: " << window_size << " px" << endl;
    // cin >> window_size;
    int window = adjust_focus_window(src, window_size, center);
    // window_size+=10;

    cv::Point2f v_up_left, v_down_right;
    v_up_left.x = center.x - window / 2;
    v_up_left.y = center.y - window / 2;
    v_down_right.x = center.x + window / 2;
    v_down_right.y = center.y + window / 2;

    cv::rectangle(contour, v_up_left, v_down_right, cv::Scalar(0, 0, 255), 3, 8, 0);

    return sl::Rect(center.x - window / 2, center.y - window / 2, window, window);
}

int adjust_focus_window(cv::Mat src, int max_size, Point center)
{
    int window = max_size + 1;
    while (1)
    {
        window--;
        if (center.x - window / 2 < 0)
            continue;
        if (center.y - window / 2 < 0)
            continue;
        if (center.x + window / 2 > src.cols)
            continue;
        if (center.y + window / 2 > src.rows)
            continue;
        break;
    }
    return window;
}
