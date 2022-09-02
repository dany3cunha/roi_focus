#include "include/headers.hpp"

int main(int argc, char **argv)
{

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080;
    init_params.depth_mode = DEPTH_MODE::ULTRA;
    init_params.coordinate_units = UNIT::METER;

    if (argc > 1)
        init_params.input.setFromSVOFile(argv[1]);

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS)
    {
        printf("%s\n", toString(err).c_str());
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE::STANDARD;

    // Prepare new image size to retrieve half-resolution images
    Resolution image_size = zed.getCameraInformation().camera_resolution;
    int new_width = image_size.width / 2;
    int new_height = image_size.height / 2;

    Resolution new_image_size(new_width, new_height);

    // To share data between sl::Mat and cv::Mat, use slMat2cvMat()
    // Only the headers and pointer to the sl::Mat are copied, not the data itself
    sl::Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    cv::Mat image_ocv = slMat2cvMat(image_zed);

    cv::Mat blank(new_width, new_height, CV_8UC4, cv::Scalar(255, 255, 255));
    cv::Mat displayImage(blank), gray(blank), blur(blank), contour(blank), equiHistImg(blank), blur3ch(blank);
    cv::Mat histogram(512, 400, CV_8UC4, cv::Scalar(0, 0, 0));

    auto start_t = std::chrono::system_clock::now();
    auto end_t = std::chrono::system_clock::now();

    char key = ' ';
    while (key != 'q')
    {
        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS)
        {
            // Retrieve the left image, depth image in half-resolution
            zed.retrieveImage(image_zed, VIEW::RIGHT, MEM::CPU, new_image_size);
        }

        end_t = chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_t - start_t;

        // cv::cvtColor(image_ocv.clone(), gray, cv::COLOR_BGR2GRAY);
        //  o threshold nao faz muito sentido pq satura tudo ao maximo confundindo as coisas
        // cv::threshold(gray, gray, 220, 255, cv::THRESH_BINARY);

        // cv::blur(gray, blur, cv::Size(1, 1));

        // cv::GaussianBlur(gray, blur, cv::Size(13, 13), 0);

        // cv::medianBlur(gray, blur, 9);

        cv::GaussianBlur(image_ocv, blur, cv::Size(13, 13), 0);
        // cv::medianBlur(image_ocv,blur,13);
        cv::cvtColor(blur, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, blur, 250, 255, cv::THRESH_BINARY);

        sl::Rect squareWindow;
        cv::Point2f left_up, right_down;
        if (elapsed_seconds.count() >= refreshRate)
        {

            cv::Point2f foc_center;

            squareWindow = focus_center(image_ocv, foc_center, gray, blur, histogram, contour, findCenter::MOMENTS);

            if (ERROR_CODE::SUCCESS != zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI, squareWindow))
            {

                cout << " Cannot aplied new ROI" << endl;
                cout << "   x: " << squareWindow.x << " y: " << squareWindow.y << endl;
            }
            start_t = end_t;
            left_up.x = squareWindow.x;
            left_up.y = squareWindow.y;
            right_down.x = left_up.x + squareWindow.width;
            right_down.y = left_up.y + squareWindow.height;

            // Remap blur image (binary) to 3 channels to highlight
            cv::cvtColor(blur, blur3ch, cv::COLOR_GRAY2BGR);
            cv::circle(blur3ch, foc_center, 4, cv::Scalar(0, 255, 0), -1);
            cv::rectangle(blur3ch, left_up, right_down, cv::Scalar(0, 0, 255), 3, 8);
        }

        displayImage = image_ocv.clone();
        cv::rectangle(displayImage, left_up, right_down, cv::Scalar(0, 0, 255), 3, 8);

        if (GRID)
        {
            grid(displayImage, displayImage, 6, new_width, new_height);
            cv::imshow("Live with grid", displayImage);
        }
        else
        {
            cv::imshow("Live", displayImage);
        }

        // cv::imshow("Grayscale", gray);
        cv::imshow("Moments", blur3ch);
        // cv::imshow("Histogram", histogram);
        // cv::imshow("Contour", contour);
        // cv::equalizeHist(gray, equiHistImg);
        // cv::imshow("EquiHistogram Image", equiHistImg);

        key = cv::waitKey(10);
    }

    zed.close();
}
