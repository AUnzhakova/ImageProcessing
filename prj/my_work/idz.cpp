#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>


using namespace cv;
using std::cout;

int threshold_value = 0;
int threshold_type = 1;
int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;

double M1;
double M2;
double M3;
Mat3b plot_image;

Mat src, src_gray, dst;
Mat hist;
int arrOfM1[256], arrOfM2[256], arrOfM3[256];
const char* window_name = "Tsai Binarisation";

const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

int tsaiMoment(Mat histo){
    double total = 0;
    double m0 = 1.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, sum = 0.0, p0 = 0.0;
    double cd, c0, c1, z0, z1; // auxiliary variables
    int threshold = -1;
    Mat pgm_double;
    histo.convertTo(pgm_double, CV_64F);
    double *I = pgm_double.ptr<double>(0);
    for (int i=0; i<pgm_double.rows; i++){
        total += pgm_double.at<double>(i);
    }
    for (int i=0; i<pgm_double.rows; i++){
        pgm_double.at<double>(i) /= total;
    }
    total = 0;
    for (int i=0; i<pgm_double.rows; i++){
        total += pgm_double.at<double>(i);
    }
    
    for (int i = 0; i < histo.rows; ++i) {
        m1 += i * pgm_double.at<double>(i);
        m2 += i * i * pgm_double.at<double>(i);
        m3 += i * i * i * pgm_double.at<double>(i);
    }
    M1 = m1;
    M2 = m2;
    M3 = m3;
    
    cd = m0 * m2 - m1 * m1;
    c0 = ( -m2 * m2 + m1 * m3 ) / cd;
    c1 = ( m0 * -m3 + m2 * m1 ) / cd;
    z0 = 0.5 * (-c1-sqrt(c1*c1 - 4.0*c0));
    z1 = 0.5 * (-c1+sqrt(c1*c1 - 4.0*c0));
    p0=(z1-m1)/(z1-z0);
    cout << "m0  " << m0 << std::endl;
    cout << "m1  " << m1 << std::endl;
    cout << "m2  " << m2 << std::endl;
    cout << "m3  " << m3 << std::endl;
    cout << "z0  " << z0 << std::endl;
    cout << "z1  " << z1 << std::endl;
    cout << "p0  " << p0 << std::endl;
    cout << "p1  " << 1 - p0 << std::endl;
    
    sum=0;
    for (int i = 0; i < histo.rows; ++i) {
        sum+=pgm_double.at<double>(i);
        if (sum>p0) {
            threshold = i;
                break; }
    }
//    cout<<"Moments"<< m2 <<"  ___  Threshold" << threshold <<std::endl;
    
    return threshold;
}

void show_histogram(std::string const& name, cv::Mat1b const& image)
{
    // Set histogram bins count
    int bins = 256;
    int histSize[] = {bins};
    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    // create matrix for histogram
    int channels[] = {0};
    
    // create matrix for histogram visualization
    int const hist_height = 256;
    Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);
    
    calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    double max_val=0;
    minMaxLoc(hist, 0, &max_val);
    
    // visualize each bin
    for(int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal*hist_height/max_val);
        cv::line
        ( hist_image
         , cv::Point(b, hist_height-height), cv::Point(b, hist_height)
         , cv::Scalar::all(255)
         );
    }
    //    cout<<"Hist"<< hist <<std::endl;
    cv::imshow(name, hist_image);
}

static void Threshold_Demo( int, void* )
{
    /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
     */
    //    threshold( src_gray, dst, threshold_value, max_binary_value, threshold_type );
    threshold( src_gray, dst, tsaiMoment(hist), max_binary_value, threshold_type );
    for(int i = 0; i < dst.rows; i++){
        for(int j=0; j < dst.cols; j++){
            //apply condition here
            if(dst.at<uchar>(i,j) == 0){
                dst.at<uchar>(i,j) = 34;
            }
            if(dst.at<uchar>(i,j) == 255){
                dst.at<uchar>(i,j) = 236;
            }
        }
    }
    imshow( window_name, dst );
    //    show_histogram("image2 hist", dst);
    //    Moments m = moments(threshold_value,true);
    //    cout<<"Moments"<< m.m01 <<std::endl;
    //26 27 27  // 130 118 90
}


void tsaiMomentsForEveryThreshold( int threshold){
    int bins = 256;
    int histSize[] = {bins};
    int channels[] = {0};
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    Mat1b thresholdImage;
    cv::threshold( src_gray, thresholdImage, threshold, max_binary_value, threshold_type );
    
//    Mat image;
    for(int i = 0; i < thresholdImage.rows; i++){
        for(int j=0; j < thresholdImage.cols; j++){
            //apply condition here
            if(thresholdImage.at<uchar>(i,j) == 0){
                thresholdImage.at<uchar>(i,j) = 34;
            }
            if(thresholdImage.at<uchar>(i,j) == 255){
                thresholdImage.at<uchar>(i,j) = 236;
            }
        }
    }
    Mat histo;
    calcHist(&thresholdImage, 1, channels, cv::Mat(), histo, 1, histSize, ranges, true, false);
    //    imshow( "Thresh", thresholdImage );
    
    double total = 0;
    double m1 = 0.0, m2 = 0.0, m3 = 0.0;
    Mat pgm_double;
    histo.convertTo(pgm_double, CV_64F);
    double *M = pgm_double.ptr<double>(0);
    for (int i=0; i<pgm_double.rows; i++){
        total += pgm_double.at<double>(i);
    }
    for (int i=0; i<pgm_double.rows; i++){
        pgm_double.at<double>(i) /= total;
    }
    total = 0;
    for (int i=0; i<pgm_double.rows; i++){
        total += pgm_double.at<double>(i);
    }
    
    for (int i = 0; i < histo.rows; ++i) {
        m1 += i * pgm_double.at<double>(i) ;
        m2 += i * i * pgm_double.at<double>(i) ;
        m3 += i * i * i * pgm_double.at<double>(i) ;
    }
    
    arrOfM1[threshold] = m1;
    arrOfM2[threshold] = m2;
    arrOfM3[threshold] = m3;
}

void drawMPlot(){
    int const plot_height = 256;
    int const threshold_levels = 256;
    plot_image = cv::Mat3b::zeros(plot_height, threshold_levels);
    double max_val1 = 0;
    double max_val2 = 0;
    double max_val3 = 0;
    for (int i=0; i < threshold_levels; ++i){
        if (arrOfM1[i] > max_val1){
            max_val1 = arrOfM1[i];
        }
        if (arrOfM2[i] > max_val2){
            max_val2 = arrOfM2[i];
        }
        if (arrOfM3[i] > max_val3){
            max_val3 = arrOfM3[i];
        }
    }
    // visualize each bin
    for(int x = 0; x < threshold_levels; x++) {
        int const y1 = cvRound(plot_height*arrOfM1[x]/max_val1);
        int const y2 = cvRound(plot_height*arrOfM2[x]/max_val2);
        int const y3 = cvRound(plot_height*arrOfM3[x]/max_val3);
        circle(plot_image, Point(x, 256 - y1 ), 1.0, Scalar(0, 255, 255), -1, 8); //yellow
        circle(plot_image, Point(x,256 - y2), 1.0, Scalar(255, 0, 0), -1, 8);
        circle(plot_image, Point(x,256 - y3), 1.0, Scalar(0, 0, 255), -1, 8);
    }
//    imshow( "plot M1, M2, M3", plot_image );
}

void drawSecondMPlot(){
    int const plot_height = 256;
    int const threshold_levels = 256;
//    Mat3b plot_image = cv::Mat3b::zeros(plot_height, threshold_levels * 3);
    double max_val1 = 0;
    double max_val2 = 0;
    double max_val3 = 0;
    for (int i=0; i < threshold_levels; ++i){
        if (arrOfM1[i] > max_val1){
            max_val1 = arrOfM1[i];
        }
        if (arrOfM2[i] > max_val2){
            max_val2 = arrOfM2[i];
        }
        if (arrOfM3[i] > max_val3){
            max_val3 = arrOfM3[i];
        }
    }
    // visualize each bin
    for(int x = 0; x < threshold_levels; x++) {
        int const y1 = cvRound(plot_height*M1/max_val1);
        int const y2 = cvRound(plot_height*M2/max_val2);
        int const y3 = cvRound(plot_height*M3/max_val3);
        circle(plot_image, Point(x,256 - y1 + 130 ), 1, Scalar(0, 227, 255), -1, 8); //yellow
        circle(plot_image, Point(x,256 - y2 + 118 ), 1, Scalar(227, 0, 0), -1, 8);
        circle(plot_image, Point(x,256 - y3 + 90), 1.0, Scalar(0, 0, 227), -1, 8);
    }
    for (int y = 0; y< 256; ++y){
        circle(plot_image, Point(tsaiMoment(hist), y ), 1.0, Scalar(255, 255, 255), -1, 8);
    }
    imshow( "second plot M1, M2, M3", plot_image );
    
}


int main( int argc, char** argv )
{
    //! [load]
    String imageName("/Users/alisaunzakova/Desktop/completed/stuff3.jpg"); // by default
    if (argc > 1)
    {
        imageName = argv[1];
    }
    src = imread( imageName, IMREAD_COLOR ); // Load an image
    if (src.empty())
    {
        cout << "Cannot read image: " << imageName << std::endl;
        return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
    //    calcHist(&src_gray, 1, ch, noArray(), hist, 2, histSize, ranges, true);
    //! [load]
    
    //! [window]
    namedWindow( window_name, WINDOW_AUTOSIZE ); // Create a window to display results
    //! [window]
    
    //! [trackbar]
    createTrackbar( trackbar_type,
                   window_name, &threshold_type,
                   max_type, Threshold_Demo ); // Create Trackbar to choose type of Threshold
    
    createTrackbar( trackbar_value,
                   window_name, &threshold_value,
                   max_value, Threshold_Demo ); // Create Trackbar to choose Threshold value
    //! [trackbar]
    show_histogram("image1 hist", src_gray);
    Threshold_Demo( 0, 0 ); // Call the function to initialize
    cv::imshow("src_gray", src_gray);
    /// Wait until user finishes program
    //    cout << "Image histo =" << hist << std::endl;
    //    tsaiMomentsForEveryThreshold(180);
    //    cout << "New moment1 =" << tsaiMomentsForEveryThreshold(150) << std::endl;
    for (int i = 0; i < 256; ++i) {
        tsaiMomentsForEveryThreshold(i);
    }
    drawMPlot();
    drawSecondMPlot();
     waitKey();
    return 0;
}


