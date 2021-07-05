#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <experimental/filesystem>


namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
CascadeClassifier haar_cascade;

/** @function main */
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{haar_cascade|/home/ismail/ford/code/detection/data/cascades_default/cascade.xml|Path to cascade.}"
                             "{test_dir|/home/ismail/ford/code/detection/data/img_test|Test image directory}");

    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect traffic signs from images.\n"
                  "It uses HAAR-Cascades to detect existence of any sign. Then, it classifies the found signs with HOG + SVM.\n\n" );
    parser.printMessage();

    string haar_cascade_name = parser.get<string>("haar_cascade");
    fs::path test_dir = parser.get<string>("test_dir");

    //-- 1. Load the cascades
    if( !haar_cascade.load( haar_cascade_name ) )
    {
        cout << "--(!)Error loading cascade\n";
        return -1;
    };

    Mat frame;
    for (const auto & entry : fs::directory_iterator(test_dir))
    {   
        frame = imread( entry.path().string(), 1 );
        if ( !frame.data )
            continue;

        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );

        if( waitKey(0) == 27 )
        {
            break; // escape
        }
        else if( waitKey(0) == 2555904)
        {
            continue; // next image
        }
    }
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect signs
    std::vector<Rect> signs;
    haar_cascade.detectMultiScale( frame_gray, signs );

    for ( size_t i = 0; i < signs.size(); i++ )
    {
        Point center( signs[i].x + signs[i].width/2, signs[i].y + signs[i].height/2 );
        ellipse( frame, center, Size( signs[i].width/2, signs[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );

        Mat faceROI = frame_gray( signs[i] );
    }

    //-- Show what you got
    imshow( "Sign detection", frame );
}
