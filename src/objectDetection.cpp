#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, int idx );

/** Global variables */
CascadeClassifier haar_cascade;

/** @function main */
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h | |}"
                             "{haar_cascade | /home/ismail/ford/code/detection/data/cascades_default/cascade.xml | Path to cascade.}"
                             "{test_dir | /home/ismail/ford/code/detection/data/img | Test image directory}");

    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect traffic signs from images.\n"
                  "It uses HAAR-Cascades to detect existence of any sign. Then, it classifies the found signs with HOG + SVM.\n\n" );
    
    //parser.printMessage();

    String haar_cascade_name = parser.get<String>("haar_cascade");
    String test_dir = parser.get<String>("test_dir");

    if( !cv::utils::fs::isDirectory(test_dir) )
    {
        cout << "Error: invalid test directory\n";
        return -1;
    }

    //-- 1. Load the cascades
    if( !haar_cascade.load( haar_cascade_name ) )
    {
        cout << "Error loading cascade\n";
        return -1;
    };

    std::vector<cv::String> filenames;
    String full_pattern = cv::utils::fs::join(test_dir, "*.jpg");
    glob(full_pattern, filenames, false);    
    int idx = 0;
    for (const auto & entry : filenames)
    {   
        Mat frame;
        frame = imread( entry, cv::IMREAD_ANYCOLOR );
        resize(frame, frame, Size(640, 480));
        if ( !frame.data )
            continue;

        //-- 3. Apply the classifier to the frame
        cout << "Processing:" << entry << endl;
        detectAndDisplay( frame, idx++ );
        /*
        if( waitKey(0) == 27 )
        {
            break; // escape
        }
        else if( waitKey(0) == 2555904)
        {
            continue; // next image
        }
        */
    }
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int idx)
{   
    //Mat frame_gray;
    //cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    //equalizeHist( frame_gray, frame_gray );

    //-- Detect signs
    std::vector<Rect> signs;
    haar_cascade.detectMultiScale( frame, signs );

    for ( size_t i = 0; i < signs.size(); i++ )
    {
        Point center( signs[i].x + signs[i].width/2, signs[i].y + signs[i].height/2 );
        ellipse( frame, center, Size( signs[i].width/2, signs[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );

        Mat faceROI = frame( signs[i] );
    }

    // imshow( "Sign detection", frame );
    string filename = "../data/detections/det_" + to_string(idx) + ".png";
    imwrite(filename, frame );
    cout << "Written results to: " << filename << endl;
    cout << "------------------\n";
}
