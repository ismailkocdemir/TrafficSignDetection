#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include <iostream>
#include <iterator>
#include <map>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, int idx, bool show);

/** Global variables */
map<string, CascadeClassifier> haar_cascades;

/** @function main */
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h       |                            |                                 }"
                             "{show s       |                            |                                 }"
                             "{cascade_dir  | ../data/cascades/real      | Path to cascade.                }"
                             "{test_dir     | ../data/dataset/img_test   | Test image directory            }");

    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect traffic signs from images.\n"
                  "It uses HAAR-Cascades to detect existence of any sign..\n\n" );
    
    parser.printMessage();

    String cascade_dir = parser.get<String>("cascade_dir");
    String test_dir = parser.get<String>("test_dir");
    bool show = parser.has("show");

    if( !utils::fs::isDirectory(test_dir) )
    {
        cerr << "Error: invalid test directory\n";
        return -1;
    }

    if( !utils::fs::isDirectory(cascade_dir) )
    {
        cerr << "Error: invalid test directory\n";
        return -1;
    }

    // Load all cascades from the directory
    vector<String> sign_dirs;
    utils::fs::glob(cascade_dir, "*", sign_dirs, false, true);
    if(sign_dirs.size() == 0)
    {
        cerr << "Could not find any cascade in the given directory: " << cascade_dir << endl;
        return -1;
    }

    for(const auto & path: sign_dirs)
    {
        string sign_name = path.substr(path.find_last_of('/') + 1, path.length());
        CascadeClassifier haar_cascade;       
        
        if(!haar_cascade.load(utils::fs::join(path, "cascade.xml")))
        {
            cerr << "Error loading cascade in: " << path << endl;
            continue;
        }
        cout << "loaded cascade for sign: " << sign_name << endl;
        haar_cascades.insert(pair<string, CascadeClassifier>(sign_name, haar_cascade));
    }

    // Loop trough test images and run the classifiers.
    vector<String> images;
    utils::fs::glob(test_dir, "*.jpg", images);    
    int idx = 0;
    for (const auto & entry : images)
    {   
        Mat frame;
        frame = imread( entry, IMREAD_ANYCOLOR );
        //resize(frame, frame, Size(640, 480));
        
        if ( !frame.data )
            continue;

        //cout << "Processing:" << entry << endl;
        detectAndDisplay( frame, idx++, show );
        if(show)
        {
            if( waitKey(0) == 27 ) break; // quit (esc)
            if( waitKey(0) == 2555904) continue; // next image (right arrow)
        }
    }
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int idx, bool show)
{   
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    bool found_signs = 0;
    
    map<string, CascadeClassifier>::iterator itr;
    for (itr = haar_cascades.begin(); itr != haar_cascades.end(); ++itr) 
    {
        vector<Rect> signs;
        itr->second.detectMultiScale( frame_gray, signs );
        for ( size_t i = 0; i < signs.size(); i++ )
        {
            rectangle( frame, signs[i], Scalar( 0, 0, 255 ), 1);
            putText(
                frame, 
                itr->first, 
                Point(signs[i].x, signs[i].y-3), 
                FONT_HERSHEY_SIMPLEX, 
                0.3, Scalar(0,0,255), 1);
        }
        found_signs = (signs.size() > 0);
    }

    if(show) imshow("Traffic Sign Detection - Test", frame);
    if(found_signs)
    {
        string filename = "../data/detection_results/det_" + to_string(idx) + ".png";
        imwrite(filename, frame);
        cout << "found sings written to: " << filename << endl;
    }
}
