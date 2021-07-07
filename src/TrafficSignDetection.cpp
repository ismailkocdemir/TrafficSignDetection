#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include <iostream>
#include <iterator>
#include <map>

#include "ShapeDetector.h"

using namespace std;
using namespace cv;


/** Function Headers */
void classifyAndDisplay( Mat &frame, map<string, Rect> &shapes, int idx, bool show);


/** Haar-Based Cascade Classifier for each Traffic Sign */
map<string, CascadeClassifier> haar_cascades;


/** @function main */
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h       |                            |                                       }"
                             "{show s       |                            | if given, will show detections in GUI }"
                             "{cascade_dir  | ../data/cascades/real      | Path to cascade (real or template)    }"
                             "{test_dir     | ../data/dataset/img_test   | Test image directory                  }");

    parser.about( "\nThis program demonstrates the usage of cv::CascadeClassifier class to classify traffic signs.\n" );

    bool help = parser.has("help");
    if (help)
    {
        parser.printMessage();
        return 0;
    }

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
        cerr << "Error: invalid cascade directory\n";
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
    // look for pretrained classifier weights for each traffic sign
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
    ShapeDetector shape_detector;
    vector<String> images;
    utils::fs::glob(test_dir, "*.jpg", images);    
    int idx = 0;
    for (const auto & entry : images)
    {   
        Mat frame;
        frame = imread( entry, IMREAD_ANYCOLOR );
        
        
        if ( !frame.data )
            continue;

        // get region proposals from the image using some shape heuristics.
        map<string,Rect> proposals = shape_detector.detect_shapes(frame, show);
        // validate and classify the region proposals by Cascaded Classifier.
        classifyAndDisplay( frame, proposals, idx++, show);
        
        if(show)
        {
            if( waitKey(0) == 27 ) break; // quit (esc)
            if( waitKey(0) == 2555904) continue; // next image (arrow)
        }

    }
    return 0;
}


/** @function classifyAndDisplay */
void classifyAndDisplay( Mat &frame, map<string, Rect> &proposals, int idx, bool show)
{   
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    bool found_sign = 0;
    
    // loop trough the proposals.
    map<string, Rect>::iterator itr_1;
    for (itr_1 = proposals.begin(); itr_1 != proposals.end(); ++itr_1)
    {   
        // loop trough the classifiers for each sign, given this specific proposal.
        map<string, CascadeClassifier>::iterator itr_2;
        for (itr_2 = haar_cascades.begin(); itr_2 != haar_cascades.end(); ++itr_2) 
        {
            vector<Rect> signs;
            itr_2->second.detectMultiScale( frame_gray, signs );
            
            // if we find can classify the sign, we break out of the loop of classifiers and be done with this proposal.
            found_sign = (signs.size() > 0);
            if (found_sign)
            {
                // draw the bounding box and the sign label on the image.
                rectangle( frame, itr_1->second, Scalar( 255, 0, 255 ), 4);
                putText(
                    frame, // original image
                    itr_2->first, // sign category
                    Point(itr_1->second.x, itr_1->second.y-3), // text location
                    FONT_HERSHEY_SIMPLEX,  // font
                    0.5, Scalar(255,0,255), 2); // size and color
                break;
            }
        }
    }
    
    if(show) imshow("Traffic Sign Detection - Test", frame);
    
    string filename = "../data/detection_results/det_" + to_string(idx) + ".jpg";
    imwrite(filename, frame);
    cout << "detection results are written to: " << filename << endl;
    
}
