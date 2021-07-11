#include "ShapeDetector.h"

ShapeDetector::ShapeDetector(){}
ShapeDetector::~ShapeDetector(){}

double ShapeDetector::angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void ShapeDetector::preprocess(Mat &image, Mat &thresh)
{
    // Equalize histogram. this seems to help. 
    equalizeHist( image, thresh);

    // Apply thresholding/edge detection before finding conours.
    GaussianBlur(thresh, thresh, Size(3, 3), 0.0);
    threshold(thresh, thresh, 150, 255, THRESH_BINARY);

    // Improve the structure by first eroding then dilating
    Mat element_open = getStructuringElement( MORPH_RECT, Size(3, 3));
    morphologyEx( thresh, thresh, MORPH_OPEN, element_open);
    
    Mat element_close = getStructuringElement( MORPH_RECT, Size(3, 3));
    morphologyEx( thresh, thresh, MORPH_CLOSE, element_close);
}

string ShapeDetector::identify_shape(const Mat &curve)
{
    string shape = "unwanted";
    vector<Point> approx;
    approxPolyDP(curve, approx, arcLength(curve, true)*0.04, true);

    if (approx.size() == 3)
    {
        shape = "triangle"; 
    }
    else if (approx.size() >= 4 && approx.size() <= 6)
    {
        // Number of vertices of polygonal curve
        int vtc = approx.size();

        // Get the cosines of all corners
        vector<double> cos;
        for (int j = 2; j < vtc+1; j++)
            cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));

        // Sort ascending the cosine values
        sort(cos.begin(), cos.end());

        // Get the lowest and the highest cosine
        double mincos = cos.front();
        double maxcos = cos.back();

        // Use the degrees obtained above and the number of vertices
        // to determine the shape of the contour
        if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
            shape = "rectangle";
        else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
            shape = "hexagon";
    }
    else
    {
        // Detect and label circles
        
        double area = contourArea(curve);
        Rect r = boundingRect(curve);
        int radius = r.width / 2;

        if (abs(1 - ((double)r.width / r.height)) <= 0.2 &&
            abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
            shape = "circle";
    }
    return shape;
}

map<string, Rect> ShapeDetector::detect_shapes(const Mat &image, bool show)
{

    map<string, Rect> shapes;
    
    Mat dummy, s_channel, hsv[3], thresh;
    //cvtColor(image, gray, CV_BGR2GRAY);
    cvtColor(image, dummy, CV_BGR2HSV);
    split(dummy, hsv);
    s_channel = hsv[1];

    // Limited but robust option: Merge red-color and blue-color masks in HSV space.
    //GaussianBlur(gray, thresh, Size(5, 5), 0);
    //threshold(thresh, thresh, 60, 255, THRESH_BINARY);
    /*
    Mat r_channel1, r_channel2, b_channel;
    inRange(dummy, Scalar(0, 70, 50), Scalar(10, 255, 255), r_channel1);
    inRange(dummy, Scalar(170, 70, 50), Scalar(180, 255, 255), r_channel2);
    inRange(dummy, Scalar(100,150,0), Scalar(140,255,255), b_channel);
    Mat thresh = r_channel1 | r_channel2 | b_channel; // | gray;
    */
    
    preprocess(s_channel, thresh);

    // Find contours and apply some heuristics to reduce the number of proposals.
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(thresh, contours, hierarchy,  RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
	for (int i = 0; i < contours.size(); i++)
	{
        /* Identify the contour shape: triangle, square etc.
        * I did not have time to use this information, but it could help reduce the complexity.
        * e.g.: when we know the sign is a square, only run classifiers that recognize square signs.
        */
        string shape_id = identify_shape(Mat(contours[i]));
        // if the contour is not one of known sign shapes, skip this one.
        if(shape_id == "unwanted")
            continue;

        // Draw a bounding box around.
        Rect bbox = boundingRect(contours[i]);
        
        // skip too wide and too long contours. 
        double ar = 1.0 * bbox.width / bbox.height;
        if (ar >= 4. || ar <= 0.25)
            continue;
        
        float b_area = bbox.width * bbox.height; 
        float c_area = fabs(contourArea(contours[i]));
		
        // Skip too small or too large contours
        if ( b_area < 50 or b_area > 100000 )
			continue;

        /*
        //Mat hull;
        //convexHull(contours[i], hull);
        //float hull_area = contourArea(hull);
        float solidity = float(b_area)/c_area;
        // Skip contours that does not fill the box very well  
        if ( solidity < 0.5 )
			continue;
        */

        // Expand a little bit in case box falls short.
        bbox = bbox + Size(50,50) + Point(-25, -25);
        bbox = bbox & cv::Rect(0, 0, image.cols, image.rows);
        
        if(show)
        {
            rectangle( thresh, bbox, Scalar( 255, 0, 255 ), 4);
            putText(
                thresh, // original image
                shape_id, // sign category
                Point(bbox.x, bbox.y-3), // text location
                FONT_HERSHEY_SIMPLEX,  // font
                0.5, Scalar(255,0,255), 2 // size and color
            );
        }
        shapes.insert(pair<string, Rect> (shape_id, bbox)); 
    }
    
    if (show)
    {
        imshow("Traffic Sign Proposals", thresh);
        waitKey(100);
    }
    
    
    return shapes;
}
