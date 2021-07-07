#pragma once
#ifndef SHAPEDETECTOR_H_
#define SHAPEDETECTOR_H_

#include <iostream>
#include <vector>
#include <string>
#include <map>

using namespace std;

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;

class ShapeDetector
{
public:
    ShapeDetector();
    ~ShapeDetector();
    map<string, Rect> detect_shapes(const Mat &image, bool show);

private:
    string identify_shape(const Mat &curve);
    double angle(Point pt1, Point pt2, Point pt0);

};

#endif
