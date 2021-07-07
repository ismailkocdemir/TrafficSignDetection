# Traffic Sign Detection with OpenCV and C++

Detection Pipeline consists of region proposals and classification:
1. A Shape detector module prepocesses the image and extracts regions that possibly include traffic signs
2. A set of Haar-based Cascade Classifiers tries to classify extracted regions into traffic sign categories.

## Data preparation

'prepare_data.py' is used for creating the data required for training of the cascade classifiers. 
It creates a set of positive samples for each traffic sign category, together with a set of negative samples 
shared among the classifier. To do that, OpenCV's precompiled tool (opencv_createsamples) is used. 

## Haar-based Cascade Classifier Training

'train_cascades.py' trains a separate classifier for each of the traffic signs. It also uses the precompiled binary
(opencv_traincascades).

Pretrained classifiers are provided under data/cascades/*/ folders. There are two types of classifier: One is trained on
real positive samples (no augmentations), the other is trained on the generated samples obtained by distorting the template 
images of the traffic signs. 


## Building/Testing

Compilation of the source code (C++) requires CMake(>2.8) and OpenCV 3.4.
To test the software run the only executable (TrafficSignDetection) by providing the cascade path (eg: data/cascades/real)
and the directory where test images are located.

The classifiers trained on the real images are trained on only subset of the signs (50,70,100 signs, give-way, pass-right-side and pedestrian-crossing), since some of them have no usable training samples.
The classifiers trained on generated images cover most of the signs making up majority of the dataset.

However, the performance of both is not so good. There are many missed signs both at region proposal stage and classification stage (e.g. speed signs 
are mostly can't be differentiated)