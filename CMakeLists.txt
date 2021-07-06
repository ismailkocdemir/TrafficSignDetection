cmake_minimum_required( VERSION 2.8 )
project( ObjectDetection )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_STATIC ON)
set(OpenCV_CUDA OFF)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories( ${OpenCV_LIB_DIR} )

file(GLOB my_source_files ./src/*)
add_executable( ObjectDetection ${my_source_files})
target_link_libraries( ObjectDetection ${OpenCV_LIBS} )
