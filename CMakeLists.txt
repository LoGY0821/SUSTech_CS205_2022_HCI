cmake_minimum_required(VERSION 3.15)
project(C_Test)
#project(Training_test)

set(CMAKE_CXX_STANDARD 14)
# 可以将下面的路径单独添加到环境变量OpenCV_DIR
set(OpenCV_DIR "D:\\Productivity\\OpenCV\\opencv\\mingw-build\\install")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
# "C_Test"与你工程的命名相关，注意和原来CMakeList.txt中的声明一致
add_executable(C_Test HCI.cpp)
#add_executable(Training_test Readfolder.cpp)


target_link_libraries(C_Test ${OpenCV_LIBS})