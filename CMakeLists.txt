cmake_minimum_required(VERSION 3.0.0)
project(squareFocus VERSION 0.1.0)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} squareFocus.cpp headers.cpp)

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBS} ${ZED_LIBS} 
                                opencv_cudabgsegm
                                opencv_cudaobjdetect
                                opencv_cudastereo
                                opencv_cudafeatures2d
                                opencv_cudacodec
                                opencv_cudaoptflow
                                opencv_cudalegacy
                                opencv_cudawarping
                                opencv_cudaimgproc
                                opencv_cudafilters)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)









