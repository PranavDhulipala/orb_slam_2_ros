cmake_minimum_required(VERSION 3.0.0)
project(orb_slam2_ros)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:64000000")
#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")
SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
SET(DBUILD_SHARED false)

if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
#else()
#   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


find_package(OpenCV 4.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 3.0 QUIET)
   if(NOT OpenCV_FOUND)
      find_package(OpenCV 2.4.3 QUIET)
      if(NOT OpenCV_FOUND)
         message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
     endif()
   endif()
endif()


find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
#message(STATUS "Compile With map save/load function")


find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs cv_bridge image_transport tf sensor_msgs dynamic_reconfigure message_generation)
set (DYNAMIC_RECONFIGURE_PATH ros/config/dynamic_reconfigure.cfg)
generate_dynamic_reconfigure_options(
        ${DYNAMIC_RECONFIGURE_PATH}
)


set(LIBS_ORBSLAM
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${CATKIN_DEVEL_PREFIX}/lib/DBoW2.lib
${CATKIN_DEVEL_PREFIX}/lib/g2o.lib

)

set(LIBS_ROS
${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}.lib
${OpenCV_LIBS}
${catkin_LIBRARIES}
)

add_service_files(
        FILES
        SaveMap.srv
)

generate_messages(
        DEPENDENCIES
        std_msgs
)

catkin_package (
  CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge image_transport tf sensor_msgs dynamic_reconfigure message_runtime
  LIBRARIES {PROJECT_NAME} DBoW2 g2o
)


include_directories(
${PROJECT_SOURCE_DIR}/orb_slam2
${PROJECT_SOURCE_DIR}/orb_slam2/include
${PROJECT_SOURCE_DIR}/ros/include
${Pangolin_INCLUDE_DIRS}   
${EIGEN3_INCLUDE_DIR}
${catkin_INCLUDE_DIRS}
${PROJECT_BINARY_DIR}
)

include(${PROJECT_SOURCE_DIR}/orb_slam2/Thirdparty/DBoW2/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/orb_slam2/Thirdparty/g2o/CMakeLists.txt)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/orb_slam2/lib)

add_library(${PROJECT_NAME} SHARED
orb_slam2/src/System.cc
orb_slam2/src/Tracking.cc
orb_slam2/src/LocalMapping.cc
orb_slam2/src/LoopClosing.cc
orb_slam2/src/ORBextractor.cc
orb_slam2/src/ORBmatcher.cc
orb_slam2/src/FrameDrawer.cc
orb_slam2/src/Converter.cc
orb_slam2/src/MapPoint.cc
orb_slam2/src/KeyFrame.cc
orb_slam2/src/Map.cc
orb_slam2/src/Optimizer.cc
orb_slam2/src/PnPsolver.cc
orb_slam2/src/Frame.cc
orb_slam2/src/KeyFrameDatabase.cc
orb_slam2/src/Sim3Solver.cc
orb_slam2/src/Initializer.cc
orb_slam2/src/Viewer.cc
orb_slam2/src/MapDrawer.cc
)

add_dependencies (${PROJECT_NAME} g2o DBoW2)

target_link_libraries(${PROJECT_NAME}
${LIBS_ORBSLAM}
)

#set(Boost_USE_STATIC_LIBS        ON)
find_package(Boost REQUIRED COMPONENTS serialization)
target_link_libraries(${PROJECT_NAME} Boost::serialization)


add_executable (${PROJECT_NAME}_mono
ros/src/MonoNode.cc
ros/src/Node.cc
)
add_dependencies (${PROJECT_NAME}_mono ${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}_mono
${LIBS_ORBSLAM}
${LIBS_ROS}
)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/ros/STEREO)
add_executable (${PROJECT_NAME}_stereo
ros/include/StereoNode.h
ros/src/StereoNode.cc
ros/include/Node.h
ros/src/Node.cc
)
add_dependencies (${PROJECT_NAME}_stereo ${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}_stereo
${LIBS_ORBSLAM}
${LIBS_ROS}
)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/ros/RGB-D)
add_executable (${PROJECT_NAME}_rgbd
ros/include/RGBDNode.h
ros/src/RGBDNode.cc
ros/include/Node.h
ros/src/Node.cc
)
add_dependencies (${PROJECT_NAME}_rgbd ${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}_rgbd
${PROJECT_NAME}
${LIBS_ROS}
)
