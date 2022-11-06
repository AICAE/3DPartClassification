CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
#ENABLE_TESTING()
PROJECT(OCC_PROJECTOR)


set(CMAKE_CXX_STANDARD 17)  # argparse needs c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow use of find_package(HDF5), find_package(MOAB), etc.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

FIND_PACKAGE(Eigen3 REQUIRED)

#FIND_PACKAGE(CGAL REQUIRED COMPONENTS Core)
#INCLUDE_DIRECTORIES(${CGAL_INCLUDE_DIRS})


# need to find either OCE or OCC
if(False)
  FIND_PACKAGE(OCC QUIET COMPONENTS FoundationClasses;ModelingData;ModelingAlgorithms)

  IF(NOT OCE_FOUND AND NOT OCC_FOUND)
    MESSAGE(STATUS "OCE and OCC not found")
  ELSEIF( OCE_FOUND AND NOT OCC_FOUND )
    SET(OCC_INCLUDE_DIRS ${OCE_INCLUDE_DIRS})
  ELSEIF ( NOT OCE_FOUND AND OCC_FOUND )  
    SET(OCC_INCLUDE_DIRS ${OpenCASCADE_INCLUDE_DIR})
  ENDIF()
  SET(OCC_LINK_LIBS TKernel TKMath  TKTopAlgo TKG3d TKBRep 
                  TKMesh TKSTL TKXMesh TKBO TKBool TKSTEPBase TKSTEP TKXSBase )
else()
  SET(OCC_INCLUDE_DIRS "/usr/include/opencascade")
  link_directories("/usr/lib/x86_64-linux-gnu/")  # debug `libd`
  SET(OCC_LINK_LIBS TKernel TKMath  TKTopAlgo TKG3d TKBRep 
                  TKMesh TKRWMesh TKSTL TKXMesh TKBO TKBool TKSTEPBase TKSTEP TKXSBase )
endif()


#find_package( OpenCV REQUIRED )
# can not install libopencv-dev for confliction, here is the hack on ubuntu 20.04
# ubuntu 18.04 opencv version 3.2 with folder name /usr/include/opencv2
# install libopencv-core-dev  libopencv-imgcodecs-dev
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4"  "/usr/include/opencv2")
set(OpenCV_LIBS "opencv_core" "opencv_imgcodecs")

include_directories( ${OpenCV_INCLUDE_DIRS} )

INCLUDE_DIRECTORIES(${OCC_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(.)
#INCLUDE_DIRECTORIES(catch)


ADD_EXECUTABLE(occProjector occProjector.cpp)

TARGET_LINK_LIBRARIES(occProjector  ${OCC_LINK_LIBS})
#TARGET_LINK_LIBRARIES(occProjector ${CGAL_LIBRARY})
target_link_libraries(occProjector Eigen3::Eigen)
target_link_libraries(occProjector ${OpenCV_LIBS})

########################################################
if(EXISTS "occ_inscribe.cpp")

ADD_EXECUTABLE(occ_inscribe occ_inscribe.cpp)
target_compile_definitions(occ_inscribe PUBLIC "-DBUILD_INSCRIBE")
TARGET_LINK_LIBRARIES(occ_inscribe  ${OCC_LINK_LIBS})
target_link_libraries(occ_inscribe Eigen3::Eigen)

endif()