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
if(TRUE)
  FIND_PACKAGE(OCC QUIET)

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
  SET(OCC_INCLUDE_DIRS "/mnt/windata/MyRepo/opencascade-7.5.0/build/include/opencascade")
  link_directories("/mnt/windata/MyRepo/opencascade-7.5.0/build/lin64/gcc/libd")  # debug `libd`
  SET(OCC_LINK_LIBS TKernel TKMath  TKTopAlgo TKG3d TKBRep 
                  TKMesh TKRWMesh TKSTL TKXMesh TKBO TKBool TKSTEPBase TKSTEP TKXSBase )
endif()



INCLUDE_DIRECTORIES(${OCC_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(.)
#INCLUDE_DIRECTORIES(catch)


ADD_EXECUTABLE(occProjector occProjector.cpp)

TARGET_LINK_LIBRARIES(occProjector  ${OCC_LINK_LIBS})
#TARGET_LINK_LIBRARIES(occProjector ${CGAL_LIBRARY})
target_link_libraries(occProjector Eigen3::Eigen)


########################################################
ADD_EXECUTABLE(occ_inscribe occ_inscribe.cpp)
target_compile_definitions(occ_inscribe PUBLIC "-DBUILD_INSCRIBE")
TARGET_LINK_LIBRARIES(occ_inscribe  ${OCC_LINK_LIBS})
target_link_libraries(occ_inscribe Eigen3::Eigen)