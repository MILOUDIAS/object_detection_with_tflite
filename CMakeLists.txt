cmake_minimum_required(VERSION 3.20)
project(TFLiteProg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# fetch latest argparse
# include(FetchContent)
# FetchContent_Declare(
#     argparse
#     GIT_REPOSITORY https://github.com/p-ranav/argparse.git
# )
# FetchContent_MakeAvailable(argparse)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ARGPARSE REQUIRED argparse)
# Add the include directories for argparse
include_directories(${ARGPARSE_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/tensorflow-lite/include/)
add_executable(${PROJECT_NAME} src/main.cpp)

ADD_LIBRARY(tensorflowlite SHARED IMPORTED GLOBAL)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/vendor/tensorflow-lite/libs/linux_x86_64/libtensorflowlite.so)
# target_link_libraries(${PROJECT_NAME} tensorflowlite ${OpenCV_LIBS} ${ARGPARSE_LIBRARIES})
target_link_libraries(${PROJECT_NAME} tensorflowlite ${OpenCV_LIBS})
