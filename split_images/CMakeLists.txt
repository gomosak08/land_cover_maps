cmake_minimum_required(VERSION 3.10)
project(tile_raster CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(GDAL REQUIRED)
add_subdirectory(cnpy)

add_executable(tile_raster main.cpp)

# 👇 añade esta línea
target_include_directories(tile_raster PRIVATE ${CMAKE_SOURCE_DIR}/cnpy)

target_link_libraries(tile_raster GDAL::GDAL cnpy)
