cmake_minimum_required(VERSION 3.0)

project(clahe_test)

set (SOURCE_FILES
    main_test.cpp
    clahe.cpp
)

include_directories(
    /home/austin/lib/cv3.2_1/include
)

link_directories(
    /home/austin/lib/cv3.2_1/lib
)

add_executable(clahe_test ${SOURCE_FILES})

target_link_libraries(clahe_test opencv_world)
