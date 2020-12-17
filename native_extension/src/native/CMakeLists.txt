cmake_minimum_required(VERSION 3.1)

project(native)

find_package(OpenCV REQUIRED core imgproc)
find_package(Eigen3 REQUIRED NO_MODULE)

set(SRC_FILES
        "${CMAKE_CURRENT_LIST_DIR}/include/native/image_operation/resize.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/native/image_operation/translate_and_scale.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/native/image_operation/channel.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/native/statistic/iou.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/native/common.h"

        "${CMAKE_CURRENT_LIST_DIR}/src/native/image_operation/translate_and_scale.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/native/image_operation/channel.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/native/image_operation/resize.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/native/statistic/iou.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/native/dll_entry.cpp"
        )

if (${ENABLE_BASE_LIBS})
    list(APPEND SRC_FILES
        "${CMAKE_CURRENT_LIST_DIR}/include/native/image_decoder.h"
        "${CMAKE_CURRENT_LIST_DIR}/src/native/image_decoder.cpp")
endif()

if (${BUILD_SHARED_LIBS})
    list(APPEND SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/src/native/dll_entry.cpp")
    add_library(native SHARED ${SRC_FILES})
else()
    add_library(native STATIC ${SRC_FILES})
endif()

if (WIN32)
    target_compile_definitions(native PRIVATE _WINDLL)
endif()
target_compile_definitions(native PRIVATE COMPILING_NATIVE)

target_link_libraries(native PUBLIC ${OpenCV_LIBS})

target_include_directories(native PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR})

if (${ENABLE_BASE_LIBS})
    target_link_libraries(native PUBLIC base)
    target_compile_definitions(native PUBLIC ENABLE_BASE_LIBS)
endif()

if (NOT ${RELEASE_INSTALL})
    install(TARGETS native
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif()
