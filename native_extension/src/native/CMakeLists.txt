cmake_minimum_required(VERSION 3.1)

project(native)

set(SRC_FILES
        "${CMAKE_CURRENT_LIST_DIR}/include/native/image_operation/resize.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/native/image_operation/translate_and_scale.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/native/statistic/iou.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/native/common.h"

        "${CMAKE_CURRENT_LIST_DIR}/src/native/image_operation/translate_and_scale.cpp"
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

target_compile_definitions(native PUBLIC ${PYNATIVE_COMPILE_DEFINITIONS})
if (WIN32)
    target_compile_definitions(native PRIVATE _WINDLL)
endif()
target_compile_definitions(native PRIVATE COMPILING_NATIVE)
target_include_directories(native PRIVATE ${PYNATIVE_INCLUDE_DIRECTORIES})
target_link_directories(native PRIVATE ${PYNATIVE_LINK_DIRECTORIES})
target_link_libraries(native ${PYNATIVE_LINK_LIBRARIES})

if (NOT ${RELEASE_INSTALL})
    install(TARGETS native
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif()
