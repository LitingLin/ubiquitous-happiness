cmake_minimum_required(VERSION 3.13)
project(ms_deform_atten)

set(PYTHON_MODULE_NAME _C)

set(SRC_FILES
    "${CMAKE_CURRENT_LIST_DIR}/ms_deform_attn.h"
    "${CMAKE_CURRENT_LIST_DIR}/vision.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/cpu/ms_deform_attn_cpu.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/cpu/ms_deform_attn_cpu.cpp"
    )
set(CUDA_SRC_FILES
    "${CMAKE_CURRENT_LIST_DIR}/cuda/ms_deform_attn_cuda.cu"
    "${CMAKE_CURRENT_LIST_DIR}/cuda/ms_deform_attn_cuda.h"
    "${CMAKE_CURRENT_LIST_DIR}/cuda/ms_deform_im2col_cuda.cuh"
    )

find_package(pybind11 REQUIRED)

Python3_add_library(${PYTHON_MODULE_NAME} MODULE ${SRC_FILES} ${CUDA_SRC_FILES})

if (WIN32)
    target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE _WINDLL)
endif()

target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${EXTRA_INCLUDE_PATHS} ${pybind11_INCLUDE_DIRS})
target_link_directories(${PYTHON_MODULE_NAME} PRIVATE ${EXTRA_LIBRARY_PATHS})
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE ${Python3_LIBRARIES})
