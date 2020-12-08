cmake_minimum_required(VERSION 3.13)

project(native_extension)

set(default_build_type "RelWithDebInfo")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Python3 COMPONENTS Interpreter Development)

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "if True:
    import sys, os
    print(os.path.exists(os.path.join(sys.prefix, 'conda-meta')))"
        OUTPUT_VARIABLE IS_ANACONDA_DISTRIBUTION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RETVAL
)

if (NOT ${RETVAL} EQUAL 0)
    message(FATAL_ERROR "Failed to get python version. Pythom executable: ${Python3_EXECUTABLE}")
endif()

if(${IS_ANACONDA_DISTRIBUTION_STRING} STREQUAL "True")
    set(IS_ANACONDA_DISTRIBUTION TRUE)
else()
    set(IS_ANACONDA_DISTRIBUTION FALSE)
endif()

message(STATUS "Anaconda distribution: ${IS_ANACONDA_DISTRIBUTION}")

if(WIN32)
    get_filename_component(PYTHON_ROOT ${Python3_INCLUDE_DIRS}/../ ABSOLUTE)
else()
    get_filename_component(PYTHON_ROOT ${Python3_INCLUDE_DIRS}/../../ ABSOLUTE)
endif()

message(STATUS "PYTHON_ROOT: ${PYTHON_ROOT}")

if(${IS_ANACONDA_DISTRIBUTION})
    if (WIN32)
        set(EXTRA_INCLUDE_PATHS ${PYTHON_ROOT}/Library/include ${Python3_INCLUDE_DIRS})
        set(EXTRA_LIBRARY_PATHS ${PYTHON_ROOT}/Library/lib ${PYTHON_ROOT}/libs)
        set(EXTRA_ROOT_PATHS ${PYTHON_ROOT} ${PYTHON_ROOT}/Library/)
    else()
        set(EXTRA_INCLUDE_PATHS ${Python3_INCLUDE_DIRS})
        set(EXTRA_LIBRARY_PATHS ${PYTHON_ROOT}/lib)
        set(EXTRA_ROOT_PATHS ${PYTHON_ROOT})
    endif()
endif()

set(EXTRA_LIBRARIES ${Python3_LIBRARIES})

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "if True:
    from distutils import sysconfig as sc
    print(sc.get_python_lib(prefix='', plat_specific=True))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RETVAL)

if (NOT ${RETVAL} EQUAL "0")
    message(FATAL_ERROR "Failed to get python site package path. Pythom executable: ${Python3_EXECUTABLE}")
endif()

set(PYTHON_SITE_PACKAGE_PATH ${PYTHON_ROOT}/${PYTHON_SITE_PACKAGE_PATH})
message(STATUS "PYTHON_SITE_PACKAGE_PATH:${PYTHON_SITE_PACKAGE_PATH}")

set(CMAKE_PREFIX_PATH ${EXTRA_ROOT_PATHS} ${CMAKE_PREFIX_PATH})

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "if True:
    from torch.utils.cpp_extension import _get_cuda_arch_flags
    print(' '.join(_get_cuda_arch_flags()))"
        OUTPUT_VARIABLE PYTORCH_CUDA_ARCH_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RETVAL)

if (NOT ${RETVAL} EQUAL "0")
    message(FATAL_ERROR "Torch not compiled with CUDA enabled")
endif()

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "if True:
    from torch.utils.cpp_extension import COMMON_NVCC_FLAGS
    print(' '.join(COMMON_NVCC_FLAGS))"
        OUTPUT_VARIABLE PYTORCH_COMMON_NVCC_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RETVAL)

if (NOT ${RETVAL} EQUAL "0")
    message(FATAL_ERROR "Torch not compiled with CUDA enabled")
endif()

enable_language(CUDA)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${PYTORCH_CUDA_ARCH_FLAGS}")

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "if True:
    from torch.utils.cpp_extension import include_paths
    print(';'.join(include_paths(True)))"
        OUTPUT_VARIABLE PYTORCH_INCLUDE_PATHS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RETVAL)

if (NOT ${RETVAL} EQUAL "0")
    message(FATAL_ERROR "Failed to get torch include paths")
endif()

list(APPEND EXTRA_INCLUDE_PATHS ${PYTORCH_INCLUDE_PATHS})

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "if True:
    from torch.utils.cpp_extension import library_paths
    print(';'.join(library_paths(True)))"
        OUTPUT_VARIABLE PYTORCH_LIBRARY_PATHS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RETVAL)

if (NOT ${RETVAL} EQUAL "0")
    message(FATAL_ERROR "Failed to get torch library paths")
endif()

list(APPEND EXTRA_LIBRARY_PATHS ${PYTORCH_LIBRARY_PATHS})

find_package(Torch REQUIRED PATHS ${PYTHON_SITE_PACKAGE_PATH}/torch/share/cmake/Torch)
list(APPEND EXTRA_LIBRARIES ${TORCH_LIBRARIES})

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}" "${EXTRA_LIBRARY_PATHS}")

if (NOT WIN32)
    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "if True:
        from distutils import sysconfig as sc
        print(sc.get_config_var('CFLAGS'))"
            OUTPUT_VARIABLE PYTHON_CFLAGS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RETVAL)

    if (NOT ${RETVAL} EQUAL "0")
        message(FATAL_ERROR "Failed to get python C compiler flags. Pythom executable: ${Python3_EXECUTABLE}")
    endif()

    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "if True:
        from distutils import sysconfig as sc
        print(sc.get_config_var('LDFLAGS'))"
            OUTPUT_VARIABLE PYTHON_LDFLAGS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RETVAL)

    if (NOT ${RETVAL} EQUAL "0")
        message(FATAL_ERROR "Failed to get python linker flags. Pythom executable: ${Python3_EXECUTABLE}")
    endif()

    set(CMAKE_C_FLAGS "${PYTHON_CFLAGS}")
    set(CMAKE_CXX_FLAGS "${PYTHON_CFLAGS}")
    string(REPLACE "-Wstrict-prototypes" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-ffast-math")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-ffast-math")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined -pthread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined -pthread")
endif()
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING /wd4251 /wd4275 /Zc:referenceBinding /Zc:throwingNew /Zc:strictStrings /Zc:inline /Zc:rvalueCast")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /O2 /Ob2 /GL /GS- /Oi /Zi")
    set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_C_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    set(CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO "/LTCG")
    set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "/MANIFEST /LTCG:incremental /DEBUG:FULL /OPT:REF /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\" /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "/MANIFEST /LTCG:incremental /DEBUG:FULL /OPT:REF /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\" /OPT:ICF")
endif()

add_subdirectory(src)
