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

option(SINGLE_MODULE "Build as a single module" ON)
option(ENABLE_AVX2 "Enable AVX2 instruction set" ON)
option(ENABLE_BASE_LIBS "Enable base-libs to enable directly access to os-specific APIs" ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
if (${SINGLE_MODULE})
    SET(BUILD_SHARED_LIB OFF CACHE BOOL "Build all module as shared lib")
endif()

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
message(STATUS "EXTRA_ROOT_PATHS:${EXTRA_ROOT_PATHS}")

set(CMAKE_PREFIX_PATH ${EXTRA_ROOT_PATHS} ${CMAKE_PREFIX_PATH})

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

if (NOT WIN32)
    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "if True:
        from distutils import sysconfig as sc
        print(sc.get_config_var('CFLAGS'), end='')"
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

    if (NOT "${PYTHON_CFLAGS}" STREQUAL "")
        set(CMAKE_C_FLAGS "${PYTHON_CFLAGS}")
        set(CMAKE_CXX_FLAGS "${PYTHON_CFLAGS}")
        string(REPLACE "-Wstrict-prototypes" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
    endif()
    if (NOT "${PYTHON_LDFLAGS}" STREQUAL "")
        set(CMAKE_EXE_LINKER_FLAGS "${PYTHON_LDFLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${PYTHON_LDFLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${PYTHON_LDFLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "")
        set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "")
        set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "")
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_EXTRA_NVCC_FLAGS}")

    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -flto")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -flto")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -flto")
    set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} -flto")
    set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -flto")
endif()
if (MSVC)
    set(COMMON_MSVC_FLAGS "/wd4251 /wd4275 /Zc:referenceBinding /Zc:throwingNew /Zc:strictStrings /Zc:inline /Zc:rvalueCast")
    set(COMMON_MSVC_CXX_FLAGS "/GR /EHsc")
    set(COMMON_MSVC_RELWITHDEBINFO "/O2 /Ob2 /GL /GS- /Oi /Zi")
    set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS ${COMMON_MSVC_FLAGS} ${COMMON_MSVC_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/DNDEBUG ${COMMON_MSVC_RELWITHDEBINFO}")
    set(CMAKE_C_FLAGS "/DWIN32 /D_WINDOWS ${COMMON_MSVC_FLAGS}")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "/DNDEBUG ${COMMON_MSVC_RELWITHDEBINFO}")
    set(CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO "/LTCG")
    set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "/MANIFEST /LTCG:incremental /DEBUG:FULL /OPT:REF /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\" /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "/MANIFEST /LTCG:incremental /DEBUG:FULL /OPT:REF /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\" /OPT:ICF")
    set(CMAKE_CUDA_FLAGS "-DWIN32 -D_WINDOWS -Xcompiler=\"${COMMON_MSVC_FLAGS} ${COMMON_MSVC_CXX_FLAGS}\" -Xcudafe \"--diag_suppress=base_class_has_different_dll_interface\" -Xcudafe \"--diag_suppress=dll_interface_conflict_dllexport_assumed\" -Xcudafe \"--diag_suppress=field_without_dll_interface\" ${TORCH_EXTRA_NVCC_FLAGS}")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-DNDEBUG -Xcompiler=\"${COMMON_MSVC_RELWITHDEBINFO}\"")
endif()

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}" "${EXTRA_LIBRARY_PATHS}")

SET(RELEASE_INSTALL ON CACHE BOOL "install binary files only")

find_package(OpenCV REQUIRED core imgproc)

set(PYNATIVE_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/native/include/
        ${OpenCV_INCLUDE_DIRS}
        ${EXTRA_INCLUDE_PATHS})

if(MSVC AND ${IS_ANACONDA_DISTRIBUTION})
    string(REPLACE "." "" OPENCV_LIBPOSTFIX ${OpenCV_VERSION})
    set(OpenCV_LIBS "${PYTHON_ROOT}/Library/lib/opencv_core${OPENCV_LIBPOSTFIX}.lib"
            "${PYTHON_ROOT}/Library/lib/opencv_imgproc${OPENCV_LIBPOSTFIX}.lib")
endif()
set(PYNATIVE_LINK_LIBRARIES ${OpenCV_LIBS})

if (${ENABLE_BASE_LIBS})
    add_subdirectory(base-libs)
    list(APPEND PYNATIVE_COMPILE_DEFINITIONS ENABLE_BASE_LIBS)
    list(APPEND PYNATIVE_LINK_LIBRARIES base)
endif()

add_subdirectory(native)
add_subdirectory(py_interop)
