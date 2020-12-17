cmake_minimum_required(VERSION 3.13)

project(native_extension)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
option(SINGLE_MODULE "Build as a single module" ON)
option(ENABLE_AVX2 "Enable AVX2 instruction set" ON)
option(ENABLE_BASE_LIBS "Enable base-libs to enable directly access to os-specific APIs" ON)

if (${SINGLE_MODULE})
    SET(BUILD_SHARED_LIB OFF CACHE BOOL "Build all module as shared lib")
endif()

find_package(Python3 COMPONENTS Interpreter Development)

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

    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -flto")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -flto")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -flto")
    set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} -flto")
    set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -flto")

    if (${ENABLE_AVX2})
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -mavx2")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -mavx2")
    endif()
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

    if (${ENABLE_AVX2})
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /arch:AVX2")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /arch:AVX2")
    endif()
endif()

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

SET(RELEASE_INSTALL ON CACHE BOOL "install binary files only")

if (${ENABLE_BASE_LIBS})
    add_subdirectory(base-libs)
endif()

add_subdirectory(native)
add_subdirectory(py_interop)
