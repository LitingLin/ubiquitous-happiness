cmake_minimum_required(VERSION 3.18)

project(native_extension)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(TORCH_EXTRA_NVCC_FLAGS "" CACHE STRING "Extra nvcc flags")
set(TORCH_CUDA_ARCHS "" CACHE STRING "CUDA ARCHS")
set(TORCH_INCLUDE_PATHS "" CACHE STRING "Extra include paths")
set(TORCH_LIBRARY_PATHS "" CACHE STRING "Extra library paths")
set(TORCH_LIBRARIES "" CACHE STRING "Extra link libraries")

find_package(Python3 COMPONENTS Interpreter Development)

set(CMAKE_CUDA_ARCHITECTURES ${TORCH_CUDA_ARCHS})
enable_language(CUDA)

list(APPEND EXTRA_INCLUDE_PATHS "${TORCH_INCLUDE_PATHS}")
list(APPEND EXTRA_LIBRARY_PATHS "${TORCH_LIBRARY_PATHS}")
list(APPEND EXTRA_LIBRARIES ${TORCH_LIBRARIES})

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}" "${EXTRA_LIBRARY_PATHS}")

message(STATUS "TORCH_CUDA_ARCHS: ${TORCH_CUDA_ARCHS}")
message(STATUS "TORCH_EXTRA_NVCC_FLAGS: ${TORCH_EXTRA_NVCC_FLAGS}")
message(STATUS "TORCH_INCLUDE_PATHS: ${TORCH_INCLUDE_PATHS}")
message(STATUS "TORCH_LIBRARY_PATHS: ${TORCH_LIBRARY_PATHS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

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

add_subdirectory(src)
