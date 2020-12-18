project(py_interop)

set(SRC_FILES
        "${CMAKE_CURRENT_LIST_DIR}/include/py_module/image_operation.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/py_module/common.h"

        "${CMAKE_CURRENT_LIST_DIR}/src/py_module/image_operation.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/py_module/module_def.cpp")

if (${ENABLE_BASE_LIBS})
    list(APPEND SRC_FILES
        "${CMAKE_CURRENT_LIST_DIR}/include/py_module/image_decoder.h"
        "${CMAKE_CURRENT_LIST_DIR}/src/py_module/image_decoder.cpp")
endif()

set(PYTHON_MODULE_NAME _C)

#find_package(pybind11 REQUIRED)
Python3_add_library(${PYTHON_MODULE_NAME} MODULE ${SRC_FILES})
if (WIN32)
    target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE _WINDLL)
endif()
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/include/)
target_link_directories(${PYTHON_MODULE_NAME} PRIVATE ${Python3_LIBRARY_DIRS} ${Python3_RUNTIME_LIBRARY_DIRS})
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE ${PYNATIVE_LINK_LIBRARIES} native ${Python3_LIBRARIES})

if (${RELEASE_INSTALL})
    install(TARGETS ${PYTHON_MODULE_NAME}
            LIBRARY DESTINATION .)
else()
    install(TARGETS ${PYTHON_MODULE_NAME}
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif()