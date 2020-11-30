#pragma once

#if defined _WIN32
#if defined COMPILING_PYTHON_MODULE && defined _WINDLL
#define PYTHON_MODULE_INTERFACE __declspec(dllexport)
#else
#define PYTHON_MODULE_INTERFACE
#endif
#else
#ifdef COMPILING_PYTHON_MODULE
#define PYTHON_MODULE_INTERFACE __attribute__ ((visibility ("default")))
#else
#define PYTHON_MODULE_INTERFACE
#endif
#endif
