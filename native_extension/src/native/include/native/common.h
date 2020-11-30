#pragma once

#if defined _WIN32
#if defined COMPILING_NATIVE && defined _WINDLL
#define NATIVE_INTERFACE __declspec(dllexport)
#else
#define NATIVE_INTERFACE
#endif
#else
#ifdef COMPILING_NATIVE
#define NATIVE_INTERFACE __attribute__ ((visibility ("default")))
#else
#define NATIVE_INTERFACE
#endif
#endif
