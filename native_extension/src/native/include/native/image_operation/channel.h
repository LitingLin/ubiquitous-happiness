#pragma once

#include <cstdint>
#include <native/common.h>

// Aligned
NATIVE_INTERFACE
void RGBImageHWCToCHW(const uint8_t* input, unsigned width, unsigned height, uint8_t* output);
NATIVE_INTERFACE
void RGBImageHWCToCHW_unaligned(const uint8_t* input, unsigned width, unsigned height, uint8_t* output);

NATIVE_INTERFACE
void RGBImageHWCToCHW_optimized(const uint8_t* input, unsigned width, unsigned height, uint8_t* output);

NATIVE_INTERFACE
void RGBFloatImageHWCToCHW_unaligned(const float* input, unsigned width, unsigned height, float* output);
NATIVE_INTERFACE
void RGBFloatImageHWCToCHW(const float* input, unsigned width, unsigned height, float* output);

NATIVE_INTERFACE
void RGBImageToGrayScale(const uint8_t* input, unsigned width, unsigned height, uint8_t* output);