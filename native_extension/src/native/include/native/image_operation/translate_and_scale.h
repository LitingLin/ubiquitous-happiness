#pragma once

#include <native/common.h>
#include <native/image_operation/resize.h>
#include <stdint.h>
#include <array>

NATIVE_INTERFACE
void RGBImageTranslateAndScale(const uint8_t* inputImage, uint32_t inputImageWidth, uint32_t inputImageHeight,
    uint8_t* outputImage, uint32_t outputImageWidth, uint32_t outputImageHeight,
    uint32_t* outputBoundingBox,
	double inputCenterX, double inputCenterY, double outputCenterX, double outputCenterY,
	double scaleRatioX, double scaleRatioY,
	std::array<uint8_t, 3> backgroundColor, InterpolationMethod method = InterpolationMethod::INTER_LINEAR);
