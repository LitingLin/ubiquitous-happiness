#pragma once

#include <py_module/common.h>
#include <pybind11/numpy.h>
#include <native/image_operation/resize.h>

namespace PyModule
{
	PYTHON_MODULE_INTERFACE
	pybind11::array_t<uint8_t> RGBImageTranslateAndScale(const pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast>& input, std::array<uint32_t, 2> outputSize, std::array<double, 2> inputCenter, std::array<double, 2> outputCenter,
		std::array<double, 2> scaleRatio, std::array<uint8_t, 3> backgroundColor, InterpolationMethod method);
}