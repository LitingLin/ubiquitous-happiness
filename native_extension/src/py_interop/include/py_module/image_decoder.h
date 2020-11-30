#pragma once

#include <py_module/common.h>
#include <native/image_decoder.h>
#include <pybind11/numpy.h>

namespace PyModule {
	class PYTHON_MODULE_INTERFACE ImageDecoder
	{
	public:
		void load(std::string_view path);
		void load(pybind11::bytes bytes, Base::ImageFormatType type);
		void close();
		size_t width() const;
		size_t height() const;
		pybind11::array_t<uint8_t> decode();
		pybind11::array_t<uint8_t> decode(std::string_view path);
		pybind11::array_t<uint8_t> decode(pybind11::bytes bytes, Base::ImageFormatType type);
		bool isLoaded() const;
	private:
		::ImageDecoder _decoder;
		pybind11::bytes _cache;
	};
}