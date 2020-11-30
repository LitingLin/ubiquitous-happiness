#include <py_module/image_decoder.h>

#include <base/utils.h>
#include <base/file.h>
#include <base/memory_mapped_io.h>

#include <native/image_decoder.h>
#include <base/logging.h>

namespace PyModule {
	void ImageDecoder::load(std::string_view path)
	{
		if (PYBIND11_BYTES_SIZE(_cache.ptr()))
			_cache = pybind11::bytes();
		_decoder.initialize(path);
	}

	void ImageDecoder::load(pybind11::bytes bytes, Base::ImageFormatType type)
	{
		_cache = bytes;

		char* buffer;
		ssize_t length;
		L_CHECK(!PYBIND11_BYTES_AS_STRING_AND_SIZE(_cache.ptr(), &buffer, &length));
		
		_decoder.initialize(buffer, length, type);
	}

	void ImageDecoder::close()
	{
		if (PYBIND11_BYTES_SIZE(_cache.ptr()))
			_cache = pybind11::bytes();
		_decoder.close();
	}

	size_t ImageDecoder::width() const
	{
		return _decoder.getWidth();
	}

	size_t ImageDecoder::height() const
	{
		return _decoder.getHeight();
	}

	pybind11::array_t<uint8_t> ImageDecoder::decode()
	{
		std::vector<ssize_t> shape = { (ssize_t)_decoder.getHeight(), (ssize_t)_decoder.getWidth(), 3 };
		pybind11::array_t<uint8_t> decompressedImage(shape);
		_decoder.decode(decompressedImage.mutable_data());
		return decompressedImage;
	}

	pybind11::array_t<uint8_t> ImageDecoder::decode(std::string_view path)
	{
		load(path);
		return decode();
	}

	pybind11::array_t<uint8_t> ImageDecoder::decode(pybind11::bytes bytes, Base::ImageFormatType type)
	{
		load(bytes, type);
		return decode();
	}

	bool ImageDecoder::isLoaded() const
	{
		return _decoder.isOpen();
	}
}
