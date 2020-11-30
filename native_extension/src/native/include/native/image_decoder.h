#pragma once

#include <native/common.h>
#include <base/file.h>
#include <base/memory_mapped_io.h>
#include <base/ext/img_codecs/decoder.h>

NATIVE_INTERFACE
Base::ImageFormatType getImageType(std::string_view path);

class NATIVE_INTERFACE ImageDecoder
{
public:
	ImageDecoder();
	~ImageDecoder();
	void initialize(std::string_view path);
	void initialize(const void* buffer, size_t size, Base::ImageFormatType formatType);	
	void close();
	uint64_t getSize() const;
	unsigned getWidth() const;
	unsigned getHeight() const;
	void decode(void* buffer);
	bool isOpen() const;
private:
	union {
		Base::File _file;
	};
	union {
		Base::MemoryMappedIO _mmap;
	};
	Base::ImageDecoder _decoder;
	bool _fileOpened;
};
