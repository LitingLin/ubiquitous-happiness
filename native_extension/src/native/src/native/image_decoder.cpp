#include <native/image_decoder.h>

#include <base/logging.h>
#include <base/memory_mapped_io.h>
#include <base/utils.h>

#include <cstring>


Base::ImageFormatType getImageType(std::string_view path)
{
	Base::ImageFormatType imageType;

	if (Base::endsWith(path, ".jpg") || Base::endsWith(path, ".JPEG") || Base::endsWith(path, ".JPG") || Base::endsWith(path, ".jpeg"))
		imageType = Base::ImageFormatType::JPEG;
	else if (Base::endsWith(path, ".png") || Base::endsWith(path, ".PNG"))
		imageType = Base::ImageFormatType::PNG;
	else if (Base::endsWith(path, ".webp") || Base::endsWith(path, ".WEBP"))
		imageType = Base::ImageFormatType::WEBP;
	else
		throw std::runtime_error("Unsupported image format");

	return imageType;
}


Base::ImageFormatType getImageType(const void* buffer, size_t size)
{
	uint8_t jpegMagicNumber[] = { 0xFF, 0xD8, 0xFF };
	uint8_t pngMagicNumber[] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
	uint8_t webPMagicNumber[] = { 0x52, 0x49, 0x46, 0x46 };
	if (size > sizeof(jpegMagicNumber) && memcmp(buffer, jpegMagicNumber, sizeof(jpegMagicNumber)) == 0)
		return Base::ImageFormatType::JPEG;
	if (size > sizeof(pngMagicNumber) && memcmp(buffer, pngMagicNumber, sizeof(pngMagicNumber)) == 0)
		return Base::ImageFormatType::PNG;
	if (size > sizeof(webPMagicNumber) && memcmp(buffer, webPMagicNumber, sizeof(webPMagicNumber)) == 0)
		return Base::ImageFormatType::WEBP;

	throw std::runtime_error("Unsupported image format");
}

ImageDecoder::ImageDecoder()
	: _fileOpened(false)
{
}

ImageDecoder::~ImageDecoder()
{
	close();
}

void ImageDecoder::initialize(std::string_view path)
{
	close();

#ifdef _WIN32
	std::wstring utf16Path = Base::UTF8ToUTF16(path);
	new (&_file) Base::File(utf16Path);
#else
	new (&_file) Base::File(path);
#endif
	try {
		new (&_mmap) Base::MemoryMappedIO(&_file);
	}
	catch (...) {
		_file.~File();
		throw;
	}

	_fileOpened = true;

	_decoder.load(_mmap.get(), _file.getSize(), getImageType(_mmap.get(), _file.getSize()));
}

void ImageDecoder::initialize(const void* buffer, size_t size, Base::ImageFormatType formatType)
{
	close();
	
	_decoder.load(buffer, size, formatType);
}

void ImageDecoder::close()
{
	if (_fileOpened)
	{
		_mmap.~MemoryMappedIO();
		_file.~File();

		_fileOpened = false;
	}
}

uint64_t ImageDecoder::getSize() const
{
	return _decoder.getDecompressedSize();
}

unsigned ImageDecoder::getWidth() const
{
	return _decoder.getWidth();
}

unsigned ImageDecoder::getHeight() const
{
	return _decoder.getHeight();
}

void ImageDecoder::decode(void* buffer)
{
	_decoder.decode(buffer);
}

bool ImageDecoder::isOpen() const
{
	return _fileOpened;
}
