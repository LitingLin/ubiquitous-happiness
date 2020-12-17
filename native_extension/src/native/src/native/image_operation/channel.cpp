#include <native/image_operation/channel.h>

#ifdef _MSC_VER
#pragma warning(push, 0)
#elif defined __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#endif
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef _MSC_VER
#pragma warning(pop)
#elif defined __GNUC__
#pragma GCC diagnostic pop
#endif

void RGBImageHWCToCHW(const uint8_t* input, unsigned width, unsigned height, uint8_t* output)
{	
	const Eigen::TensorMap<Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>, Eigen::Aligned> inputTensor((uint8_t*)input, height, width, 3);

	Eigen::TensorMap<Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>, Eigen::Aligned> outputTensor(output, 3, height, width);

	// HWC to CHW
	outputTensor = inputTensor.shuffle(Eigen::array<uint32_t, 3>{2, 0, 1});
}

void RGBImageHWCToCHW_unaligned(const uint8_t* input, unsigned width, unsigned height, uint8_t* output)
{
	const Eigen::TensorMap<Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>, Eigen::Aligned> inputTensor((uint8_t*)input, height, width, 3);

	Eigen::TensorMap<Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>, Eigen::Unaligned> outputTensor(output, 3, height, width);

	// HWC to CHW
	outputTensor = inputTensor.shuffle(Eigen::array<uint32_t, 3>{2, 0, 1});
}

void RGBImageHWCToCHW_optimized(const uint8_t* input, unsigned width, unsigned height, uint8_t* output)
{
	const size_t imageSize = size_t(width) * size_t(height);
	uint8_t* r = output;
	uint8_t* g = r + imageSize;
	uint8_t* b = g + imageSize;

	for (size_t i=0;i< imageSize;++i)
	{
		*r++ = input[0];
		*g++ = input[1];
		*b++ = input[2];
		input+=3;
	}
}

void RGBFloatImageHWCToCHW(const float* input, unsigned width, unsigned height, float* output)
{
	const Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>, Eigen::Aligned> inputTensor((float*)input, height, width, 3);

	Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>, Eigen::Aligned> outputTensor(output, 3, height, width);

	// HWC to CHW
	outputTensor = inputTensor.shuffle(Eigen::array<uint32_t, 3>{2, 0, 1});
}

void RGBFloatImageHWCToCHW_unaligned(const float* input, unsigned width, unsigned height, float* output)
{
	const Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>, Eigen::Aligned> inputTensor((float*)input, height, width, 3);

	Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>, Eigen::Unaligned> outputTensor(output, 3, height, width);

	// HWC to CHW
	outputTensor = inputTensor.shuffle(Eigen::array<uint32_t, 3>{2, 0, 1});
}

void RGBImageToGrayScale(const uint8_t* input, unsigned width, unsigned height, uint8_t* output)
{
	for (size_t i = 0; i < width * height; ++i)
	{
		output[3 * i] = output[3 * i + 1] = output[3 * i + 2] = uint8_t(uint32_t(input[3 * i]) * 299 / 1000 + uint32_t(input[3 * i + 1]) * 587 / 1000 + uint32_t(input[3 * i + 2]) * 114 / 1000);
	}
}
