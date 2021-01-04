#include <py_module/image_operation.h>

#include <native/image_operation/translate_and_scale.h>

namespace PyModule {
	template <typename DataType>
	void checkRGBImageArgument(const pybind11::array_t<DataType, pybind11::array::c_style | pybind11::array::forcecast> &input)
	{
		if (input.ndim() != 3)
			throw std::runtime_error("expected dimension of array is 3");
		if (input.shape(2) != 3)
			throw std::runtime_error("channels of image should be 3");
		if (input.shape(0) <= 0 || input.shape(1) <= 0)
			throw std::runtime_error("invalid shape of image");
		if (input.shape(0) > std::numeric_limits<unsigned>::max() || input.shape(1) > std::numeric_limits<unsigned>::max())
			throw std::runtime_error("image too large");
	}

	template <typename DataType>
	unsigned getImageWidth(const pybind11::array_t<DataType, pybind11::array::c_style | pybind11::array::forcecast> &input)
	{
		if (input.shape(1) > std::numeric_limits<unsigned>::max())
			throw std::runtime_error("the shape of image array exceed the capability of processing library");

		return unsigned(input.shape(1));
	}

	template <typename DataType>
	unsigned getImageHeight(const pybind11::array_t<DataType, pybind11::array::c_style | pybind11::array::forcecast>& input)
	{
		if (input.shape(0) > std::numeric_limits<unsigned>::max())
			throw std::runtime_error("the shape of image array exceed the capability of processing library");

		return unsigned(input.shape(0));
	}

    std::pair<pybind11::array_t<uint8_t>, pybind11::array_t<int32_t>> RGBImageTranslateAndScaleWithBoundingBox(const pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast>& input, std::array<uint32_t, 2> outputSize, std::array<double, 2> inputCenter, std::array<double, 2> outputCenter,
		std::array<double, 2> scaleRatio, std::array<uint8_t, 3> backgroundColor, InterpolationMethod method)
	{
		checkRGBImageArgument(input);

		std::vector<ssize_t> outputShape = { outputSize[1], outputSize[0], 3 };
		pybind11::array_t<uint8_t> output(outputShape);

        pybind11::array_t<int32_t> outputBoundingBox({4});
		
		::RGBImageTranslateAndScale(input.data(), getImageWidth(input), getImageHeight(input), output.mutable_data(), outputSize[0], outputSize[1],
            outputBoundingBox.mutable_data(), inputCenter[0], inputCenter[1],
			outputCenter[0], outputCenter[1], scaleRatio[0], scaleRatio[1], backgroundColor, method);
		return std::make_pair(output, outputBoundingBox);
	}

    pybind11::array_t<uint8_t> RGBImageTranslateAndScale(const pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast>& input, std::array<uint32_t, 2> outputSize, std::array<double, 2> inputCenter, std::array<double, 2> outputCenter,
                                                                                                                std::array<double, 2> scaleRatio, std::array<uint8_t, 3> backgroundColor, InterpolationMethod method)
    {
        checkRGBImageArgument(input);

        std::vector<ssize_t> outputShape = { outputSize[1], outputSize[0], 3 };
        pybind11::array_t<uint8_t> output(outputShape);

        ::RGBImageTranslateAndScale(input.data(), getImageWidth(input), getImageHeight(input), output.mutable_data(), outputSize[0], outputSize[1],
                                    nullptr, inputCenter[0], inputCenter[1],
                                    outputCenter[0], outputCenter[1], scaleRatio[0], scaleRatio[1], backgroundColor, method);
        return output;
    }

	pybind11::array_t<uint8_t> RGBImageToGrayScale(const pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast>& input)
	{
		checkRGBImageArgument(input);

		pybind11::array_t<uint8_t> output(std::vector<ssize_t>(input.shape(), input.shape() + 3));
		::RGBImageToGrayScale(input.data(), getImageWidth(input), getImageHeight(input), output.mutable_data());

		return output;
	}
}
