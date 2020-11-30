#include <native/image_operation/resize.h>

#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

int toCVValue(InterpolationMethod method)
{
	switch (method)
	{
		case InterpolationMethod::INTER_NEAREST:
			return cv::INTER_NEAREST;
		case InterpolationMethod::INTER_LINEAR:
			return cv::INTER_LINEAR;
		case InterpolationMethod::INTER_CUBIC:
			return cv::INTER_CUBIC;
		case InterpolationMethod::INTER_AREA:
			return cv::INTER_AREA;
		case InterpolationMethod::INTER_LANCZOS4:			
			return cv::INTER_LANCZOS4;
		default:
			throw std::runtime_error("unsupported value");
	}
}