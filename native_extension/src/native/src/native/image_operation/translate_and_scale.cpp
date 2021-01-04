#include <native/image_operation/translate_and_scale.h>

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <native/statistic/iou.h>

double makeInRange(double value, double min, double max)
{
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

void RGBImageTranslateAndScale(const uint8_t* inputImage, uint32_t inputImageWidth, uint32_t inputImageHeight,
    uint8_t* outputImage, uint32_t outputImageWidth, uint32_t outputImageHeight,
	int32_t* outputBoundingBox,
	double inputCenterX, double inputCenterY, double outputCenterX, double outputCenterY,
	double scaleRatioX, double scaleRatioY,
	std::array<uint8_t, 3> backgroundColor, InterpolationMethod method)
{
	const cv::Mat inputMat(inputImageHeight, inputImageWidth, CV_8UC3, (void*)inputImage);
	cv::Mat outputMat(outputImageHeight, outputImageWidth, CV_8UC3, (void*)outputImage);

	outputMat.setTo(cv::Vec3b(backgroundColor[0], backgroundColor[1], backgroundColor[2]));

	double outputOffset1X = -outputCenterX, outputOffset1Y = -outputCenterY, outputOffset2X = double(outputImageWidth) - outputCenterX, outputOffset2Y = double(outputImageHeight) - outputCenterY;
	double inputOffset1X = outputOffset1X / scaleRatioX, inputOffset1Y = outputOffset1Y / scaleRatioY, inputOffset2X = outputOffset2X / scaleRatioX, inputOffset2Y = outputOffset2Y / scaleRatioY;

	double inputBoundingBox1X = inputCenterX + inputOffset1X, inputBoundingBox1Y = inputCenterY + inputOffset1Y, inputBoundingBox2X = inputCenterX + inputOffset2X, inputBoundingBox2Y = inputCenterY + inputOffset2Y;

	double maxInputX = double(inputImageWidth);
	double maxInputY = double(inputImageHeight);
	if (computeIOU({ inputBoundingBox1X, inputBoundingBox1Y, inputBoundingBox2X - inputBoundingBox1X, inputBoundingBox2Y - inputBoundingBox1Y },
		{ 0., 0., maxInputX, maxInputY }) <= 0.)
		return;

	double inInputImageBoundingBox1X = makeInRange(inputBoundingBox1X, 0., std::nextafter(maxInputX, maxInputX - 1.));
	double inInputImageBoundingBox1Y = makeInRange(inputBoundingBox1Y, 0., std::nextafter(maxInputY, maxInputY - 1.));
	double inInputImageBoundingBox2X = makeInRange(inputBoundingBox2X, std::nextafter(0., 1.), maxInputX);
	double inInputImageBoundingBox2Y = makeInRange(inputBoundingBox2Y, std::nextafter(0., 1.), maxInputY);

	if (inInputImageBoundingBox2X - inInputImageBoundingBox1X < 1. || inInputImageBoundingBox2Y - inInputImageBoundingBox1Y < 1.)
		return;

	double relocatedOutputOffset1X = outputOffset1X - (inputBoundingBox1X - inInputImageBoundingBox1X) * scaleRatioX;
	double relocatedOutputOffset1Y = outputOffset1Y - (inputBoundingBox1Y - inInputImageBoundingBox1Y) * scaleRatioX;
	double relocatedOutputOffset2X = outputOffset2X - (inputBoundingBox2X - inInputImageBoundingBox2X) * scaleRatioY;
	double relocatedOutputOffset2Y = outputOffset2Y - (inputBoundingBox2Y - inInputImageBoundingBox2Y) * scaleRatioY;

	double inOutputImageBoundingBox1X = outputCenterX + relocatedOutputOffset1X;
	double inOutputImageBoundingBox1Y = outputCenterY + relocatedOutputOffset1Y;
	double inOutputImageBoundingBox2X = outputCenterX + relocatedOutputOffset2X;
	double inOutputImageBoundingBox2Y = outputCenterY + relocatedOutputOffset2Y;

	if (inOutputImageBoundingBox2X - inOutputImageBoundingBox1X < 1. || inOutputImageBoundingBox2Y - inOutputImageBoundingBox1Y < 1.)
		return;

	int quantizedInputImageBoundingBox1X = int(std::round(inInputImageBoundingBox1X));
	int quantizedInputImageBoundingBox1Y = int(std::round(inInputImageBoundingBox1Y));
	int quantizedInputImageBoundingBox2X = int(std::round(inInputImageBoundingBox2X));
	int quantizedInputImageBoundingBox2Y = int(std::round(inInputImageBoundingBox2Y));

	int quantizedOutputImageBoundingBox1X = int(std::round(inOutputImageBoundingBox1X));
	int quantizedOutputImageBoundingBox1Y = int(std::round(inOutputImageBoundingBox1Y));
	int quantizedOutputImageBoundingBox2X = int(std::round(inOutputImageBoundingBox2X));
	int quantizedOutputImageBoundingBox2Y = int(std::round(inOutputImageBoundingBox2Y));

	int inputCropWidth = quantizedInputImageBoundingBox2X - quantizedInputImageBoundingBox1X;
	int inputCropHeight = quantizedInputImageBoundingBox2Y - quantizedInputImageBoundingBox1Y;
	int outputCropWidth = quantizedOutputImageBoundingBox2X - quantizedOutputImageBoundingBox1X;
	int outputCropHeight = quantizedOutputImageBoundingBox2Y - quantizedOutputImageBoundingBox1Y;

	const cv::Mat inputCrop = inputMat(cv::Rect(quantizedInputImageBoundingBox1X, quantizedInputImageBoundingBox1Y, inputCropWidth, inputCropHeight));
	cv::Mat outputCrop = outputMat(cv::Rect(quantizedOutputImageBoundingBox1X, quantizedOutputImageBoundingBox1Y, outputCropWidth, outputCropHeight));

	if (inputCrop.cols == outputCrop.cols && inputCrop.rows == outputCrop.rows)
		inputCrop.copyTo(outputCrop);
	else
		cv::resize(inputCrop, outputCrop, cv::Size(outputCropWidth, outputCropHeight), 0, 0, toCVValue(method));

	if (outputBoundingBox) {
        outputBoundingBox[0] = quantizedOutputImageBoundingBox1X;
        outputBoundingBox[1] = quantizedOutputImageBoundingBox1Y;
        outputBoundingBox[2] = outputCropWidth;
        outputBoundingBox[3] = outputCropHeight;
    }
}
