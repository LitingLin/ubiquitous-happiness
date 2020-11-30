#include <native/statistic/iou.h>

#include <cmath>

std::array<double, 4> getIntersectionArea(const std::array<double, 4>& boundingBoxA, const std::array<double, 4>& boundingBoxB)
{
	auto inter_x1 = std::max(boundingBoxA[0], boundingBoxB[0]),
		inter_y1 = std::max(boundingBoxA[1], boundingBoxB[1]);
	auto A_x2 = boundingBoxA[0] + boundingBoxA[2], A_y2 = boundingBoxA[1] + boundingBoxA[3],
		B_x2 = boundingBoxB[0] + boundingBoxB[2], B_y2 = boundingBoxB[1] + boundingBoxB[3];
	auto inter_x2 = std::min(A_x2, B_x2), inter_y2 = std::min(A_y2, B_y2);

	auto inter_w = std::max(0., inter_x2 - inter_x1),
		inter_h = std::max(0., inter_y2 - inter_y1);
	return std::array<double, 4>{inter_x1, inter_y1, inter_w, inter_h};
}

double computeIOU(const std::array<double, 4>& boundingBoxA, const std::array<double, 4>& boundingBoxB)
{
	auto intersectionBoundingBox = getIntersectionArea(boundingBoxA, boundingBoxB);
	auto A_area = boundingBoxA[2] * boundingBoxA[3], B_area = boundingBoxB[2] * boundingBoxB[3];
	auto inter_area = intersectionBoundingBox[2] * intersectionBoundingBox[3];
	auto iou = inter_area / (A_area + B_area - inter_area);
	return iou;
}
