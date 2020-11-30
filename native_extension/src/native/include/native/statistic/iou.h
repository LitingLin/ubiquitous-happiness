#pragma once

#include <array>
#include <native/common.h>
#include <memory>

NATIVE_INTERFACE
std::array<double, 4> getIntersectionArea(const std::array<double, 4> & boundingBoxA, const std::array<double, 4> & boundingBoxB);

// x, y, w, h
NATIVE_INTERFACE
double computeIOU(const std::array<double, 4>& boundingBoxA, const std::array<double, 4>& boundingBoxB);
