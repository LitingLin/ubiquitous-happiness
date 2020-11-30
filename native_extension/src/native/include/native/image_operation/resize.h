#pragma once

#include <stdint.h>

enum class InterpolationMethod : uint32_t
{
	INTER_NEAREST = 0,
	INTER_LINEAR = 1,
	INTER_CUBIC = 2,
	INTER_AREA = 3,
	INTER_LANCZOS4 = 4
};

int toCVValue(InterpolationMethod method);