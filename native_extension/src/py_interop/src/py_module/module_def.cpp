#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <py_module/image_operation.h>
#ifdef ENABLE_BASE_LIBS
#include <py_module/image_decoder.h>
#endif

#include <native/image_operation/resize.h>

PYBIND11_MODULE(_C, m)
{
	m.doc() = "Some native methods";

	pybind11::enum_<InterpolationMethod>(m, "InterpolationMethod")
		.value("INTER_NEAREST", InterpolationMethod::INTER_NEAREST)
		.value("INTER_LINEAR", InterpolationMethod::INTER_LINEAR)
		.value("INTER_CUBIC", InterpolationMethod::INTER_CUBIC)
		.value("INTER_AREA", InterpolationMethod::INTER_AREA)
		.value("INTER_LANCZOS4", InterpolationMethod::INTER_LANCZOS4);

	m.def("RGBImageTranslateAndScale", &PyModule::RGBImageTranslateAndScale);
#ifdef ENABLE_BASE_LIBS
	pybind11::class_<PyModule::ImageDecoder>(m, "ImageDecoder")
		.def(pybind11::init())
		.def("load", pybind11::overload_cast<std::string_view>(&PyModule::ImageDecoder::load))
		.def("load", pybind11::overload_cast<pybind11::bytes, Base::ImageFormatType>(&PyModule::ImageDecoder::load))
		.def("width", &PyModule::ImageDecoder::width)
		.def("height", &PyModule::ImageDecoder::height)
		.def("close", &PyModule::ImageDecoder::close)
		.def("decode", pybind11::overload_cast<>(&PyModule::ImageDecoder::decode))
		.def("decode", pybind11::overload_cast<std::string_view>(&PyModule::ImageDecoder::decode))
		.def("decode", pybind11::overload_cast<pybind11::bytes, Base::ImageFormatType>(&PyModule::ImageDecoder::decode))
		.def("isLoaded", &PyModule::ImageDecoder::isLoaded)
		.def(pybind11::pickle(
			[](const PyModule::ImageDecoder& decoder) {
				if (decoder.isLoaded()) throw std::runtime_error("not support pickling when image loaded");
				return pybind11::make_tuple();
			},
			[](pybind11::tuple t)
				{
				return std::make_unique<PyModule::ImageDecoder>();
				}
		));
#endif
}
