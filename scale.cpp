// scale.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   

#include <vector>

std::vector<float> scale(const std::vector<float>& input, float factor) {
    std::vector<float> output;
    output.reserve(input.size());
    for (float value : input) {
        output.push_back(value * factor);
    }
    return output;
}

PYBIND11_MODULE(fast_scale, module) {
    module.def("scale", &scale,
               pybind11::arg("input"),
               pybind11::arg("factor"));
}
