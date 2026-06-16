#pragma once

#include <nanobind/nanobind.h>

#include <string>

namespace nb = nanobind;

namespace mifrost {

nb::object batch_encoding_to_device(nb::handle self, nb::handle device);

nb::object batch_encoding_getattr(nb::handle self, const std::string& key);

void batch_encoding_setattr(nb::handle self, const std::string& key, nb::handle value);

nb::list batch_encoding_keys(nb::handle self);

nb::list batch_encoding_items(nb::handle self);

}  // namespace mifrost
