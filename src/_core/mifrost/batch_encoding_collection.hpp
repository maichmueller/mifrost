#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace mifrost {

nb::object
batch_encodings_from_sequence(nb::sequence encodings, nb::object collate_spec_obj, bool fast_path);

}  // namespace mifrost
