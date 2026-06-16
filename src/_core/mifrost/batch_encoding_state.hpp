#pragma once

#include <nanobind/nanobind.h>

#include "mifrost/core/batch_builder.hpp"

namespace nb = nanobind;

namespace mifrost {

void materialize_batch_encoding_lazy_graph_attrs(BatchBuilder::BatchEncoding& encoding);

BatchBuilder::BatchEncoding batch_encoding_from_state_dict(const nb::dict& state);

nb::dict batch_encoding_state_from_instance(nb::handle self, bool include_metadata);

nb::object batch_encoding_object_from_state(const nb::dict& state);

}  // namespace mifrost
