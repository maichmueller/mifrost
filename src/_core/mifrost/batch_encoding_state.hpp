#pragma once

#include <nanobind/nanobind.h>

#include <string>

#include "mifrost/core/batch_builder.hpp"

namespace nb = nanobind;

namespace mifrost {

void materialize_batch_encoding_lazy_graph_attrs(BatchBuilder::BatchEncoding& encoding);

BatchBuilder::BatchEncoding batch_encoding_from_state_dict(const nb::dict& state);

nb::dict batch_encoding_state_from_instance(nb::handle self, bool include_metadata);

nb::object batch_encoding_object_from_state(const nb::dict& state);

void batch_encoding_save(nb::handle self, const std::string& path, bool include_metadata);

nb::object batch_encoding_load(const std::string& path);

nb::bytes batch_encoding_dumps(nb::handle self, bool include_metadata);

nb::object batch_encoding_loads(nb::bytes payload);

nb::dict batch_encoding_getstate(nb::handle self);

nb::tuple batch_encoding_reduce(nb::handle self);

void batch_encoding_setstate(nb::handle self, const nb::dict& state);

}  // namespace mifrost
