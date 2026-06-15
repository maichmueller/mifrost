#pragma once

#include <nanobind/nanobind.h>

#include <string>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/graph_fields.hpp"

namespace nb = nanobind;

namespace mifrost {

GraphFieldSpec graph_field_spec_from_dict(const nb::dict& spec_dict);

nb::dict graph_field_spec_to_dict(const GraphFieldSpec& spec);

nb::dict graph_field_map_to_dict(const hash_map< std::string, GraphField >& fields);

hash_map< std::string, GraphField > graph_field_map_from_dict(const nb::dict& payload);

}  // namespace mifrost
