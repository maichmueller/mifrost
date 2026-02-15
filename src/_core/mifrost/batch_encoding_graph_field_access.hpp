#pragma once

#include <nanobind/nanobind.h>

#include <set>
#include <string>
#include <string_view>

#include "mifrost/core/batch_builder.hpp"

namespace mifrost {

std::set< std::string > batch_encoding_native_graph_field_keys(
   const BatchBuilder::BatchEncoding& encoding
);

bool batch_encoding_has_graph_field(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view key
);

nanobind::object batch_encoding_get_graph_field(
   BatchBuilder::BatchEncoding& encoding,
   std::string_view key,
   nanobind::handle owner
);

void validate_batch_encoding_graph_fields(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view context
);

}  // namespace mifrost
