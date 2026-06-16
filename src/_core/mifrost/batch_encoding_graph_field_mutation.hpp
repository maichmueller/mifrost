#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <string_view>

#include "mifrost/core/batch_builder.hpp"

namespace nb = nanobind;

namespace mifrost {

bool is_native_graph_field_ptr_key(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view key
);

void set_batch_builder_graph_field(BatchBuilder& builder, const std::string& key, nb::handle value);

void set_batch_builder_graph_fields(BatchBuilder& builder, const nb::dict& values);

void set_batch_encoding_graph_field(
   BatchBuilder::BatchEncoding& encoding,
   const std::string& key,
   nb::handle value
);

void set_batch_encoding_graph_fields(BatchBuilder::BatchEncoding& encoding, const nb::dict& values);

}  // namespace mifrost
