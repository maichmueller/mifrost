#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_schema.hpp"
#include "mifrost/batch_encoding_tensor_cache.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/schema.hpp"

namespace nb = nanobind;

namespace mifrost {

int64_t batch_encoding_num_nodes(const BatchBuilder::BatchEncoding& encoding);

int64_t batch_encoding_num_edges(const BatchBuilder::BatchEncoding& encoding);

nb::list batch_encoding_edge_types(const BatchBuilder::BatchEncoding& encoding);

nb::object
batch_encoding_as_pyg(BatchBuilder::BatchEncoding& encoding, std::optional< bool > as_batch);

std::string batch_encoding_repr(nb::handle self, const BatchBuilder::BatchEncoding& encoding);

std::string batch_encoding_str(nb::handle self, const BatchBuilder::BatchEncoding& encoding);

}  // namespace mifrost
