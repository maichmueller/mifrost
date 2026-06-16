#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mifrost/core/batch_builder.hpp"

namespace nb = nanobind;

namespace mifrost {

int64_t batch_encoding_num_nodes(const BatchBuilder::BatchEncoding& encoding);

int64_t batch_encoding_num_edges(const BatchBuilder::BatchEncoding& encoding);

std::vector< std::string > batch_encoding_node_types(
   const BatchBuilder::BatchEncoding& encoding
);

nb::list batch_encoding_edge_types(const BatchBuilder::BatchEncoding& encoding);

nb::dict batch_encoding_as_dict(BatchBuilder::BatchEncoding& encoding, nb::handle owner);

std::optional< nb::object >
batch_encoding_graph_attr_if_present(BatchBuilder::BatchEncoding& encoding, std::string_view key);

nb::object
batch_encoding_as_pyg(BatchBuilder::BatchEncoding& encoding, std::optional< bool > as_batch);

}  // namespace mifrost
