#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/schema.hpp"

namespace nb = nanobind;

namespace mifrost {

int64_t batch_encoding_num_nodes(const BatchBuilder::BatchEncoding& encoding);

int64_t batch_encoding_num_edges(const BatchBuilder::BatchEncoding& encoding);

nb::list batch_encoding_edge_types(const BatchBuilder::BatchEncoding& encoding);

std::optional< nb::dict > owner_tensor_cache_if_present(nb::handle owner);

void set_owner_target_device(nb::handle owner, nb::handle device);

void materialize_owner_tensor_cache(nb::handle owner, BatchBuilder::BatchEncoding& encoding);

nb::object zeros_f32_on_owner_device(nb::handle owner, int64_t rows, int64_t cols);

std::optional< std::string >
find_node_attr_key(const Schema& schema, std::string_view node_type, std::string_view attr);

std::pair< std::optional< std::string >, std::optional< std::string > >
find_edge_index_keys(const Schema& schema, int edge_type_idx);

std::optional< std::string > find_edge_attr_key(const Schema& schema, int edge_type_idx);

}  // namespace mifrost
