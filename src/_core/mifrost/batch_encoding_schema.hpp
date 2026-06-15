#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/schema.hpp"

namespace mifrost {

uint64_t schema_fingerprint(const BatchBuilder::BatchEncoding& encoding);

std::optional< std::string >
find_node_attr_key(const Schema& schema, std::string_view node_type, std::string_view attr);

std::pair< std::optional< std::string >, std::optional< std::string > >
find_edge_index_keys(const Schema& schema, int edge_type_idx);

std::optional< std::string > find_edge_attr_key(const Schema& schema, int edge_type_idx);

}  // namespace mifrost
