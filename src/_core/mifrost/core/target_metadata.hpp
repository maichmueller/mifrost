#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "batch_builder.hpp"

namespace mifrost {

inline constexpr std::string_view kTargetPositionsField = "target_positions";
inline constexpr std::string_view kTargetIndicesField = "target_indices";
inline constexpr std::string_view kTargetCandidateIdsField = "target_candidate_ids";
inline constexpr std::string_view kTargetDepthsField = "target_depths";
inline constexpr std::string_view kTargetNamesAttr = "target_names";
inline constexpr std::string_view kTargetSymbolPrefixAttr = "target_symbol_prefix";
inline constexpr std::string_view kDefaultTargetSymbolPrefix = "target:";
inline constexpr std::string_view kParentRelationAttr = "parent_relation";

struct TargetRecord {
   int64_t position = 0;
   int64_t index = 0;
   std::optional< int64_t > candidate_id = std::nullopt;
   std::optional< int64_t > depth = std::nullopt;
   std::string name;
};

struct TargetColumns {
   std::vector< int64_t > positions;
   std::vector< int64_t > indices;
   std::vector< int64_t > candidate_ids;
   std::vector< int64_t > depths;
   std::vector< std::string > names;

   void clear();
   void reserve(size_t count, bool include_depth);
   void append(TargetRecord record, bool include_depth);
   [[nodiscard]] bool empty() const { return positions.empty(); }
   [[nodiscard]] size_t size() const { return positions.size(); }
   void validate(bool include_depth) const;
};

struct TargetMetadataEmitConfig {
   std::string symbol_type_id;
   std::string symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
   bool include_depth = false;
   std::optional< std::string > parent_relation = std::nullopt;
};

void register_target_fields(BatchBuilder& builder, const TargetMetadataEmitConfig& config);
void set_target_fields(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
);
void set_target_graph_attrs(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
);
void emit_target_metadata(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
);

}  // namespace mifrost
