#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "batch_builder.hpp"
#include "common_types.hpp"

namespace mifrost {

class TransitionDAG;

inline constexpr std::string_view kTargetPositionsField = "target_positions";
inline constexpr std::string_view kTargetIndicesField = "target_indices";
inline constexpr std::string_view kTargetCandidateIdsField = "target_candidate_ids";
inline constexpr std::string_view kTargetDepthsField = "target_depths";
inline constexpr std::string_view kTargetGroupIdsField = "target_group_ids";
inline constexpr std::string_view kTargetNamesAttr = "target_names";
inline constexpr std::string_view kTargetGroupsAttr = "target_groups";
inline constexpr std::string_view kTargetSymbolPrefixAttr = "target_symbol_prefix";
inline constexpr std::string_view kDefaultTargetSymbolPrefix = "target:";
inline constexpr std::string_view kParentRelationAttr = "parent_relation";

struct TargetRecord {
   int64_t position = 0;
   int64_t index = 0;
   std::optional< int64_t > candidate_id = std::nullopt;
   std::optional< int64_t > depth = std::nullopt;
   std::optional< int64_t > group_id = std::nullopt;
   std::string name;
};

struct TargetCandidateRow {
   int64_t position = 0;
   int64_t index = 0;
   std::optional< int64_t > candidate_id = std::nullopt;
   std::optional< int64_t > depth = std::nullopt;
   std::optional< int64_t > group_id = std::nullopt;
   std::string name;
};

struct TargetColumns {
   std::vector< int64_t > positions;
   std::vector< int64_t > indices;
   std::vector< int64_t > candidate_ids;
   std::vector< int64_t > depths;
   std::vector< int64_t > group_ids;
   std::vector< std::string > names;

   void clear();
   void reserve(size_t count, bool include_depth, bool include_group);
   void append(TargetRecord record, bool include_depth, bool include_group);
   [[nodiscard]] bool empty() const { return positions.empty(); }
   [[nodiscard]] size_t size() const { return positions.size(); }
   void validate(bool include_depth, bool include_group) const;
};

struct TargetCandidateAppendConfig {
   bool include_depth = false;
   bool include_group = false;
   std::string missing_candidate_id_prefix = "missing candidate_id for target index ";
   std::string duplicate_candidate_id_prefix = "duplicate candidate_id ";
};

struct TargetMetadataEmitConfig {
   std::string position_node_type_id;
   std::string symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
   bool include_depth = false;
   bool include_group = false;
   bool include_names = true;
   std::vector< std::string > groups;
   std::optional< std::string > parent_relation = std::nullopt;
};

void append_target_candidate_rows(
   TargetColumns& columns,
   const std::vector< TargetCandidateRow >& rows,
   const TargetCandidateAppendConfig& config
);
void append_target_candidate_row(
   TargetColumns& columns,
   TargetCandidateRow row,
   const TargetCandidateAppendConfig& config
);
std::vector< TargetCandidateRow > collect_transition_dag_target_candidate_rows(
   const TransitionDAG& dag,
   const hash_map< int64_t, int64_t >& positions_by_index,
   bool exclude_root_candidate,
   std::optional< int64_t > group_id,
   bool include_names = true
);
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
