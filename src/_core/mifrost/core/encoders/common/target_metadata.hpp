/**
 * @file target_metadata.hpp
 * @brief Shared helpers for exported target metadata.
 *
 * This file defines the exported target columns: positions, indices, candidate
 * ids, depths, groups, and optional names. Flat and hetero encoders both use
 * the same layout.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/root_policy.hpp"

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

/**
 * @brief One target row before it is split into columns.
 *
 * This is the normalized row representation used while collecting target
 * metadata from actions, goals, history entries, or horizon DAG nodes.
 */
struct TargetCandidateRow {
   int64_t position = 0;
   int64_t index = 0;
   std::optional< int64_t > candidate_id = std::nullopt;
   std::optional< int64_t > depth = std::nullopt;
   std::optional< int64_t > group_id = std::nullopt;
   std::string name;
};

/**
 * @brief Target metadata stored as columns during encoding.
 *
 * The vectors are kept in lockstep and later written to `BatchBuilder` fields.
 * Optional columns are only populated when the corresponding emit config asks
 * for them.
 */
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

/// Append pre-normalized candidate rows while enforcing metadata invariants.
void append_target_candidate_rows(
   TargetColumns& columns,
   const std::vector< TargetCandidateRow >& rows,
   const TargetCandidateAppendConfig& config
);
/// Append one target row while enforcing candidate-id uniqueness rules.
void append_target_candidate_row(
   TargetColumns& columns,
   TargetCandidateRow row,
   const TargetCandidateAppendConfig& config
);
/// Derive target rows for transition-DAG nodes under a specific root policy.
std::vector< TargetCandidateRow > collect_transition_dag_target_candidate_rows(
   const TransitionDAG& dag,
   const hash_map< int64_t, int64_t >& positions_by_index,
   RootPolicy root_policy,
   std::optional< int64_t > group_id,
   bool include_names = true
);
/// Register the graph fields and attrs required by one target metadata schema.
void register_target_fields(BatchBuilder& builder, const TargetMetadataEmitConfig& config);
/// Write target columns into an already-registered builder schema.
void set_target_fields(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
);
/// Write target names/group labels as graph attrs when enabled.
void set_target_graph_attrs(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
);
/// Convenience wrapper that registers nothing and only emits populated metadata.
void emit_target_metadata(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
);

}  // namespace mifrost
