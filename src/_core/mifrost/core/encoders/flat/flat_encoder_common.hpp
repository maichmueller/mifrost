/**
 * @file flat_encoder_common.hpp
 * @brief Shared builder and schema helpers for flat encoders.
 *
 * This file keeps the graph attr names, graph field names, and registration
 * helpers used by both flat relation and flat horizon encoders.
 */
#pragma once

#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "flat_relation_schema.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/encoders/common/target_source.hpp"

namespace mifrost {

inline constexpr std::string_view kFlatEntityNodeType = "entity";
inline constexpr std::string_view kFlatEntityTypeAttr = "entity_node_type";
inline constexpr std::string_view kIncludeLGANEdgesAttr = "include_lgan_edges";
inline constexpr std::string_view kTargetSourcesAttr = "target_sources";
inline constexpr std::string_view kLGANAnchorSourcesAttr = "lgan_anchor_sources";
inline constexpr std::string_view kRelationNamesAttr = "relation_names";
inline constexpr std::string_view kRelationAritiesAttr = "relation_arities";
inline constexpr std::string_view kRelationSourcesAttr = "relation_sources";
inline constexpr std::string_view kNodeSizesField = "node_sizes";
inline constexpr std::string_view kObjectSizesField = "object_sizes";
inline constexpr std::string_view kObjectIndicesField = "object_indices";
inline constexpr std::string_view kHistoryEntitySizesField = "history_entity_sizes";
inline constexpr std::string_view kHistoryEntityIndicesField = "history_entity_indices";
inline constexpr std::string_view kHistoryEntityDtField = "history_entity_dt";
inline constexpr std::string_view kTargetEntitySizesField = "target_entity_sizes";
inline constexpr std::string_view kTargetEntityIndicesField = "target_entity_indices";
inline constexpr std::string_view kTargetEntityGroupIdsField = "target_entity_group_ids";
inline constexpr std::string_view kTargetEntityGroupsAttr = "target_entity_groups";
inline constexpr std::string_view kTargetSizesField = "target_sizes";
inline constexpr std::string_view kRelationInstanceSizesField = "relation_instance_sizes";
inline constexpr std::string_view kRelationCountsField = "relation_counts";
inline constexpr std::string_view kRelationArgsField = "relation_args";
inline constexpr std::string_view kLGANTNSizesField = "lgan_tn_sizes";
inline constexpr std::string_view kLGANTNRelationIndicesField = "lgan_tn_relation_indices";
inline constexpr std::string_view kLGANTNEntityIndicesField = "lgan_tn_entity_indices";
inline constexpr std::string_view kLGANNNSizesField = "lgan_nn_sizes";
inline constexpr std::string_view kLGANNNRelationIndicesField = "lgan_nn_relation_indices";
inline constexpr std::string_view kLGANNNEntityIndicesField = "lgan_nn_entity_indices";
inline constexpr std::string_view kLGANRRSizesField = "lgan_rr_sizes";
inline constexpr std::string_view kLGANRRSrcRelationIndicesField = "lgan_rr_src_relation_indices";
inline constexpr std::string_view kLGANRRDstRelationIndicesField = "lgan_rr_dst_relation_indices";
inline constexpr std::string_view kLGANTNEdgePosAttr = "lgan_tn_edge_pos";
inline constexpr std::string_view kLGANNNEdgePosAttr = "lgan_nn_edge_pos";
inline constexpr std::string_view kLGANRREdgePosAttr = "lgan_rr_edge_pos";

/**
 * @brief Common graph-level settings exported by flat encoders.
 *
 * These values describe the graph format. They are not per-graph state.
 */
struct FlatBuilderGraphConfig {
   bool include_lgan_edges = false;
   bool use_predicate_virtual_nodes = false;
   std::optional< std::vector< std::string > > target_sources;
   std::optional< std::vector< std::string > > lgan_anchor_sources;
   std::optional< std::string > target_symbol_prefix;
   std::vector< std::string > target_entity_group_names;
   std::string lgan_tn_edge_pos;
   std::string lgan_nn_edge_pos;
   std::string lgan_rr_edge_pos;
};

/// Convert configured target sources to stable exported source names.
std::vector< std::string > source_names_for(const std::set< TargetSource >& sources);

/// Write all shared flat graph attrs after schema materialization.
void set_flat_graph_attrs(
   BatchBuilder& builder,
   const FlatRelationSchemaMetadata& metadata,
   const FlatBuilderGraphConfig& config
);

/// Register the flat node-table fields.
void register_flat_entity_fields(BatchBuilder& builder);
/// Register history helper fields used by the flat relation encoder.
void register_flat_history_entity_fields(BatchBuilder& builder);
/// Register target-node fields shared by flat encoders.
void register_flat_target_entity_fields(BatchBuilder& builder);
/// Register relation-instance fields, including arg storage and per-relation counts.
void register_flat_relation_instance_fields(BatchBuilder& builder, int relation_count_dim);
/// Register LGAN adjacency fields for flat graphs.
void register_flat_lgan_fields(BatchBuilder& builder);

}  // namespace mifrost
