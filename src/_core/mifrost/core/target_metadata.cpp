#include "target_metadata.hpp"

#include <mimir/search/formatter.hpp>
#include <sstream>
#include <stdexcept>

#include "common_types.hpp"
#include "transition_dag.hpp"

namespace mifrost {

void TargetColumns::clear()
{
   positions.clear();
   indices.clear();
   candidate_ids.clear();
   depths.clear();
   group_ids.clear();
   names.clear();
}

void TargetColumns::reserve(size_t count, bool include_depth, bool include_group)
{
   positions.reserve(positions.size() + count);
   indices.reserve(indices.size() + count);
   candidate_ids.reserve(candidate_ids.size() + count);
   names.reserve(names.size() + count);
   if(include_depth) {
      depths.reserve(depths.size() + count);
   }
   if(include_group) {
      group_ids.reserve(group_ids.size() + count);
   }
}

void TargetColumns::append(TargetRecord record, bool include_depth, bool include_group)
{
   positions.push_back(record.position);
   indices.push_back(record.index);
   if(not record.candidate_id.has_value()) {
      throw std::invalid_argument("target record candidate_id is required for target metadata");
   }
   candidate_ids.push_back(*record.candidate_id);
   if(include_depth) {
      if(not record.depth.has_value()) {
         throw std::invalid_argument("target record depth is required for this target metadata");
      }
      depths.push_back(*record.depth);
   }
   if(include_group) {
      if(not record.group_id.has_value()) {
         throw std::invalid_argument("target record group_id is required for this target metadata");
      }
      group_ids.push_back(*record.group_id);
   }
   names.push_back(std::move(record.name));
}

void TargetColumns::validate(bool include_depth, bool include_group) const
{
   const size_t rows = positions.size();
   if(indices.size() != rows) {
      throw std::invalid_argument("target metadata has mismatched positions/indices lengths");
   }
   if(candidate_ids.size() != rows) {
      throw std::invalid_argument("target metadata has mismatched positions/candidate_ids lengths");
   }
   if(names.size() != rows) {
      throw std::invalid_argument("target metadata has mismatched positions/names lengths");
   }
   if(include_depth) {
      if(depths.size() != rows) {
         throw std::invalid_argument("target metadata has mismatched positions/depths lengths");
      }
   } else if(not depths.empty()) {
      throw std::invalid_argument("target metadata depths must be empty when depth is disabled");
   }
   if(include_group) {
      if(group_ids.size() != rows) {
         throw std::invalid_argument("target metadata has mismatched positions/group_ids lengths");
      }
   } else if(not group_ids.empty()) {
      throw std::invalid_argument("target metadata group_ids must be empty when group is disabled");
   }
}

void append_target_candidate_rows(
   TargetColumns& columns,
   const std::vector< TargetCandidateRow >& rows,
   const TargetCandidateAppendConfig& config
)
{
   bool has_explicit_candidate_ids = false;
   std::optional< int64_t > first_missing_candidate_id_index = std::nullopt;
   for(const auto& row : rows) {
      if(row.candidate_id.has_value()) {
         has_explicit_candidate_ids = true;
      } else if(not first_missing_candidate_id_index.has_value()) {
         first_missing_candidate_id_index = row.index;
      }
   }
   if(has_explicit_candidate_ids and first_missing_candidate_id_index.has_value()) {
      throw std::invalid_argument(
         config.missing_candidate_id_prefix + std::to_string(*first_missing_candidate_id_index)
      );
   }

   columns.reserve(rows.size(), config.include_depth, config.include_group);
   hash_set< int64_t > seen_candidate_ids;
   seen_candidate_ids.reserve(rows.size());
   for(const auto& row : rows) {
      const int64_t candidate_id = has_explicit_candidate_ids ? *row.candidate_id : row.index;
      if(not seen_candidate_ids.emplace(candidate_id).second) {
         throw std::invalid_argument(
            config.duplicate_candidate_id_prefix + std::to_string(candidate_id)
         );
      }
      columns.append(
         TargetRecord{
            .position = row.position,
            .index = row.index,
            .candidate_id = candidate_id,
            .depth = row.depth,
            .group_id = row.group_id,
            .name = row.name,
         },
         config.include_depth,
         config.include_group
      );
   }
}

void append_target_candidate_row(
   TargetColumns& columns,
   TargetCandidateRow row,
   const TargetCandidateAppendConfig& config
)
{
   const int64_t candidate_id = row.candidate_id.value_or(row.index);
   columns.reserve(1, config.include_depth, config.include_group);
   columns.append(
      TargetRecord{
         .position = row.position,
         .index = row.index,
         .candidate_id = candidate_id,
         .depth = row.depth,
         .group_id = row.group_id,
         .name = std::move(row.name),
      },
      config.include_depth,
      config.include_group
   );
}

std::vector< TargetCandidateRow > collect_transition_dag_target_candidate_rows(
   const TransitionDAG& dag,
   const hash_map< int64_t, int64_t >& positions_by_index,
   bool exclude_root_candidate,
   std::optional< int64_t > group_id,
   bool include_names
)
{
   const auto& nodes = dag.nodes();
   const size_t reserved = (exclude_root_candidate and not nodes.empty()) ? (nodes.size() - 1)
                                                                          : nodes.size();
   std::vector< TargetCandidateRow > rows;
   rows.reserve(reserved);

   for(const auto& node : nodes) {
      if(exclude_root_candidate and node.index == dag.root_index()) {
         continue;
      }
      const auto position_it = positions_by_index.find(node.index);
      if(position_it == positions_by_index.end()) {
         continue;
      }
      rows.push_back(
         TargetCandidateRow{
            .position = position_it->second,
            .index = node.index,
            .candidate_id = node.candidate_id,
            .depth = node.depth,
            .group_id = group_id,
            .name = [&]() {
               if(not include_names) {
                  return std::string{};
               }
               std::ostringstream stream;
               stream << node.state;
               return stream.str();
            }(),
         }
      );
   }

   return rows;
}

namespace {

GraphFieldSpec make_target_positions_spec(const std::string& position_node_type_id)
{
   return GraphFieldSpec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::RAGGED_CAT,
      .dim = 1,
      .cat_dim = 0,
      .inc = GraphFieldInc{
         .kind = GraphFieldInc::Kind::NODE_OFFSET,
         .node_type = position_node_type_id,
      },
   };
}

GraphFieldSpec make_target_indices_spec()
{
   return GraphFieldSpec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::RAGGED_CAT,
      .dim = 1,
      .cat_dim = 0,
      .inc = GraphFieldInc{},
   };
}

GraphFieldSpec make_target_depths_spec()
{
   return make_target_indices_spec();
}

GraphFieldSpec make_target_candidate_ids_spec()
{
   return make_target_indices_spec();
}

GraphFieldSpec make_target_group_ids_spec()
{
   return make_target_indices_spec();
}

}  // namespace

void register_target_fields(BatchBuilder& builder, const TargetMetadataEmitConfig& config)
{
   builder.register_field(
      std::string(kTargetPositionsField), make_target_positions_spec(config.position_node_type_id)
   );
   builder.register_field(std::string(kTargetIndicesField), make_target_indices_spec());
   builder.register_field(std::string(kTargetCandidateIdsField), make_target_candidate_ids_spec());
   if(config.include_depth) {
      builder.register_field(std::string(kTargetDepthsField), make_target_depths_spec());
   }
   if(config.include_group) {
      builder.register_field(std::string(kTargetGroupIdsField), make_target_group_ids_spec());
   }
}

void set_target_fields(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
)
{
   columns.validate(config.include_depth, config.include_group);
   builder.set_field(
      std::string(kTargetPositionsField), std::span< const int64_t >(columns.positions)
   );
   builder.set_field(std::string(kTargetIndicesField), std::span< const int64_t >(columns.indices));
   builder.set_field(
      std::string(kTargetCandidateIdsField), std::span< const int64_t >(columns.candidate_ids)
   );
   if(config.include_depth) {
      builder.set_field(
         std::string(kTargetDepthsField), std::span< const int64_t >(columns.depths)
      );
   }
   if(config.include_group) {
      builder.set_field(
         std::string(kTargetGroupIdsField), std::span< const int64_t >(columns.group_ids)
      );
   }
}

void set_target_graph_attrs(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
)
{
   columns.validate(config.include_depth, config.include_group);
   if(config.include_names) {
      builder.set_graph_attr(std::string(kTargetNamesAttr), columns.names);
   }
   builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config.symbol_prefix);
   if(config.include_group) {
      builder.set_graph_attr(std::string(kTargetGroupsAttr), config.groups);
   }
   if(config.parent_relation.has_value()) {
      builder.set_graph_attr(std::string(kParentRelationAttr), *config.parent_relation);
   }
}

void emit_target_metadata(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
)
{
   register_target_fields(builder, config);
   set_target_fields(builder, columns, config);
   set_target_graph_attrs(builder, columns, config);
}

}  // namespace mifrost
