#include "target_metadata.hpp"

#include <stdexcept>

namespace mifrost {

void TargetColumns::clear()
{
   positions.clear();
   indices.clear();
   candidate_ids.clear();
   depths.clear();
   names.clear();
}

void TargetColumns::reserve(size_t count, bool include_depth)
{
   positions.reserve(positions.size() + count);
   indices.reserve(indices.size() + count);
   candidate_ids.reserve(candidate_ids.size() + count);
   names.reserve(names.size() + count);
   if(include_depth) {
      depths.reserve(depths.size() + count);
   }
}

void TargetColumns::append(TargetRecord record, bool include_depth)
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
   names.push_back(std::move(record.name));
}

void TargetColumns::validate(bool include_depth) const
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
}

namespace {

GraphFieldSpec make_target_positions_spec(const std::string& symbol_type_id)
{
   return GraphFieldSpec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::RAGGED_CAT,
      .dim = 1,
      .cat_dim = 0,
      .inc = GraphFieldInc{
         .kind = GraphFieldInc::Kind::NODE_OFFSET,
         .node_type = symbol_type_id,
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

}  // namespace

void register_target_fields(BatchBuilder& builder, const TargetMetadataEmitConfig& config)
{
   builder.register_field(
      std::string(kTargetPositionsField), make_target_positions_spec(config.symbol_type_id)
   );
   builder.register_field(std::string(kTargetIndicesField), make_target_indices_spec());
   builder.register_field(std::string(kTargetCandidateIdsField), make_target_candidate_ids_spec());
   if(config.include_depth) {
      builder.register_field(std::string(kTargetDepthsField), make_target_depths_spec());
   }
}

void set_target_fields(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
)
{
   columns.validate(config.include_depth);
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
}

void set_target_graph_attrs(
   BatchBuilder& builder,
   const TargetColumns& columns,
   const TargetMetadataEmitConfig& config
)
{
   columns.validate(config.include_depth);
   builder.set_graph_attr(std::string(kTargetNamesAttr), columns.names);
   builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config.symbol_prefix);
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
