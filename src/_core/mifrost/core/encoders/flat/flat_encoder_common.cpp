#include "flat_encoder_common.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mifrost {

namespace {

GraphFieldInc entity_node_offset_inc()
{
   return GraphFieldInc{
      .kind = GraphFieldInc::Kind::NODE_OFFSET,
      .node_type = std::string(kFlatEntityNodeType),
   };
}

std::vector< int64_t >&
expect_i64_graph_field(BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   const std::string key_str(key);
   const auto it = encoding.graph_fields.find(key_str);
   if(it == encoding.graph_fields.end()) {
      throw std::invalid_argument("Flat batch is missing graph field '" + key_str + "'");
   }
   if(it->second.spec.dtype != GraphFieldDType::I64) {
      throw std::invalid_argument("Flat graph field '" + key_str + "' must have dtype=i64");
   }
   auto* values = std::get_if< std::vector< int64_t > >(&it->second.values);
   if(values == nullptr) {
      throw std::invalid_argument("Flat graph field '" + key_str + "' storage dtype mismatch");
   }
   return *values;
}

size_t checked_relation_width(int64_t count, int64_t arity)
{
   if(count < 0) {
      throw std::invalid_argument("relation_counts must be non-negative");
   }
   if(arity < 0) {
      throw std::invalid_argument("relation_arities must be non-negative");
   }
   const auto count_size = static_cast< size_t >(count);
   const auto arity_size = static_cast< size_t >(arity);
   if(arity_size != 0 and count_size > std::numeric_limits< size_t >::max() / arity_size) {
      throw std::invalid_argument("relation slot width exceeds size_t range");
   }
   return count_size * arity_size;
}

}  // namespace

std::vector< std::string > source_names_for(const std::set< TargetSource >& sources)
{
   std::vector< std::string > out;
   out.reserve(sources.size());
   for(const auto source : kCanonicalTargetSourceOrder) {
      if(sources.contains(source)) {
         out.emplace_back(target_source_group_name(source));
      }
   }
   return out;
}

void set_flat_graph_attrs(
   BatchBuilder& builder,
   const FlatRelationSchemaMetadata& metadata,
   const FlatBuilderGraphConfig& config
)
{
   builder.set_graph_kind("flat");
   builder.set_schema_flag("flat_relations", true);
   builder.set_graph_attr(std::string(kFlatEntityTypeAttr), std::string(kFlatEntityNodeType));
   builder.set_graph_attr(
      std::string(kIncludeLGANEdgesAttr), static_cast< int64_t >(config.include_lgan_edges)
   );
   if(config.target_sources.has_value()) {
      builder.set_graph_attr(std::string(kTargetSourcesAttr), *config.target_sources);
   }
   if(config.lgan_anchor_sources.has_value()) {
      builder.set_graph_attr(std::string(kLGANAnchorSourcesAttr), *config.lgan_anchor_sources);
   }
   builder.set_graph_attr(std::string(kEntityRoleNamesAttr), flat_entity_role_names());
   builder.set_graph_attr(std::string(kRelationNamesAttr), metadata.relation_names);
   builder.set_graph_attr(std::string(kRelationAritiesAttr), metadata.relation_arities);
   builder.set_graph_attr(std::string(kRelationSourcesAttr), metadata.relation_sources);
   builder.set_graph_attr(
      std::string(kRelationLogicalAritiesAttr), metadata.relation_logical_arities
   );
   builder.set_graph_attr(
      std::string(kRelationEncodedAritiesAttr), metadata.relation_encoded_arities
   );
   builder.set_graph_attr(std::string(kRelationSlotRolesAttr), metadata.relation_slot_roles);
   builder.set_graph_attr(
      std::string(kRelationSlotRoleOffsetsAttr), metadata.relation_slot_role_offsets
   );
   builder.set_graph_attr(std::string(kSlotRoleNamesAttr), metadata.slot_role_names);
   builder.set_graph_attr(std::string(kTargetEntityGroupsAttr), config.target_entity_group_names);
   if(config.target_symbol_prefix.has_value()) {
      builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), *config.target_symbol_prefix);
   }
   builder.set_graph_attr(std::string(kLGANTNEdgePosAttr), config.lgan_tn_edge_pos);
   builder.set_graph_attr(std::string(kLGANNNEdgePosAttr), config.lgan_nn_edge_pos);
   builder.set_graph_attr(std::string(kLGANRREdgePosAttr), config.lgan_rr_edge_pos);
   builder.set_graph_attr(
      std::string(kRelationArgsLayoutAttr),
      std::string(
         config.pack_relation_args_relation_major ? kRelationArgsRelationMajorLayout
                                                  : kRelationArgsGraphMajorLayout
      )
   );
   builder.set_graph_attr(
      std::string(kUsePredicateVirtualNodesAttr),
      static_cast< int64_t >(config.use_predicate_virtual_nodes)
   );
}

void register_flat_entity_fields(BatchBuilder& builder)
{
   builder.register_field(
      std::string(kNodeSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kObjectSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kObjectIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = entity_node_offset_inc(),
      }
   );
   builder.register_field(
      std::string(kEntityRoleIdsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
      }
   );
}

void register_flat_history_entity_fields(BatchBuilder& builder)
{
   builder.register_field(
      std::string(kHistoryEntitySizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kHistoryEntityIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = entity_node_offset_inc(),
      }
   );
   builder.register_field(
      std::string(kHistoryEntityDtField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
      }
   );
}

void register_flat_target_entity_fields(BatchBuilder& builder)
{
   builder.register_field(
      std::string(kTargetEntitySizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kTargetEntityIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = entity_node_offset_inc(),
      }
   );
   builder.register_field(
      std::string(kTargetEntityGroupIdsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
      }
   );
}

void register_flat_relation_instance_fields(BatchBuilder& builder, int relation_count_dim)
{
   builder.register_field(
      std::string(kRelationInstanceSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kRelationCountsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = relation_count_dim,
      }
   );
   builder.register_field(
      std::string(kRelationArgsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = entity_node_offset_inc(),
      }
   );
}

void register_flat_lgan_fields(BatchBuilder& builder)
{
   builder.register_field(
      std::string(kLGANTNSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kLGANTNRelationIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::FIELD_OFFSET,
            .field_key = std::string(kRelationInstanceSizesField),
         },
      }
   );
   builder.register_field(
      std::string(kLGANTNEntityIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = entity_node_offset_inc(),
      }
   );
   builder.register_field(
      std::string(kLGANNNSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kLGANNNRelationIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::FIELD_OFFSET,
            .field_key = std::string(kRelationInstanceSizesField),
         },
      }
   );
   builder.register_field(
      std::string(kLGANNNEntityIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = entity_node_offset_inc(),
      }
   );
   builder.register_field(
      std::string(kLGANRRSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kLGANRRSrcRelationIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::FIELD_OFFSET,
            .field_key = std::string(kRelationInstanceSizesField),
         },
      }
   );
   builder.register_field(
      std::string(kLGANRRDstRelationIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::FIELD_OFFSET,
            .field_key = std::string(kRelationInstanceSizesField),
         },
      }
   );
}

void pack_flat_relation_args_relation_major(
   BatchBuilder::BatchEncoding& encoding,
   std::span< const int64_t > relation_arities
)
{
   if(encoding.num_graphs <= 1) {
      return;
   }

   auto& relation_counts = expect_i64_graph_field(encoding, std::string_view(kRelationCountsField));
   auto& relation_args = expect_i64_graph_field(encoding, std::string_view(kRelationArgsField));
   const size_t graph_count = static_cast< size_t >(encoding.num_graphs);
   const size_t relation_count = relation_arities.size();
   if(relation_count == 0) {
      if(not relation_counts.empty() or not relation_args.empty()) {
         throw std::invalid_argument(
            "relation_counts/relation_args must be empty when relation_arities is empty"
         );
      }
      return;
   }
   if(relation_counts.size() != graph_count * relation_count) {
      throw std::invalid_argument(
         "relation_counts shape does not match num_graphs * relation_arities"
      );
   }
   if(relation_args.empty()) {
      const bool has_slots = std::ranges::any_of(relation_counts, [](int64_t count) {
         return count != 0;
      });
      if(has_slots) {
         throw std::invalid_argument(
            "relation_args is empty but relation_counts implies relation slots"
         );
      }
      return;
   }

   std::vector< size_t > segment_starts(graph_count * relation_count, 0);
   std::vector< size_t > segment_widths(graph_count * relation_count, 0);
   size_t cursor = 0;
   for(size_t graph_idx = 0; graph_idx < graph_count; ++graph_idx) {
      for(size_t relation_idx = 0; relation_idx < relation_count; ++relation_idx) {
         const size_t index = graph_idx * relation_count + relation_idx;
         const size_t width = checked_relation_width(
            relation_counts[index], relation_arities[relation_idx]
         );
         if(width > relation_args.size() - std::min(cursor, relation_args.size())) {
            throw std::invalid_argument(
               "relation_counts/relation_arities imply more relation_args slots than stored"
            );
         }
         segment_starts[index] = cursor;
         segment_widths[index] = width;
         cursor += width;
      }
   }
   if(cursor != relation_args.size()) {
      throw std::invalid_argument(
         "relation_args length does not match packed slot width implied by relation_counts"
      );
   }

   std::vector< int64_t > relation_major;
   relation_major.reserve(relation_args.size());
   for(size_t relation_idx = 0; relation_idx < relation_count; ++relation_idx) {
      for(size_t graph_idx = 0; graph_idx < graph_count; ++graph_idx) {
         const size_t index = graph_idx * relation_count + relation_idx;
         const size_t start = segment_starts[index];
         const size_t width = segment_widths[index];
         relation_major.insert(
            relation_major.end(),
            relation_args.begin() + static_cast< std::vector< int64_t >::difference_type >(start),
            relation_args.begin()
               + static_cast< std::vector< int64_t >::difference_type >(start + width)
         );
      }
   }
   relation_args.swap(relation_major);
}

}  // namespace mifrost
