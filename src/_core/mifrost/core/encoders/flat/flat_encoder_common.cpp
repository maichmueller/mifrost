#include "flat_encoder_common.hpp"

namespace mifrost {

namespace {

GraphFieldInc entity_node_offset_inc()
{
   return GraphFieldInc{
      .kind = GraphFieldInc::Kind::NODE_OFFSET,
      .node_type = std::string(kFlatEntityNodeType),
   };
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

}  // namespace mifrost
