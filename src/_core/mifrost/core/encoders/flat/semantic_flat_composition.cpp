#include "semantic_flat_composition.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace mifrost {

void SemanticFlatEntityComponent::declare_schema(FlatSchemaPlanBuilder& builder) const
{
   (void) builder.declare_node_type(
      std::string(kFlatEntityNodeType), FlatNodeKind::object, 1, export_names_
   );
}

void SemanticFlatEntityComponent::plan_graph(
   const FlatInputView& input,
   FlatNodePlanBuilder& builder
) const
{
   const auto& objects = input.get< SemanticFlatCompositionInput >().composition.objects;
   for(size_t index = 0; index < objects.size(); ++index) {
      (void) builder.add_node_from_source(
         std::string(kFlatEntityNodeType), static_cast< int64_t >(index), objects[index]
      );
   }
}

void SemanticFlatEntityComponent::declare_node_features(FlatNodeFeaturePlanBuilder& builder) const
{
   builder.register_feature(std::string(kFlatEntityNodeType), "x", 1);
}

void SemanticFlatEntityComponent::write_node_features(
   const FlatGraphContext& context,
   FlatNodeFeatureWriter& writer
) const
{
   const auto type = context.nodes.schema().id_for(std::string(kFlatEntityNodeType));
   std::vector< float > values(static_cast< size_t >(context.nodes.count(type)), 0.0F);
   writer.set(std::string(kFlatEntityNodeType), "x", values);
}

SemanticFlatMetadataComponent::SemanticFlatMetadataComponent(
   std::vector< std::string > graph_attrs,
   std::vector< std::string > optional_graph_attrs
)
    : graph_attrs_(std::move(graph_attrs)), optional_graph_attrs_(std::move(optional_graph_attrs))
{
   std::ranges::sort(graph_attrs_);
   graph_attrs_.erase(std::ranges::unique(graph_attrs_).begin(), graph_attrs_.end());
   std::ranges::sort(optional_graph_attrs_);
   optional_graph_attrs_.erase(
      std::ranges::unique(optional_graph_attrs_).begin(), optional_graph_attrs_.end()
   );
}

void SemanticFlatMetadataComponent::declare_metadata(FlatMetadataPlanBuilder& builder) const
{
   builder.claim_object_names();
   for(const auto& key : graph_attrs_) {
      builder.claim_graph_attr(key);
   }
   for(const auto& key : optional_graph_attrs_) {
      builder.claim_optional_graph_attr(key);
   }
}

void SemanticFlatMetadataComponent::write_metadata(
   const FlatGraphContext& context,
   FlatMetadataWriter& writer
) const
{
   const auto& input = context.input.get< SemanticFlatCompositionInput >();
   writer.set_object_names(input.object_names);
   writer.add_lazy_target_names(input.lazy_target_name_strings);
   for(const auto& key : graph_attrs_) {
      const auto it = input.graph_attrs.find(key);
      if(it == input.graph_attrs.end()) {
         throw std::invalid_argument(
            "Semantic flat composition input is missing graph attribute '" + key + "'"
         );
      }
      writer.set_graph_attr(key, it->second);
   }
   for(const auto& key : optional_graph_attrs_) {
      if(const auto it = input.graph_attrs.find(key); it != input.graph_attrs.end()) {
         writer.set_graph_attr(key, it->second);
      }
   }
}

SemanticFlatRelationComponent::SemanticFlatRelationComponent(
   std::string component_name,
   std::vector< FlatCompositionRelationSpec > relations
)
    : component_name_(std::move(component_name)), relations_(std::move(relations))
{
   if(component_name_.empty() or relations_.empty()) {
      throw std::invalid_argument("Semantic flat relation component requires a name and relations");
   }
}

void SemanticFlatRelationComponent::declare_schema(FlatSchemaPlanBuilder& builder) const
{
   for(const auto& relation : relations_) {
      builder.register_relation(relation.key, relation.layout, relation.usage);
   }
}

void SemanticFlatRelationComponent::emit(
   const FlatInputView& input,
   FlatGraphContext& context
) const
{
   const auto& carrier = input.get< SemanticFlatCompositionInput >();
   const auto group = carrier.relation_indices_by_component.find(component_name_);
   if(group == carrier.relation_indices_by_component.end()) {
      throw std::invalid_argument(
         "Semantic flat relation component '" + component_name_ + "' has no carrier index"
      );
   }
   for(const auto relation_index : group->second) {
      const auto& relation = carrier.composition.relations.at(relation_index);
      if(context.should_emit_relation(component_name_, relation.relation_id)) {
         context.emit(relation.relation_id, relation.args);
      }
   }
}

SemanticFlatFieldComponent::SemanticFlatFieldComponent(
   std::string component_name,
   std::vector< FieldDeclaration > fields
)
    : component_name_(std::move(component_name)), fields_(std::move(fields))
{
   if(component_name_.empty() or fields_.empty()) {
      throw std::invalid_argument("Semantic flat field component requires a name and fields");
   }
}

void SemanticFlatFieldComponent::declare_fields(FlatFieldPlanBuilder& builder) const
{
   for(const auto& [key, spec] : fields_) {
      builder.register_field(key, spec);
   }
}

void SemanticFlatFieldComponent::write_fields(
   const FlatGraphContext& context,
   FlatFieldWriter& writer
) const
{
   const auto& carrier = context.input.get< SemanticFlatCompositionInput >();
   for(const auto& [key, spec] : fields_) {
      const auto field_index = carrier.field_index_by_key.find(key);
      if(field_index == carrier.field_index_by_key.end()) {
         throw std::invalid_argument("Semantic flat input is missing field '" + key + "'");
      }
      const auto& field = carrier.composition.fields.at(field_index->second);
      if(spec.dtype == GraphFieldDType::I64) {
         const auto* values = std::get_if< std::vector< int64_t > >(&field.values);
         if(values == nullptr) {
            throw std::invalid_argument("Semantic flat field '" + key + "' has wrong dtype");
         }
         writer.set(key, *values);
      } else {
         const auto* values = std::get_if< std::vector< float > >(&field.values);
         if(values == nullptr) {
            throw std::invalid_argument("Semantic flat field '" + key + "' has wrong dtype");
         }
         writer.set(key, *values);
      }
   }
}

}  // namespace mifrost
