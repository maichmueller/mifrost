/**
 * @file semantic_flat_composition.hpp
 * @brief Backend-neutral carrier and adapter seam for semantic flat encoders.
 */
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "flat_composition.hpp"

namespace mifrost {

/**
 * A graph-local carrier for semantic flat data that has already been resolved
 * to the compiled flat schema.  This is the stable adapter seam between
 * semantic extraction and the compiled composition runtime: an adapter may
 * populate it directly from a semantic input, without exposing planner types
 * to the composition runtime.
 *
 * Semantic engines populate this carrier directly from their owned semantic
 * inputs. Legacy encoders may be invoked independently by parity tests, but
 * production composition never constructs or compares a second encoding.
 */
struct SemanticFlatCompositionInput {
   FlatCompositionInput composition;
   std::vector< std::string > object_names;
   std::vector< std::string > lazy_target_name_strings;
   std::unordered_map< std::string, BatchBuilder::GraphAttrValue > graph_attrs;
   std::unordered_map< std::string, std::vector< size_t > > relation_indices_by_component;
   std::unordered_map< std::string, size_t > field_index_by_key;

   void rebuild_indexes()
   {
      relation_indices_by_component.clear();
      for(size_t index = 0; index < composition.relations.size(); ++index) {
         relation_indices_by_component[composition.relations[index].component].push_back(index);
      }
      field_index_by_key.clear();
      for(size_t index = 0; index < composition.fields.size(); ++index) {
         if(not field_index_by_key.emplace(composition.fields[index].key, index).second) {
            throw std::invalid_argument(
               "Semantic flat composition input declares field '" + composition.fields[index].key
               + "' more than once"
            );
         }
      }
   }
};

class MIFROST_API SemanticFlatEntityComponent final: public FlatEmitterComponent {
  public:
   explicit SemanticFlatEntityComponent(bool export_names = false) : export_names_(export_names) {}

   [[nodiscard]] std::string_view name() const noexcept override { return "semantic_entities"; }
   void declare_schema(FlatSchemaPlanBuilder&) const override;
   void plan_graph(const FlatInputView&, FlatNodePlanBuilder&) const override;
   void declare_node_features(FlatNodeFeaturePlanBuilder&) const override;
   void write_node_features(const FlatGraphContext&, FlatNodeFeatureWriter&) const override;

  private:
   bool export_names_ = false;
};

/**
 * Writes the metadata portion of a semantic carrier.  Relations, fields, and
 * entity rows remain ordinary backend-neutral composition components.
 */
class MIFROST_API SemanticFlatMetadataComponent final: public FlatEmitterComponent {
  public:
   explicit SemanticFlatMetadataComponent(
      std::vector< std::string > graph_attrs,
      std::vector< std::string > optional_graph_attrs = {}
   );

   [[nodiscard]] std::string_view name() const noexcept override { return "semantic_metadata"; }
   void declare_metadata(FlatMetadataPlanBuilder&) const override;
   void write_metadata(const FlatGraphContext&, FlatMetadataWriter&) const override;

  private:
   std::vector< std::string > graph_attrs_;
   std::vector< std::string > optional_graph_attrs_;
};

class MIFROST_API SemanticFlatRelationComponent final: public FlatEmitterComponent {
  public:
   SemanticFlatRelationComponent(
      std::string component_name,
      std::vector< FlatCompositionRelationSpec > relations
   );

   [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
   void declare_schema(FlatSchemaPlanBuilder&) const override;
   void emit(const FlatInputView&, FlatGraphContext&) const override;

  private:
   std::string component_name_;
   std::vector< FlatCompositionRelationSpec > relations_;
};

class MIFROST_API SemanticFlatFieldComponent final: public FlatEmitterComponent {
  public:
   using FieldDeclaration = std::pair< std::string, GraphFieldSpec >;

   SemanticFlatFieldComponent(std::string component_name, std::vector< FieldDeclaration > fields);

   [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
   void declare_fields(FlatFieldPlanBuilder&) const override;
   void write_fields(const FlatGraphContext&, FlatFieldWriter&) const override;

  private:
   std::string component_name_;
   std::vector< FieldDeclaration > fields_;
};

}  // namespace mifrost
