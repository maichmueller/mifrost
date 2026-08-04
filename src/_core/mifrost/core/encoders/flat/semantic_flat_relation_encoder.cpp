/**
 * @file semantic_flat_relation_encoder.cpp
 * @brief Native encoding of owned, planning-backend-neutral flat graph inputs.
 */
#include "semantic_flat_relation_encoder.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

#include "flat_encoder_common.hpp"
#include "flat_lgan.hpp"
#include "flat_relation_schema.hpp"
#include "flat_tuple_layout.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/encoders/flat/flat_composition.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_composition.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"
#include "mifrost/core/semantic/semantic_transition_dag.hpp"

namespace mifrost {

namespace {

const std::shared_ptr< const SemanticTaskContext >& require_task_context(
   const std::shared_ptr< const SemanticTaskContext >& task_context,
   std::string_view encoder_name
)
{
   if(not task_context) {
      throw std::invalid_argument(std::string(encoder_name) + " task context must not be null");
   }
   return task_context;
}

constexpr std::array< SemanticPredicateCategory, 3 > kCategoryOrder = {
   SemanticPredicateCategory::static_predicate,
   SemanticPredicateCategory::fluent,
   SemanticPredicateCategory::derived,
};

constexpr std::array< std::string_view, 4 > kGoalLevelSuffixes = {
   "[g]",
   "[sg]",
   "[ssg]",
   "[sssg]",
};

const std::set< std::string, std::less<> > kTopTypePredicates = {
   "object",
   "number",
   "symbol",
   "_action_",
};

bool supports_semantic_goal_derivation(GoalDerivation derivation)
{
   return derivation == GoalDerivation::plain or derivation == GoalDerivation::satisfied
          or derivation == GoalDerivation::unsatisfied;
}

bool has_target_source(const FlatRelationEncoderConfig& config, TargetSource source)
{
   return config.target_sources.contains(source);
}

bool has_lgan_anchor_source(const FlatRelationEncoderConfig& config, TargetSource source)
{
   return config.include_lgan_edges and config.lgan_anchor_sources.contains(source);
}

bool has_anchor_entity_source(const FlatRelationEncoderConfig& config, TargetSource source)
{
   return has_target_source(config, source) or has_lgan_anchor_source(config, source);
}

std::optional< TargetSource >
source_for_goal_level(const FlatRelationEncoderConfig& config, size_t level)
{
   const auto source = level > 0 ? TargetSource::subgoals : TargetSource::goals;
   return has_anchor_entity_source(config, source) ? std::optional(source) : std::nullopt;
}

FlatTupleLayout
semantic_goal_layout(const FlatRelationEncoderConfig& config, int logical_arity, size_t level)
{
   std::vector< FlatSlotRole > roles;
   if(const auto source = source_for_goal_level(config, level); source.has_value()) {
      roles.push_back(slot_role_for_target_source(*source));
   }
   return make_predicate_tuple_layout(
      logical_arity, std::span{roles}, config.use_predicate_virtual_nodes
   );
}

FlatTupleLayout semantic_history_layout(const FlatRelationEncoderConfig& config, int logical_arity)
{
   std::vector< FlatSlotRole > roles;
   if(has_anchor_entity_source(config, TargetSource::history)) {
      roles.push_back(FlatSlotRole::history_target_slot);
   }
   roles.push_back(FlatSlotRole::history_slot);
   return make_predicate_tuple_layout(
      logical_arity, std::span{roles}, config.use_predicate_virtual_nodes
   );
}

std::string atom_display_name(
   const SemanticAtom& atom,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< std::string >& objects
)
{
   std::string result = "(" + predicates.at(static_cast< size_t >(atom.predicate)).name;
   for(const int64_t object : atom.arguments) {
      result += " ";
      result += objects.at(static_cast< size_t >(object));
   }
   result += ")";
   return result;
}

std::string goal_display_name(
   const SemanticLiteral& literal,
   size_t level,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< std::string >& objects
)
{
   std::string result = literal.positive ? "[+]" : "[-]";
   result += atom_display_name(literal.atom, predicates, objects);
   result += kGoalLevelSuffixes.at(level);
   return result;
}

std::string action_display_name(
   const SemanticGroundAction& action,
   const std::vector< SemanticActionSpec >& actions,
   const std::vector< std::string >& objects
)
{
   std::string result = "(" + actions.at(static_cast< size_t >(action.action)).name + " ";
   for(size_t idx = 0; idx < action.arguments.size(); ++idx) {
      if(idx > 0) {
         result += " ";
      }
      result += objects.at(static_cast< size_t >(action.arguments[idx]));
   }
   result += ")";
   return result;
}

void validate_name(std::string_view name, std::string_view kind)
{
   if(name.empty()) {
      throw std::invalid_argument(std::string(kind) + " name must not be empty");
   }
}

template < typename Spec >
void validate_unique_names(const std::vector< Spec >& specs, std::string_view kind)
{
   std::set< std::string, std::less<> > names;
   for(const auto& spec : specs) {
      validate_name(spec.name, kind);
      if(spec.arity < 0) {
         throw std::invalid_argument(std::string(kind) + " arity must be non-negative");
      }
      if(not names.emplace(spec.name).second) {
         throw std::invalid_argument(
            "Semantic flat encoder requires unique " + std::string(kind) + " names"
         );
      }
   }
}

struct GoalEntityKey {
   TargetSource source = TargetSource::goals;
   SemanticLiteral literal;
   size_t level = 0;

   auto operator<=>(const GoalEntityKey&) const = default;
};

struct SemanticLiteralHash {
   size_t operator()(const SemanticLiteral& literal) const noexcept
   {
      size_t value = SemanticAtomHash{}(literal.atom);
      mix_semantic_hash(value, literal.positive ? 1 : 0);
      return value;
   }
};

struct SemanticGroundActionHash {
   size_t operator()(const SemanticGroundAction& action) const noexcept
   {
      size_t value = 0;
      mix_semantic_hash(value, action.action);
      for(const auto argument : action.arguments) {
         mix_semantic_hash(value, argument);
      }
      return value;
   }
};

struct GoalEntityKeyHash {
   size_t operator()(const GoalEntityKey& key) const noexcept
   {
      size_t value = SemanticLiteralHash{}(key.literal);
      mix_semantic_hash(value, static_cast< int64_t >(key.source));
      mix_semantic_hash(value, static_cast< int64_t >(key.level));
      return value;
   }
};

struct HistoryEntityKey {
   int64_t dt = 0;
   size_t entry_index = 0;
   SemanticLiteral literal;

   auto operator<=>(const HistoryEntityKey&) const = default;
};

struct HistoryEntityKeyHash {
   size_t operator()(const HistoryEntityKey& key) const noexcept
   {
      size_t value = SemanticLiteralHash{}(key.literal);
      mix_semantic_hash(value, key.dt);
      mix_semantic_hash(value, static_cast< int64_t >(key.entry_index));
      return value;
   }
};

struct PreparedHistoryEntry {
   int64_t dt = 0;
   size_t entry_index = 0;
   std::vector< SemanticLiteral > literals;
   int64_t entity_index = -1;
};

struct SemanticEncodingContext {
   int64_t entity_count = 0;
   std::vector< std::string > entity_names;
   std::vector< int64_t > entity_role_ids;
   std::vector< int64_t > object_indices;
   std::vector< int64_t > predicate_entity_indices;
   std::vector< int64_t > state_entity_indices;
   std::vector< int64_t > history_entity_indices;
   std::vector< int64_t > history_entity_dt;
   std::vector< int64_t > target_entity_indices;
   std::vector< int64_t > target_entity_group_ids;
   hash_map< GoalEntityKey, int64_t, GoalEntityKeyHash > goal_entity_indices;
   hash_map< SemanticGroundAction, int64_t, SemanticGroundActionHash > action_entity_indices;
   hash_map< HistoryEntityKey, int64_t, HistoryEntityKeyHash > history_target_entity_indices;
   std::vector< SemanticGroundAction > unique_actions;
   std::vector< PreparedHistoryEntry > history_entries;
   TargetColumns target_columns;
};

bool split_full_state_relations(const SemanticFlatHorizonEncoderConfig& config)
{
   return config.transition_mode == SemanticHorizonMode::full
          and root_uses_split_state_relations(config.root_policy);
}

GraphFieldSpec semantic_stack_field(int dim = 1)
{
   return GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = dim};
}

GraphFieldSpec semantic_cat_field(GraphFieldInc inc = {})
{
   return GraphFieldSpec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::CAT,
      .dim = 1,
      .inc = std::move(inc),
   };
}

GraphFieldSpec semantic_ragged_cat_field()
{
   return GraphFieldSpec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::RAGGED_CAT,
      .dim = 1,
      .cat_dim = 0,
   };
}

GraphFieldInc semantic_entity_inc()
{
   return GraphFieldInc{.kind = GraphFieldInc::Kind::NODE_OFFSET, .node_type = "entity"};
}

GraphFieldInc semantic_relation_instance_inc()
{
   return GraphFieldInc{
      .kind = GraphFieldInc::Kind::FIELD_OFFSET,
      .field_key = std::string(kRelationInstanceSizesField),
   };
}

std::vector< SemanticFlatFieldComponent::FieldDeclaration >
semantic_fields(bool horizon, bool target_metadata, bool include_lgan)
{
   using Field = SemanticFlatFieldComponent::FieldDeclaration;
   std::vector< Field > fields{
      {std::string(kNodeSizesField), semantic_stack_field()},
      {std::string(kObjectSizesField), semantic_stack_field()},
      {std::string(kObjectIndicesField), semantic_cat_field(semantic_entity_inc())},
      {std::string(kEntityRoleIdsField), semantic_cat_field()},
      {std::string(kTargetEntitySizesField), semantic_stack_field()},
      {std::string(kTargetEntityIndicesField), semantic_cat_field(semantic_entity_inc())},
      {std::string(kTargetEntityGroupIdsField), semantic_cat_field()},
   };
   if(not horizon) {
      fields.emplace_back(std::string(kHistoryEntitySizesField), semantic_stack_field());
      fields.emplace_back(
         std::string(kHistoryEntityIndicesField), semantic_cat_field(semantic_entity_inc())
      );
      fields.emplace_back(std::string(kHistoryEntityDtField), semantic_cat_field());
      if(target_metadata) {
         fields.emplace_back(std::string(kTargetSizesField), semantic_stack_field());
      }
   } else {
      fields.emplace_back(std::string(kTargetSizesField), semantic_stack_field());
      fields.emplace_back(
         std::string(kTargetPositionsField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::RAGGED_CAT,
            .dim = 1,
            .cat_dim = 0,
            .inc = semantic_entity_inc(),
         }
      );
      fields.emplace_back(std::string(kTargetIndicesField), semantic_ragged_cat_field());
      fields.emplace_back(std::string(kTargetCandidateIdsField), semantic_ragged_cat_field());
      fields.emplace_back(std::string(kTargetDepthsField), semantic_ragged_cat_field());
      fields.emplace_back(std::string(kTargetGroupIdsField), semantic_ragged_cat_field());
   }
   if(target_metadata and not horizon) {
      fields.emplace_back(
         std::string(kTargetPositionsField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::RAGGED_CAT,
            .dim = 1,
            .cat_dim = 0,
            .inc = semantic_entity_inc(),
         }
      );
      fields.emplace_back(std::string(kTargetIndicesField), semantic_ragged_cat_field());
      fields.emplace_back(std::string(kTargetCandidateIdsField), semantic_ragged_cat_field());
      fields.emplace_back(std::string(kTargetGroupIdsField), semantic_ragged_cat_field());
   }
   if(include_lgan) {
      fields.emplace_back(std::string(kLGANTNSizesField), semantic_stack_field());
      fields.emplace_back(
         std::string(kLGANTNRelationIndicesField),
         semantic_cat_field(semantic_relation_instance_inc())
      );
      fields.emplace_back(
         std::string(kLGANTNEntityIndicesField), semantic_cat_field(semantic_entity_inc())
      );
      fields.emplace_back(std::string(kLGANNNSizesField), semantic_stack_field());
      fields.emplace_back(
         std::string(kLGANNNRelationIndicesField),
         semantic_cat_field(semantic_relation_instance_inc())
      );
      fields.emplace_back(
         std::string(kLGANNNEntityIndicesField), semantic_cat_field(semantic_entity_inc())
      );
      fields.emplace_back(std::string(kLGANRRSizesField), semantic_stack_field());
      fields.emplace_back(
         std::string(kLGANRRSrcRelationIndicesField),
         semantic_cat_field(semantic_relation_instance_inc())
      );
      fields.emplace_back(
         std::string(kLGANRRDstRelationIndicesField),
         semantic_cat_field(semantic_relation_instance_inc())
      );
   }
   return fields;
}

RelationUsage semantic_relation_usage(std::string_view source)
{
   if(source == "state")
      return RelationUsage::state;
   if(source == "goal")
      return RelationUsage::goal;
   if(source == "goal_derivation")
      return RelationUsage::goal_derivation;
   if(source == "goal_satisfaction")
      return RelationUsage::goal_satisfaction;
   if(source == "action")
      return RelationUsage::action;
   if(source == "history")
      return RelationUsage::history;
   if(source == "parent")
      return RelationUsage::parent;
   if(source == "sibling")
      return RelationUsage::sibling;
   if(source == "cousin")
      return RelationUsage::cousin;
   return RelationUsage::state;
}

std::string semantic_relation_component(RelationUsage usage)
{
   switch(usage) {
      case RelationUsage::state: return "semantic_facts";
      case RelationUsage::goal: return "semantic_goals";
      case RelationUsage::goal_derivation:
      case RelationUsage::goal_satisfaction: return "semantic_derivations";
      case RelationUsage::action: return "semantic_actions";
      case RelationUsage::history: return "semantic_history";
      case RelationUsage::parent:
      case RelationUsage::sibling:
      case RelationUsage::cousin: return "semantic_topology";
   }
   return "semantic_facts";
}

std::vector< FlatCompositionRelationSpec > semantic_relation_specs(const FlatRelationSchema& schema)
{
   std::vector< FlatCompositionRelationSpec > specs;
   const auto& metadata = schema.as_metadata();
   for(size_t id = 0; id < schema.size(); ++id) {
      const auto offset = static_cast< size_t >(metadata.relation_slot_role_offsets[id]);
      const auto encoded = static_cast< size_t >(metadata.relation_encoded_arities[id]);
      const auto logical = static_cast< int >(metadata.relation_logical_arities[id]);
      std::vector< FlatSlotRole > auxiliary;
      bool predicate = false;
      for(size_t slot = 0; slot < encoded; ++slot) {
         const auto role = static_cast< FlatSlotRole >(metadata.relation_slot_roles[offset + slot]);
         if(role == FlatSlotRole::argument_slot) {
            continue;
         }
         if(role == FlatSlotRole::predicate_slot) {
            predicate = true;
         } else {
            auxiliary.push_back(role);
         }
      }
      specs.push_back(
         FlatCompositionRelationSpec{
            .key = opaque_relation_key(metadata.relation_names[id]),
            .layout =
               FlatTupleLayout{
                  .logical_arity = logical,
                  .auxiliary_slot_roles = std::move(auxiliary),
                  .include_predicate_virtual_node = predicate,
               },
            .usage = semantic_relation_usage(metadata.relation_sources[id]),
         }
      );
   }
   return specs;
}

void set_semantic_carrier_field(
   SemanticFlatCompositionInput& carrier,
   std::string_view key,
   std::span< const int64_t > values
)
{
   carrier.composition.fields.push_back(
      FlatCompositionFieldRecord{
         .key = std::string(key),
         .values = std::vector< int64_t >(values.begin(), values.end()),
      }
   );
}

void append_semantic_carrier_relations(
   SemanticFlatCompositionInput& carrier,
   const FlatRelationSchema& schema,
   const FlatRelationSink& sink
)
{
   const auto& args = sink.relation_args();
   size_t offset = 0;
   for(size_t relation = 0; relation < schema.size(); ++relation) {
      const auto count = static_cast< size_t >(sink.relation_counts().at(relation));
      const auto arity = static_cast< size_t >(schema.arities().at(relation));
      for(size_t instance = 0; instance < count; ++instance) {
         const auto begin = args.begin() + static_cast< std::ptrdiff_t >(offset);
         carrier.composition.relations.push_back(
            FlatCompositionRelationRecord{
               .relation_id = static_cast< int >(relation),
               .args = std::vector< int64_t >(begin, begin + arity),
               .component = semantic_relation_component(
                  semantic_relation_usage(schema.as_metadata().relation_sources[relation])
               ),
            }
         );
         offset += arity;
      }
   }

   if(offset != args.size()) {
      throw std::invalid_argument("semantic flat relation sink emitted inconsistent argument data");
   }
}

void rebuild_semantic_carrier_indexes(
   SemanticFlatCompositionInput& carrier,
   const FlatRelationSchema& schema
)
{
   carrier.rebuild_indexes();
   for(const auto& source : schema.as_metadata().relation_sources) {
      carrier.relation_indices_by_component.try_emplace(
         semantic_relation_component(semantic_relation_usage(source))
      );
   }
}

}  // namespace

struct SemanticFlatRelationEncoderEngine::Impl {
   struct GoalRelationIds {
      std::array< std::array< int, 3 >, 2 > by_polarity = {
         std::array< int, 3 >{-1, -1, -1},
         std::array< int, 3 >{-1, -1, -1},
      };
   };
   struct HorizonGoalRelationIds {
      std::array< std::array< int, 5 >, 2 > root = {
         std::array< int, 5 >{-1, -1, -1, -1, -1},
         std::array< int, 5 >{-1, -1, -1, -1, -1},
      };
      std::array< std::array< int, 5 >, 2 > candidate = {
         std::array< int, 5 >{-1, -1, -1, -1, -1},
         std::array< int, 5 >{-1, -1, -1, -1, -1},
      };
   };

   struct PreparedRelationGraph {
      const SemanticFlatRelationInput* input = nullptr;
      std::vector< SemanticGoalLevel > goal_levels;
      std::vector< SemanticLiteral > grouped_goals;
      hash_set< SemanticAtom, SemanticAtomHash > fact_keys;
      mutable SemanticEncodingContext context;
      bool suppress_empty_target_names = false;
   };

   enum class RelationLane {
      facts,
      goals,
      actions,
      history,
   };

   class RelationEntityComponent final: public FlatEmitterComponent {
     public:
      explicit RelationEntityComponent(const Impl* owner) : owner_(owner) {}
      [[nodiscard]] std::string_view name() const noexcept override { return "semantic_entities"; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type(
            std::string(kFlatEntityNodeType),
            FlatNodeKind::object,
            1,
            owner_->config.export_node_names
         );
      }
      void plan_graph(const FlatInputView& input, FlatNodePlanBuilder& builder) const override
      {
         const auto& prepared = input.get< PreparedRelationGraph >();
         for(int64_t index = 0; index < prepared.context.entity_count; ++index) {
            const auto key = owner_->config.export_node_names
                                ? prepared.context.entity_names.at(static_cast< size_t >(index))
                                : "entity:" + std::to_string(index);
            (void) builder.add_node_from_source(std::string(kFlatEntityNodeType), index, key);
         }
      }
      void declare_node_features(FlatNodeFeaturePlanBuilder& builder) const override
      {
         builder.register_feature(std::string(kFlatEntityNodeType), "x", 1);
      }
      void write_node_features(
         const FlatGraphContext& context,
         FlatNodeFeatureWriter& writer
      ) const override
      {
         const auto type = context.nodes.schema().id_for(std::string(kFlatEntityNodeType));
         const std::vector< float > values(static_cast< size_t >(context.nodes.count(type)), 0.0F);
         writer.set(std::string(kFlatEntityNodeType), "x", values);
      }

     private:
      const Impl* owner_;
   };

   class RelationLaneComponent final: public FlatEmitterComponent {
     public:
      RelationLaneComponent(
         const Impl* owner,
         RelationLane lane,
         std::string component_name,
         std::vector< FlatCompositionRelationSpec > relations
      )
          : owner_(owner),
            lane_(lane),
            component_name_(std::move(component_name)),
            relations_(std::move(relations))
      {
      }
      [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         for(const auto& relation : relations_) {
            builder.register_relation(relation.key, relation.layout, relation.usage);
         }
      }
      void emit(const FlatInputView& input, FlatGraphContext& context) const override
      {
         owner_->emit_relation_lane(input.get< PreparedRelationGraph >(), lane_, context);
      }

     private:
      const Impl* owner_;
      RelationLane lane_;
      std::string component_name_;
      std::vector< FlatCompositionRelationSpec > relations_;
   };

   class RelationFieldComponent final: public FlatEmitterComponent {
     public:
      RelationFieldComponent(
         const Impl* owner,
         std::string component_name,
         std::vector< SemanticFlatFieldComponent::FieldDeclaration > fields
      )
          : owner_(owner), component_name_(std::move(component_name)), fields_(std::move(fields))
      {
      }
      [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
      void declare_fields(FlatFieldPlanBuilder& builder) const override
      {
         for(const auto& [key, spec] : fields_) {
            builder.register_field(key, spec);
         }
      }
      void write_fields(const FlatGraphContext& context, FlatFieldWriter& writer) const override
      {
         owner_->write_relation_fields(
            context.input.get< PreparedRelationGraph >(), fields_, context, writer
         );
      }

     private:
      const Impl* owner_;
      std::string component_name_;
      std::vector< SemanticFlatFieldComponent::FieldDeclaration > fields_;
   };

   class RelationMetadataComponent final: public FlatEmitterComponent {
     public:
      explicit RelationMetadataComponent(const Impl* owner) : owner_(owner) {}
      [[nodiscard]] std::string_view name() const noexcept override { return "semantic_metadata"; }
      void declare_metadata(FlatMetadataPlanBuilder& builder) const override
      {
         if(owner_->config.export_node_names) {
            builder.claim_object_names();
         }
         if(not owner_->target_group_names.empty() and owner_->config.export_node_names) {
            builder.claim_optional_graph_attr(std::string(kTargetNamesAttr));
         }
      }
      void
      write_metadata(const FlatGraphContext& context, FlatMetadataWriter& writer) const override
      {
         owner_->write_relation_metadata(context.input.get< PreparedRelationGraph >(), writer);
      }

     private:
      const Impl* owner_;
   };

   struct PreparedHorizonNode {
      hash_set< SemanticAtom, SemanticAtomHash > fact_keys;
      std::vector< SemanticLiteral > deltas;
      std::set< SemanticAtom > added_fluent;
      std::set< SemanticAtom > removed_fluent;
      std::set< SemanticAtom > added_derived;
      std::set< SemanticAtom > removed_derived;
   };

   struct PreparedHorizonGraph {
      const SemanticTransitionDAG* dag = nullptr;
      const SemanticFlatHorizonEncoderConfig* config = nullptr;
      std::vector< SemanticGoalLevel > goal_levels;
      std::vector< SemanticLiteral > goals;
      std::vector< PreparedHorizonNode > nodes;
      mutable SemanticEncodingContext context;
      bool suppress_empty_target_names = false;
   };

   class HorizonEntityComponent final: public FlatEmitterComponent {
     public:
      HorizonEntityComponent(const Impl* owner, bool export_names)
          : owner_(owner), export_names_(export_names)
      {
      }
      [[nodiscard]] std::string_view name() const noexcept override { return "semantic_entities"; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type(
            std::string(kFlatEntityNodeType), FlatNodeKind::object, 1, export_names_
         );
      }
      void plan_graph(const FlatInputView& input, FlatNodePlanBuilder& builder) const override
      {
         const auto& prepared = input.get< PreparedHorizonGraph >();
         for(int64_t index = 0; index < prepared.context.entity_count; ++index) {
            const auto key = prepared.config->export_node_names
                                ? prepared.context.entity_names.at(static_cast< size_t >(index))
                                : "entity:" + std::to_string(index);
            (void) builder.add_node_from_source(std::string(kFlatEntityNodeType), index, key);
         }
      }
      void declare_node_features(FlatNodeFeaturePlanBuilder& builder) const override
      {
         builder.register_feature(std::string(kFlatEntityNodeType), "x", 1);
      }
      void write_node_features(
         const FlatGraphContext& context,
         FlatNodeFeatureWriter& writer
      ) const override
      {
         const auto type = context.nodes.schema().id_for(std::string(kFlatEntityNodeType));
         const std::vector< float > values(static_cast< size_t >(context.nodes.count(type)), 0.0F);
         writer.set(std::string(kFlatEntityNodeType), "x", values);
      }

     private:
      const Impl* owner_;
      bool export_names_ = false;
   };

   class HorizonRelationComponent final: public FlatEmitterComponent {
     public:
      HorizonRelationComponent(
         const Impl* owner,
         std::string component_name,
         std::vector< FlatCompositionRelationSpec > relations,
         bool topology
      )
          : owner_(owner),
            component_name_(std::move(component_name)),
            relations_(std::move(relations)),
            topology_(topology)
      {
      }
      [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         for(const auto& relation : relations_) {
            builder.register_relation(relation.key, relation.layout, relation.usage);
         }
      }
      void emit(const FlatInputView& input, FlatGraphContext& context) const override
      {
         const auto& prepared = input.get< PreparedHorizonGraph >();
         if(topology_) {
            owner_->emit_horizon_topology(prepared, context);
         } else {
            owner_->emit_horizon_semantics(prepared, context);
         }
      }

     private:
      const Impl* owner_;
      std::string component_name_;
      std::vector< FlatCompositionRelationSpec > relations_;
      bool topology_ = false;
   };

   class HorizonFieldComponent final: public FlatEmitterComponent {
     public:
      HorizonFieldComponent(
         const Impl* owner,
         std::string component_name,
         std::vector< SemanticFlatFieldComponent::FieldDeclaration > fields
      )
          : owner_(owner), component_name_(std::move(component_name)), fields_(std::move(fields))
      {
      }
      [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
      void declare_fields(FlatFieldPlanBuilder& builder) const override
      {
         for(const auto& [key, spec] : fields_) {
            builder.register_field(key, spec);
         }
      }
      void write_fields(const FlatGraphContext& context, FlatFieldWriter& writer) const override
      {
         owner_->write_horizon_fields(
            context.input.get< PreparedHorizonGraph >(), fields_, context, writer
         );
      }

     private:
      const Impl* owner_;
      std::string component_name_;
      std::vector< SemanticFlatFieldComponent::FieldDeclaration > fields_;
   };

   class HorizonMetadataComponent final: public FlatEmitterComponent {
     public:
      HorizonMetadataComponent(const Impl* owner, bool export_names)
          : owner_(owner), export_names_(export_names)
      {
      }
      [[nodiscard]] std::string_view name() const noexcept override { return "semantic_metadata"; }
      void declare_metadata(FlatMetadataPlanBuilder& builder) const override
      {
         if(export_names_) {
            builder.claim_object_names();
            builder.claim_optional_graph_attr(std::string(kTargetNamesAttr));
         }
      }
      void
      write_metadata(const FlatGraphContext& context, FlatMetadataWriter& writer) const override
      {
         owner_->write_horizon_metadata(context.input.get< PreparedHorizonGraph >(), writer);
      }

     private:
      const Impl* owner_;
      bool export_names_ = false;
   };

   Config config;
   std::shared_ptr< const SemanticTaskContext > task_context;
   const std::vector< SemanticPredicateSpec >& predicates;
   const std::vector< SemanticActionSpec >& actions;
   FlatRelationSchema schema_;
   std::vector< std::string > target_entity_group_names;
   std::map< TargetSource, int64_t > target_entity_group_ids;
   std::vector< std::string > target_group_names;
   std::map< TargetSource, int64_t > target_group_ids;
   std::vector< int > state_relation_ids;
   std::vector< int > action_relation_ids;
   std::vector< std::array< int, 2 > > history_relation_ids;
   std::vector< std::vector< GoalRelationIds > > goal_relation_ids;
   std::vector< int > horizon_state_relation_ids;
   std::vector< int > horizon_state_anchored_relation_ids;
   std::vector< int > horizon_action_relation_ids;
   std::vector< std::array< int, 2 > > horizon_literal_relation_ids;
   std::vector< std::vector< HorizonGoalRelationIds > > horizon_goal_relation_ids;
   int horizon_parent_relation_id = -1;
   int horizon_sibling_relation_id = -1;
   int horizon_cousin_relation_id = -1;
   std::unique_ptr< CompiledFlatPlan > composition_plan;

   Impl(
      std::vector< SemanticPredicateSpec > predicate_specs,
      std::vector< SemanticActionSpec > action_specs,
      Config encoder_config
   )
       : Impl(
            std::make_shared< SemanticTaskContext >(SemanticTaskContext{
               .predicates = std::move(predicate_specs),
               .actions = std::move(action_specs),
            }),
            std::move(encoder_config)
         )
   {
   }

   Impl(std::shared_ptr< const SemanticTaskContext > context, Config encoder_config)
       : config(std::move(encoder_config)),
         task_context(require_task_context(context, "Semantic flat")),
         predicates(task_context->predicates),
         actions(task_context->actions)
   {
      validate_config();
      validate_unique_names(predicates, "predicate");
      validate_unique_names(actions, "action");
      build_groups();
      build_schema();
      build_relation_ids();
      build_composition_plan(false, nullptr);
   }

   void build_composition_plan(bool horizon, const SemanticFlatHorizonEncoderConfig* horizon_config)
   {
      FlatEncoderPlan plan;
      if(horizon) {
         plan.add_component(
            std::make_shared< HorizonEntityComponent >(
               this,
               horizon_config != nullptr ? horizon_config->export_node_names
                                         : config.export_node_names
            )
         );
      } else {
         plan.add_component(std::make_shared< RelationEntityComponent >(this));
      }
      const auto specs = semantic_relation_specs(schema_);
      std::array< std::vector< FlatCompositionRelationSpec >, 6 > lanes;
      for(const auto& spec : specs) {
         const auto component = semantic_relation_component(spec.usage);
         const auto index = component == "semantic_facts"         ? 0
                            : component == "semantic_goals"       ? 1
                            : component == "semantic_derivations" ? 2
                            : component == "semantic_actions"     ? 3
                            : component == "semantic_history"     ? 4
                                                                  : 5;
         lanes.at(index).push_back(spec);
      }
      constexpr std::array< std::string_view, 6 > names = {
         "semantic_facts",
         "semantic_goals",
         "semantic_derivations",
         "semantic_actions",
         "semantic_history",
         "semantic_topology",
      };
      if(not horizon) {
         lanes[1].insert(
            lanes[1].end(),
            std::make_move_iterator(lanes[2].begin()),
            std::make_move_iterator(lanes[2].end())
         );
         lanes[2].clear();
      } else {
         for(size_t index = 1; index < 5; ++index) {
            lanes[0].insert(
               lanes[0].end(),
               std::make_move_iterator(lanes[index].begin()),
               std::make_move_iterator(lanes[index].end())
            );
            lanes[index].clear();
         }
      }
      for(size_t index = 0; index < lanes.size(); ++index) {
         if(not lanes[index].empty()) {
            if(horizon) {
               plan.add_component(
                  std::make_shared< HorizonRelationComponent >(
                     this,
                     index == 5 ? "semantic_topology" : "semantic_transitions",
                     std::move(lanes[index]),
                     index == 5
                  )
               );
            } else {
               const auto lane = index == 0                 ? RelationLane::facts
                                 : index == 1 or index == 2 ? RelationLane::goals
                                 : index == 3               ? RelationLane::actions
                                                            : RelationLane::history;
               plan.add_component(
                  std::make_shared< RelationLaneComponent >(
                     this, lane, std::string(names[index]), std::move(lanes[index])
                  )
               );
            }
         }
      }

      const bool target_metadata = horizon or not target_group_names.empty();
      auto fields = semantic_fields(
         horizon,
         target_metadata,
         horizon_config != nullptr ? horizon_config->include_lgan_edges : config.include_lgan_edges
      );
      std::vector< SemanticFlatFieldComponent::FieldDeclaration > history_fields;
      std::vector< SemanticFlatFieldComponent::FieldDeclaration > target_fields;
      std::vector< SemanticFlatFieldComponent::FieldDeclaration > lgan_fields;
      std::vector< SemanticFlatFieldComponent::FieldDeclaration > effect_fields;
      for(auto& field : fields) {
         const auto& key = field.first;
         if(key.starts_with("history_")) {
            history_fields.push_back(std::move(field));
         } else if(key.starts_with("target_")) {
            target_fields.push_back(std::move(field));
         } else if(key.starts_with("lgan_")) {
            lgan_fields.push_back(std::move(field));
         } else {
            effect_fields.push_back(std::move(field));
         }
      }
      if(not effect_fields.empty()) {
         if(horizon) {
            plan.add_component(
               std::make_shared< HorizonFieldComponent >(
                  this, "semantic_effects", std::move(effect_fields)
               )
            );
         } else {
            plan.add_component(
               std::make_shared< RelationFieldComponent >(
                  this, "semantic_effects", std::move(effect_fields)
               )
            );
         }
      }
      if(not history_fields.empty()) {
         if(horizon) {
            plan.add_component(
               std::make_shared< HorizonFieldComponent >(
                  this, "semantic_history_fields", std::move(history_fields)
               )
            );
         } else {
            plan.add_component(
               std::make_shared< RelationFieldComponent >(
                  this, "semantic_history_fields", std::move(history_fields)
               )
            );
         }
      }
      if(not target_fields.empty()) {
         if(horizon) {
            plan.add_component(
               std::make_shared< HorizonFieldComponent >(
                  this, "semantic_targets", std::move(target_fields)
               )
            );
         } else {
            plan.add_component(
               std::make_shared< RelationFieldComponent >(
                  this, "semantic_targets", std::move(target_fields)
               )
            );
         }
      }
      if(not lgan_fields.empty()) {
         if(horizon) {
            plan.add_component(
               std::make_shared< HorizonFieldComponent >(
                  this, "semantic_lgan_fields", std::move(lgan_fields)
               )
            );
         } else {
            plan.add_component(
               std::make_shared< RelationFieldComponent >(
                  this, "semantic_lgan_fields", std::move(lgan_fields)
               )
            );
         }
      }
      std::vector< std::string > metadata_keys;
      std::vector< std::string > optional_metadata_keys;
      if(target_metadata) {
         metadata_keys.emplace_back(kTargetGroupsAttr);
         const bool export_names = horizon_config != nullptr ? horizon_config->export_node_names
                                                             : config.export_node_names;
         if(export_names) {
            optional_metadata_keys.emplace_back(kTargetNamesAttr);
         }
      }
      if(horizon_config != nullptr) {
         metadata_keys.emplace_back(kParentRelationAttr);
      }
      if(horizon) {
         plan.add_component(
            std::make_shared< HorizonMetadataComponent >(
               this, horizon_config != nullptr and horizon_config->export_node_names
            )
         );
      } else {
         plan.add_component(std::make_shared< RelationMetadataComponent >(this));
      }

      FlatCompositionConfig composition_config;
      composition_config.max_goal_level = horizon_config != nullptr ? horizon_config->max_goal_level
                                                                    : config.max_goal_level;
      composition_config.support_literals = horizon_config != nullptr
                                               ? horizon_config->support_literals
                                               : config.support_literals;
      composition_config.goal_derivations = horizon_config != nullptr
                                               ? horizon_config->goal_derivations
                                               : config.goal_derivations;
      composition_config.relation_args_node_type = std::string(kFlatEntityNodeType);
      composition_config.entity_node_type = std::string(kFlatEntityNodeType);
      composition_config.track_relation_instances = horizon_config != nullptr
                                                       ? horizon_config->include_lgan_edges
                                                       : config.include_lgan_edges;
      composition_config
         .pack_relation_args_relation_major = horizon_config != nullptr
                                                 ? horizon_config->pack_relation_args_relation_major
                                                 : config.pack_relation_args_relation_major;
      composition_config.graph_config.include_lgan_edges = composition_config
                                                              .track_relation_instances;
      if(horizon_config == nullptr) {
         composition_config.graph_config.target_sources = source_names_for(config.target_sources);
         composition_config.graph_config.lgan_anchor_sources = source_names_for(
            config.lgan_anchor_sources
         );
      }
      composition_config.graph_config
         .target_entity_group_names = horizon_config != nullptr
                                         ? std::vector< std::string >{std::string(
                                              target_source_group_name(TargetSource::states)
                                           )}
                                         : target_entity_group_names;
      composition_config.graph_config
         .target_symbol_prefix = horizon_config != nullptr
                                    ? std::optional{horizon_config->target_symbol_prefix}
                                    : std::optional{config.target_symbol_prefix};
      composition_config.graph_config
         .use_predicate_virtual_nodes = horizon_config != nullptr
                                           ? horizon_config->use_predicate_virtual_nodes
                                           : config.use_predicate_virtual_nodes;
      composition_config.graph_config.lgan_tn_edge_pos = horizon_config != nullptr
                                                            ? horizon_config->lgan_tn_edge_pos
                                                            : config.lgan_tn_edge_pos;
      composition_config.graph_config.lgan_nn_edge_pos = horizon_config != nullptr
                                                            ? horizon_config->lgan_nn_edge_pos
                                                            : config.lgan_nn_edge_pos;
      composition_config.graph_config.lgan_rr_edge_pos = horizon_config != nullptr
                                                            ? horizon_config->lgan_rr_edge_pos
                                                            : config.lgan_rr_edge_pos;
      composition_plan = std::make_unique< CompiledFlatPlan >(
         std::move(plan).compile(composition_config)
      );
   }

   static size_t goal_derivation_index(std::optional< GoalDerivation > derivation)
   {
      if(not derivation.has_value() or *derivation == GoalDerivation::plain) {
         return 0;
      }
      if(*derivation == GoalDerivation::satisfied) {
         return 1;
      }
      if(*derivation == GoalDerivation::unsatisfied) {
         return 2;
      }
      throw std::invalid_argument("unsupported semantic Flat goal derivation");
   }

   static size_t horizon_goal_derivation_index(std::optional< GoalDerivation > derivation)
   {
      if(not derivation.has_value() or *derivation == GoalDerivation::plain) {
         return 0;
      }
      switch(*derivation) {
         case GoalDerivation::satisfied: return 1;
         case GoalDerivation::unsatisfied: return 2;
         case GoalDerivation::added_satisfied: return 3;
         case GoalDerivation::added_unsatisfied: return 4;
         case GoalDerivation::plain: break;
      }
      throw std::invalid_argument("unsupported semantic Flat Horizon goal derivation");
   }

   int relation_id_at_construction(const RelationKey& key) const
   {
      const auto id = schema_.try_id_for(key);
      if(not id.has_value()) {
         throw std::invalid_argument("missing precomputed semantic Flat relation");
      }
      return *id;
   }

   int optional_relation_id_at_construction(const RelationKey& key) const
   {
      return schema_.try_id_for(key).value_or(-1);
   }

   int required_relation_id(int id) const
   {
      if(id < 0) {
         throw std::invalid_argument("semantic Flat input requires a disabled relation");
      }
      return id;
   }

   void build_relation_ids()
   {
      state_relation_ids.assign(predicates.size(), -1);
      history_relation_ids.assign(predicates.size(), std::array< int, 2 >{-1, -1});
      goal_relation_ids.assign(
         predicates.size(), std::vector< GoalRelationIds >(config.max_goal_level + 1)
      );
      for(size_t predicate_index = 0; predicate_index < predicates.size(); ++predicate_index) {
         const auto& predicate = predicates[predicate_index];
         if(config.ignore_zero_arity_relations and predicate.arity == 0) {
            continue;
         }
         state_relation_ids[predicate_index] = relation_id_at_construction(
            predicate_relation_key(predicate.name)
         );
         for(const bool positive : {false, true}) {
            const auto polarity_index = positive ? size_t{1} : size_t{0};
            history_relation_ids[predicate_index][polarity_index] = relation_id_at_construction(
               predicate_relation_key(
                  predicate.name, positive, std::nullopt, std::nullopt, "[hist]"
               )
            );
            if(kTopTypePredicates.contains(predicate.name)) {
               continue;
            }
            for(size_t level = 0; level <= config.max_goal_level; ++level) {
               for(const auto derivation : {
                      std::optional< GoalDerivation >{},
                      std::optional< GoalDerivation >{GoalDerivation::satisfied},
                      std::optional< GoalDerivation >{GoalDerivation::unsatisfied},
                   }) {
                  const auto slot = goal_derivation_index(derivation);
                  const auto key = predicate_relation_key(
                     predicate.name, positive, GoalLevel(level), derivation
                  );
                  if(const auto id = schema_.try_id_for(key); id.has_value()) {
                     goal_relation_ids[predicate_index][level]
                        .by_polarity[polarity_index][slot] = *id;
                  }
               }
            }
         }
      }
      action_relation_ids.resize(actions.size());
      for(size_t action_index = 0; action_index < actions.size(); ++action_index) {
         action_relation_ids[action_index] = relation_id_at_construction(
            action_relation_key(actions[action_index].name)
         );
      }
   }

   void validate_config() const
   {
      if(config.max_goal_level >= kGoalLevelSuffixes.size()) {
         throw std::invalid_argument("Semantic flat max_goal_level must be in [0, 3]");
      }
      auto validate_sources = [](const std::set< TargetSource >& sources, std::string_view field) {
         for(const auto source : sources) {
            if(source == TargetSource::goals or source == TargetSource::subgoals
               or source == TargetSource::actions or source == TargetSource::history) {
               continue;
            }
            throw std::invalid_argument(
               "Semantic flat " + std::string(field)
               + " supports action, goal, subgoal, and history only"
            );
         }
      };
      validate_sources(config.target_sources, "target_sources");
      validate_sources(config.lgan_anchor_sources, "lgan_anchor_sources");
      for(const auto derivation : config.goal_derivations) {
         if(not supports_semantic_goal_derivation(derivation)) {
            throw std::invalid_argument(
               "Semantic flat encoder supports plain/satisfied/unsatisfied goal derivations only"
            );
         }
      }
   }

   void build_groups()
   {
      auto append = [](std::vector< std::string >& names,
                       std::map< TargetSource, int64_t >& ids,
                       TargetSource source) {
         ids.emplace(source, static_cast< int64_t >(names.size()));
         names.emplace_back(target_source_group_name(source));
      };
      for(const auto source : kCanonicalTargetSourceOrder) {
         if(source == TargetSource::states) {
            continue;
         }
         if(source == TargetSource::actions or has_anchor_entity_source(config, source)) {
            append(target_entity_group_names, target_entity_group_ids, source);
         }
         if(has_target_source(config, source)) {
            append(target_group_names, target_group_ids, source);
         }
      }
   }

   void build_schema()
   {
      FlatRelationSchemaBuilder builder;
      for(const auto& predicate : predicates) {
         const auto arity = static_cast< int >(predicate.arity);
         if(config.ignore_zero_arity_relations and arity == 0) {
            continue;
         }
         builder.register_relation(
            predicate_relation_key(predicate.name),
            make_predicate_tuple_layout(arity, {}, config.use_predicate_virtual_nodes),
            RelationUsage::state
         );
         if(not kTopTypePredicates.contains(predicate.name)) {
            if(config.goal_derivations.contains(GoalDerivation::plain)) {
               for(size_t level = 0; level <= config.max_goal_level; ++level) {
                  for(const bool positive : {true, false}) {
                     builder.register_relation(
                        predicate_relation_key(predicate.name, positive, GoalLevel(level)),
                        semantic_goal_layout(config, arity, level),
                        RelationUsage::goal
                     );
                  }
               }
               if(config.support_literals) {
                  for(const bool positive : {true, false}) {
                     std::vector< FlatSlotRole > roles;
                     if(has_anchor_entity_source(config, TargetSource::goals)) {
                        roles.push_back(FlatSlotRole::goal_target_slot);
                     }
                     builder.register_relation(
                        predicate_relation_key(predicate.name, positive),
                        make_predicate_tuple_layout(
                           arity, std::span{roles}, config.use_predicate_virtual_nodes
                        ),
                        RelationUsage::goal
                     );
                  }
               }
            }
            for(const auto derivation : config.goal_derivations) {
               if(derivation == GoalDerivation::plain) {
                  continue;
               }
               for(size_t level = 0; level <= config.max_goal_level; ++level) {
                  for(const bool positive : {true, false}) {
                     builder.register_relation(
                        predicate_relation_key(
                           predicate.name, positive, GoalLevel(level), derivation
                        ),
                        make_predicate_tuple_layout(arity, {}, config.use_predicate_virtual_nodes),
                        RelationUsage::goal_derivation
                     );
                  }
               }
               if(config.support_literals) {
                  for(const bool positive : {true, false}) {
                     builder.register_relation(
                        predicate_relation_key(predicate.name, positive, std::nullopt, derivation),
                        make_predicate_tuple_layout(arity, {}, config.use_predicate_virtual_nodes),
                        RelationUsage::goal_derivation
                     );
                  }
               }
            }
         }
         for(const bool positive : {true, false}) {
            builder.register_relation(
               predicate_relation_key(
                  predicate.name, positive, std::nullopt, std::nullopt, "[hist]"
               ),
               semantic_history_layout(config, arity),
               RelationUsage::history
            );
         }
      }
      for(const auto& action : actions) {
         builder.register_relation(
            action_relation_key(action.name),
            make_nonpredicate_tuple_layout(
               static_cast< int >(action.arity), {FlatSlotRole::action_slot}
            ),
            RelationUsage::action
         );
      }
      schema_ = std::move(builder).finalize(
         static_cast< int >(config.max_goal_level),
         config.support_literals,
         config.goal_derivations,
         "SemanticFlatRelationEncoderEngine requires at least one relation"
      );
   }

   void configure_horizon(const SemanticFlatHorizonEncoderConfig& horizon)
   {
      if(horizon.max_goal_level >= kGoalLevelSuffixes.size()) {
         throw std::invalid_argument("Semantic flat Horizon max_goal_level must be in [0, 3]");
      }
      if(horizon.transition_mode == SemanticHorizonMode::action and horizon.ignore_actions) {
         throw std::invalid_argument("Action flat horizon encoding requires ignore_actions=false.");
      }

      FlatRelationSchemaBuilder builder;
      const bool root_state_slot = root_in_state_relations(horizon.root_policy);
      const bool split_candidates = split_full_state_relations(horizon);
      auto predicate_layout = [&](int arity, bool state_slot) {
         const std::array roles = {FlatSlotRole::state_slot};
         return make_predicate_tuple_layout(
            arity,
            state_slot ? std::span{roles} : std::span< const FlatSlotRole >{},
            horizon.use_predicate_virtual_nodes
         );
      };
      auto add_root = [&](RelationKey key, int arity, RelationUsage usage) {
         builder.register_relation(std::move(key), predicate_layout(arity, root_state_slot), usage);
      };
      auto add_full = [&](const RelationKey& key, int arity, RelationUsage usage) {
         add_root(key, arity, usage);
         if(split_candidates) {
            RelationKey anchored = key;
            anchored.state_anchored = true;
            builder.register_relation(std::move(anchored), predicate_layout(arity, true), usage);
         }
      };
      auto add_candidate = [&](RelationKey key, int arity, RelationUsage usage) {
         builder.register_relation(std::move(key), predicate_layout(arity, true), usage);
      };

      for(const auto& predicate : predicates) {
         const int arity = static_cast< int >(predicate.arity);
         add_full(predicate_relation_key(predicate.name), arity, RelationUsage::state);
         if(kTopTypePredicates.contains(predicate.name)) {
            continue;
         }
         if(horizon.goal_derivations.contains(GoalDerivation::plain)) {
            for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
               for(const bool positive : {true, false}) {
                  add_root(
                     predicate_relation_key(predicate.name, positive, GoalLevel(level)),
                     arity,
                     RelationUsage::goal
                  );
               }
            }
         }
         if(horizon.support_literals) {
            for(const bool positive : {true, false}) {
               auto key = predicate_relation_key(predicate.name, positive);
               if(horizon.transition_mode == SemanticHorizonMode::delta) {
                  add_candidate(std::move(key), arity, RelationUsage::state);
               } else {
                  add_root(std::move(key), arity, RelationUsage::state);
               }
            }
         }
         for(const auto derivation : horizon.goal_derivations) {
            if(derivation == GoalDerivation::plain) {
               continue;
            }
            const bool root_derivation = derivation == GoalDerivation::satisfied
                                         or derivation == GoalDerivation::unsatisfied;
            const bool delta_derivation = derivation == GoalDerivation::added_satisfied
                                          or derivation == GoalDerivation::added_unsatisfied;
            if(root_derivation) {
               for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
                  for(const bool positive : {true, false}) {
                     add_root(
                        predicate_relation_key(
                           predicate.name, positive, GoalLevel(level), derivation
                        ),
                        arity,
                        RelationUsage::goal_satisfaction
                     );
                  }
               }
               if(horizon.support_literals) {
                  for(const bool positive : {true, false}) {
                     add_root(
                        predicate_relation_key(predicate.name, positive, std::nullopt, derivation),
                        arity,
                        RelationUsage::goal_satisfaction
                     );
                  }
               }
               if(horizon.transition_mode == SemanticHorizonMode::full and split_candidates) {
                  for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
                     for(const bool positive : {true, false}) {
                        builder.register_relation(
                           predicate_relation_key(
                              predicate.name,
                              positive,
                              GoalLevel(level),
                              derivation,
                              /*modifier=*/"",
                              /*state_anchored=*/true
                           ),
                           predicate_layout(arity, true),
                           RelationUsage::goal_satisfaction
                        );
                     }
                  }
                  if(horizon.support_literals) {
                     for(const bool positive : {true, false}) {
                        builder.register_relation(
                           predicate_relation_key(
                              predicate.name,
                              positive,
                              std::nullopt,
                              derivation,
                              /*modifier=*/"",
                              /*state_anchored=*/true
                           ),
                           predicate_layout(arity, true),
                           RelationUsage::goal_satisfaction
                        );
                     }
                  }
               }
            }
            if(horizon.transition_mode == SemanticHorizonMode::delta and delta_derivation) {
               for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
                  for(const bool positive : {true, false}) {
                     add_candidate(
                        predicate_relation_key(
                           predicate.name, positive, GoalLevel(level), derivation
                        ),
                        arity,
                        RelationUsage::goal_satisfaction
                     );
                  }
               }
               if(horizon.support_literals) {
                  for(const bool positive : {true, false}) {
                     add_candidate(
                        predicate_relation_key(predicate.name, positive, std::nullopt, derivation),
                        arity,
                        RelationUsage::goal_satisfaction
                     );
                  }
               }
            }
         }
      }
      if(not horizon.ignore_actions) {
         for(const auto& action : actions) {
            builder.register_relation(
               action_relation_key(action.name),
               make_nonpredicate_tuple_layout(
                  static_cast< int >(action.arity), {FlatSlotRole::state_slot}
               ),
               RelationUsage::action
            );
         }
      }
      const auto topology_layout = make_nonpredicate_tuple_layout(
         0, {FlatSlotRole::state_slot, FlatSlotRole::state_slot}
      );
      if(horizon.enable_parent_relation) {
         builder.register_relation(
            opaque_relation_key(horizon.parent_relation), topology_layout, RelationUsage::parent
         );
      }
      if(horizon.enable_sibling_relation) {
         builder.register_relation(
            opaque_relation_key(horizon.sibling_relation), topology_layout, RelationUsage::sibling
         );
      }
      if(horizon.enable_cousin_relation) {
         builder.register_relation(
            opaque_relation_key(horizon.cousin_relation), topology_layout, RelationUsage::cousin
         );
      }
      schema_ = std::move(builder).finalize(
         static_cast< int >(horizon.max_goal_level),
         horizon.support_literals,
         horizon.goal_derivations,
         "SemanticFlatHorizonEncoderEngine requires at least one relation"
      );
      build_horizon_relation_ids(horizon);
      build_composition_plan(true, &horizon);
   }

   void build_horizon_relation_ids(const SemanticFlatHorizonEncoderConfig& horizon)
   {
      const bool split_candidates = split_full_state_relations(horizon);
      horizon_state_relation_ids.assign(predicates.size(), -1);
      horizon_state_anchored_relation_ids.assign(predicates.size(), -1);
      horizon_literal_relation_ids.assign(predicates.size(), std::array< int, 2 >{-1, -1});
      horizon_goal_relation_ids.assign(
         predicates.size(), std::vector< HorizonGoalRelationIds >(horizon.max_goal_level + 1)
      );

      for(size_t predicate_index = 0; predicate_index < predicates.size(); ++predicate_index) {
         const auto& predicate = predicates[predicate_index];
         const int state_id = relation_id_at_construction(predicate_relation_key(predicate.name));
         horizon_state_relation_ids[predicate_index] = state_id;
         horizon_state_anchored_relation_ids[predicate_index] = split_candidates
                                                                   ? relation_id_at_construction(
                                                                        predicate_relation_key(
                                                                           predicate.name,
                                                                           std::nullopt,
                                                                           std::nullopt,
                                                                           std::nullopt,
                                                                           "",
                                                                           true
                                                                        )
                                                                     )
                                                                   : state_id;
         if(kTopTypePredicates.contains(predicate.name)) {
            continue;
         }
         for(const bool positive : {false, true}) {
            const auto polarity = positive ? size_t{1} : size_t{0};
            if(horizon.support_literals) {
               horizon_literal_relation_ids[predicate_index]
                                           [polarity] = optional_relation_id_at_construction(
                                              predicate_relation_key(predicate.name, positive)
                                           );
            }
            for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
               auto& ids = horizon_goal_relation_ids[predicate_index][level];
               for(const auto derivation : {
                      std::optional< GoalDerivation >{},
                      std::optional< GoalDerivation >{GoalDerivation::satisfied},
                      std::optional< GoalDerivation >{GoalDerivation::unsatisfied},
                      std::optional< GoalDerivation >{GoalDerivation::added_satisfied},
                      std::optional< GoalDerivation >{GoalDerivation::added_unsatisfied},
                   }) {
                  const auto slot = horizon_goal_derivation_index(derivation);
                  const auto key = predicate_relation_key(
                     predicate.name, positive, GoalLevel(level), derivation
                  );
                  const int root = optional_relation_id_at_construction(key);
                  ids.root[polarity][slot] = root;
                  ids.candidate[polarity][slot] = root;
                  if(split_candidates
                     and (derivation == GoalDerivation::satisfied or derivation == GoalDerivation::unsatisfied)) {
                     RelationKey anchored = key;
                     anchored.state_anchored = true;
                     ids.candidate[polarity][slot] = optional_relation_id_at_construction(anchored);
                  }
               }
            }
         }
      }

      horizon_action_relation_ids.assign(actions.size(), -1);
      if(not horizon.ignore_actions) {
         for(size_t action_index = 0; action_index < actions.size(); ++action_index) {
            horizon_action_relation_ids[action_index] = relation_id_at_construction(
               action_relation_key(actions[action_index].name)
            );
         }
      }
      horizon_parent_relation_id = horizon.enable_parent_relation
                                      ? relation_id_at_construction(
                                           opaque_relation_key(horizon.parent_relation)
                                        )
                                      : -1;
      horizon_sibling_relation_id = horizon.enable_sibling_relation
                                       ? relation_id_at_construction(
                                            opaque_relation_key(horizon.sibling_relation)
                                         )
                                       : -1;
      horizon_cousin_relation_id = horizon.enable_cousin_relation
                                      ? relation_id_at_construction(
                                           opaque_relation_key(horizon.cousin_relation)
                                        )
                                      : -1;
   }

   void prepare_horizon_builder(
      BatchBuilder& builder,
      const SemanticFlatHorizonEncoderConfig& horizon
   ) const
   {
      const std::vector< std::string > groups = {
         std::string(target_source_group_name(TargetSource::states))
      };
      set_flat_graph_attrs(
         builder,
         schema_.as_metadata(),
         FlatBuilderGraphConfig{
            .include_lgan_edges = horizon.include_lgan_edges,
            .use_predicate_virtual_nodes = horizon.use_predicate_virtual_nodes,
            .target_symbol_prefix = horizon.target_symbol_prefix,
            .target_entity_group_names = groups,
            .lgan_tn_edge_pos = horizon.lgan_tn_edge_pos,
            .lgan_nn_edge_pos = horizon.lgan_nn_edge_pos,
            .lgan_rr_edge_pos = horizon.lgan_rr_edge_pos,
            .pack_relation_args_relation_major = horizon.pack_relation_args_relation_major,
         }
      );
      register_flat_entity_fields(builder);
      register_flat_target_entity_fields(builder);
      builder.register_field(
         std::string(kTargetSizesField),
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
      );
      const TargetMetadataEmitConfig target_config{
         .position_node_type_id = std::string(kFlatEntityNodeType),
         .symbol_prefix = horizon.target_symbol_prefix,
         .include_depth = true,
         .include_group = true,
         .include_names = false,
         .groups = groups,
         .parent_relation = horizon.parent_relation,
      };
      register_target_fields(builder, target_config);
      builder.set_graph_attr(std::string(kTargetGroupsAttr), groups);
      builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), horizon.target_symbol_prefix);
      builder.set_graph_attr(std::string(kParentRelationAttr), horizon.parent_relation);
      register_flat_relation_instance_fields(builder, static_cast< int >(schema_.size()));
      if(horizon.include_lgan_edges) {
         register_flat_lgan_fields(builder);
      }
   }

   void prepare_builder(BatchBuilder& builder) const
   {
      set_flat_graph_attrs(
         builder,
         schema_.as_metadata(),
         FlatBuilderGraphConfig{
            .include_lgan_edges = config.include_lgan_edges,
            .use_predicate_virtual_nodes = config.use_predicate_virtual_nodes,
            .target_sources = source_names_for(config.target_sources),
            .lgan_anchor_sources = source_names_for(config.lgan_anchor_sources),
            .target_symbol_prefix = config.target_symbol_prefix,
            .target_entity_group_names = target_entity_group_names,
            .lgan_tn_edge_pos = config.lgan_tn_edge_pos,
            .lgan_nn_edge_pos = config.lgan_nn_edge_pos,
            .lgan_rr_edge_pos = config.lgan_rr_edge_pos,
            .pack_relation_args_relation_major = config.pack_relation_args_relation_major,
         }
      );
      register_flat_entity_fields(builder);
      register_flat_history_entity_fields(builder);
      register_flat_target_entity_fields(builder);
      if(not target_group_names.empty()) {
         builder.register_field(
            std::string(kTargetSizesField),
            GraphFieldSpec{
               .dtype = GraphFieldDType::I64,
               .mode = GraphFieldMode::STACK,
               .dim = 1,
            }
         );
         register_target_fields(
            builder,
            TargetMetadataEmitConfig{
               .position_node_type_id = std::string(kFlatEntityNodeType),
               .symbol_prefix = config.target_symbol_prefix,
               .include_depth = false,
               .include_group = true,
               .include_names = false,
               .groups = target_group_names,
               .parent_relation = std::nullopt,
            }
         );
         builder.set_graph_attr(std::string(kTargetGroupsAttr), target_group_names);
         builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config.target_symbol_prefix);
      }
      register_flat_relation_instance_fields(builder, static_cast< int >(schema_.size()));
      if(config.include_lgan_edges) {
         register_flat_lgan_fields(builder);
      }
   }

   void validate_atom(const SemanticAtom& atom, size_t object_count, std::string_view lane) const
   {
      if(atom.predicate < 0 or static_cast< size_t >(atom.predicate) >= predicates.size()) {
         throw std::invalid_argument(
            "Semantic flat " + std::string(lane) + " predicate out of range"
         );
      }
      const auto& predicate = predicates[static_cast< size_t >(atom.predicate)];
      if(atom.arguments.size() != static_cast< size_t >(predicate.arity)) {
         throw std::invalid_argument(
            "Semantic flat " + std::string(lane) + " arity does not match predicate schema"
         );
      }
      for(const int64_t object : atom.arguments) {
         if(object < 0 or static_cast< size_t >(object) >= object_count) {
            throw std::invalid_argument(
               "Semantic flat " + std::string(lane) + " object index out of range"
            );
         }
      }
   }

   void validate_input(const SemanticFlatRelationInput& input) const
   {
      const auto& objects = semantic_objects(input);
      const auto& goals = semantic_goals(input);
      const auto& static_facts = semantic_static_facts(input);
      std::set< std::string, std::less<> > object_names;
      for(const auto& object : objects) {
         validate_name(object, "object");
         if(not object_names.emplace(object).second) {
            throw std::invalid_argument("Semantic flat object names must be unique");
         }
      }
      for(const auto& fact : input.state_facts) {
         validate_atom(fact, objects.size(), "state fact");
      }
      for(const auto& fact : static_facts) {
         validate_atom(fact, objects.size(), "static fact");
      }
      for(const auto& goal : goals) {
         validate_atom(goal.atom, objects.size(), "goal");
      }
      if(input.subgoal_layers.size() > config.max_goal_level) {
         throw std::invalid_argument("Semantic flat subgoal layer count exceeds max_goal_level");
      }
      for(const auto& layer : input.subgoal_layers) {
         for(const auto& goal : layer) {
            validate_atom(goal.atom, objects.size(), "subgoal");
         }
      }
      for(const auto& action : input.actions) {
         if(action.action < 0 or static_cast< size_t >(action.action) >= actions.size()) {
            throw std::invalid_argument("Semantic flat action schema index out of range");
         }
         if(action.arguments.size()
            != static_cast< size_t >(actions[static_cast< size_t >(action.action)].arity)) {
            throw std::invalid_argument("Semantic flat ground action arity mismatch");
         }
         for(const int64_t object : action.arguments) {
            if(object < 0 or static_cast< size_t >(object) >= objects.size()) {
               throw std::invalid_argument("Semantic flat action object index out of range");
            }
         }
      }
      for(const auto& entry : input.history) {
         if(entry.dt >= 0) {
            throw std::invalid_argument("Semantic flat history requires negative dt values");
         }
         for(const auto& literal : entry.literals) {
            validate_atom(literal.atom, objects.size(), "history literal");
         }
      }
   }

   int64_t ensure_predicate_entity(SemanticEncodingContext& context, int64_t predicate) const
   {
      if(predicate < 0 or static_cast< size_t >(predicate) >= predicates.size()) {
         throw std::invalid_argument("semantic Flat predicate entity index out of range");
      }
      auto& existing = context.predicate_entity_indices[static_cast< size_t >(predicate)];
      if(existing >= 0) {
         return existing;
      }
      const int64_t index = context.entity_count++;
      existing = index;
      if(config.export_node_names) {
         context.entity_names.emplace_back(
            "predicate:" + predicates.at(static_cast< size_t >(predicate)).name
         );
      }
      context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::predicate_virtual));
      return index;
   }

   FlatTupleArguments tuple_args(
      SemanticEncodingContext& context,
      const SemanticAtom& atom,
      std::span< const int64_t > auxiliary = {}
   ) const
   {
      const std::optional< int64_t > predicate_entity = config.use_predicate_virtual_nodes
                                                           ? std::optional(ensure_predicate_entity(
                                                                context, atom.predicate
                                                             ))
                                                           : std::nullopt;
      return build_flat_tuple_args(std::span{atom.arguments}, auxiliary, predicate_entity);
   }

   void append_target_row(
      SemanticEncodingContext& context,
      TargetSource source,
      int64_t position,
      std::string display_name
   ) const
   {
      const int64_t index = static_cast< int64_t >(context.target_columns.size());
      context.target_columns.append(
         TargetRecord{
            .position = position,
            .index = index,
            .candidate_id = index,
            .depth = std::nullopt,
            .group_id = target_group_ids.at(source),
            .name = std::move(display_name),
         },
         false,
         true,
         config.export_node_names
      );
   }

   template < typename Key, typename Map >
   int64_t ensure_target_entity(
      SemanticEncodingContext& context,
      Map& indices,
      Key key,
      TargetSource source,
      FlatEntityRole role,
      const std::string& display_name,
      bool& inserted
   ) const
   {
      if(const auto it = indices.find(key); it != indices.end()) {
         inserted = false;
         return it->second;
      }
      inserted = true;
      const int64_t index = context.entity_count++;
      indices.emplace(std::move(key), index);
      if(config.export_node_names) {
         context.entity_names.push_back(display_name);
      }
      context.entity_role_ids.push_back(static_cast< int64_t >(role));
      context.target_entity_indices.push_back(index);
      context.target_entity_group_ids.push_back(target_entity_group_ids.at(source));
      return index;
   }

   SemanticEncodingContext make_context(
      const SemanticFlatRelationInput& input,
      const std::vector< SemanticLiteral >& grouped_goals,
      const std::vector< SemanticGoalLevel >& goal_levels
   ) const
   {
      SemanticEncodingContext context;
      const auto& objects = semantic_objects(input);
      context.entity_count = static_cast< int64_t >(objects.size());
      if(config.export_node_names) {
         context.entity_names = objects;
      }
      context.entity_role_ids.assign(
         objects.size(), static_cast< int64_t >(FlatEntityRole::object)
      );
      context.object_indices.resize(objects.size());
      std::iota(context.object_indices.begin(), context.object_indices.end(), int64_t{0});
      context.predicate_entity_indices.assign(predicates.size(), -1);

      for(const auto source : {TargetSource::goals, TargetSource::subgoals}) {
         if(not has_anchor_entity_source(config, source)) {
            continue;
         }
         for(const auto& literal : grouped_goals) {
            const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               continue;
            }
            const size_t level = semantic_goal_level(goal_levels, literal);
            if((source == TargetSource::goals and level > 0)
               or (source == TargetSource::subgoals and level == 0)) {
               continue;
            }
            const auto display = config.export_node_names
                                    ? goal_display_name(literal, level, predicates, objects)
                                    : std::string{};
            bool inserted = false;
            const auto key = GoalEntityKey{source, literal, level};
            const int64_t position = ensure_target_entity(
               context,
               context.goal_entity_indices,
               key,
               source,
               entity_role_for_target_source(source),
               display,
               inserted
            );
            if(has_target_source(config, source)) {
               append_target_row(context, source, position, display);
            }
         }
      }

      for(const auto& action : input.actions) {
         const auto display = config.export_node_names
                                 ? action_display_name(action, actions, objects)
                                 : std::string{};
         bool inserted = false;
         const int64_t position = ensure_target_entity(
            context,
            context.action_entity_indices,
            action,
            TargetSource::actions,
            FlatEntityRole::action,
            display,
            inserted
         );
         if(inserted) {
            context.unique_actions.push_back(action);
         }
         if(has_target_source(config, TargetSource::actions)) {
            append_target_row(context, TargetSource::actions, position, display);
         }
      }

      context.history_entries.reserve(input.history.size());
      for(const auto& entry : input.history) {
         if(input.history_max_steps.has_value() and std::abs(entry.dt) > *input.history_max_steps) {
            continue;
         }
         context.history_entries.push_back(
            PreparedHistoryEntry{
               .dt = entry.dt,
               .entry_index = context.history_entries.size(),
               .literals = entry.literals,
            }
         );
      }
      std::ranges::stable_sort(context.history_entries, {}, &PreparedHistoryEntry::dt);
      for(size_t idx = 0; idx < context.history_entries.size(); ++idx) {
         auto& entry = context.history_entries[idx];
         entry.entry_index = idx;
         entry.entity_index = context.entity_count++;
         if(config.export_node_names) {
            context.entity_names.push_back(
               "history:" + std::to_string(entry.dt) + "#" + std::to_string(idx)
            );
         }
         context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::history));
         context.history_entity_indices.push_back(entry.entity_index);
         context.history_entity_dt.push_back(entry.dt);
      }

      if(has_anchor_entity_source(config, TargetSource::history)) {
         for(const auto& entry : context.history_entries) {
            for(const auto& literal : entry.literals) {
               const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
               if(config.ignore_zero_arity_relations and predicate.arity == 0) {
                  continue;
               }
               const auto display = config.export_node_names
                                       ? "history:" + std::to_string(entry.dt) + "#"
                                            + std::to_string(entry.entry_index) + ":"
                                            + std::string(literal.positive ? "[+]" : "[-]")
                                            + atom_display_name(literal.atom, predicates, objects)
                                       : std::string{};
               const auto key = HistoryEntityKey{entry.dt, entry.entry_index, literal};
               bool inserted = false;
               const int64_t position = ensure_target_entity(
                  context,
                  context.history_target_entity_indices,
                  key,
                  TargetSource::history,
                  FlatEntityRole::history_target,
                  display,
                  inserted
               );
               if(has_target_source(config, TargetSource::history)) {
                  append_target_row(context, TargetSource::history, position, display);
               }
            }
         }
      }
      return context;
   }

   PreparedRelationGraph prepare_relation_graph(const SemanticFlatRelationInput& input) const
   {
      validate_input(input);
      PreparedRelationGraph prepared;
      prepared.input = &input;
      prepared.goal_levels = semantic_goal_levels(input);
      const auto& goals = semantic_goals(input);
      for(const auto category : kCategoryOrder) {
         const auto append_category = [&](const std::vector< SemanticLiteral >& literals) {
            for(const auto& literal : literals) {
               if(predicates.at(static_cast< size_t >(literal.atom.predicate)).category
                  == category) {
                  prepared.grouped_goals.push_back(literal);
               }
            }
         };
         append_category(goals);
         for(const auto& layer : input.subgoal_layers) {
            append_category(layer);
         }
      }
      prepared.context = make_context(input, prepared.grouped_goals, prepared.goal_levels);

      const auto& static_facts = semantic_static_facts(input);
      const auto
         record_facts =
            [&](const std::vector< SemanticAtom >& facts, bool emit_facts) {
               for(const auto& fact : facts) {
                  const auto& predicate = predicates.at(static_cast< size_t >(fact.predicate));
                  if(emit_facts
               and (predicate.category != SemanticPredicateCategory::static_predicate
                    or config.include_static)
               and not(config.ignore_zero_arity_relations and predicate.arity == 0)
               and config.use_predicate_virtual_nodes) {
                     (void) ensure_predicate_entity(prepared.context, fact.predicate);
                  }
                  prepared.fact_keys.emplace(fact);
               }
            };
      record_facts(static_facts, config.include_static);
      record_facts(input.state_facts, true);

      for(const auto& literal : prepared.grouped_goals) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(kTopTypePredicates.contains(predicate.name)
            or (config.ignore_zero_arity_relations and predicate.arity == 0)) {
            continue;
         }
         const bool satisfied = prepared.fact_keys.contains(literal.atom) == literal.positive;
         const bool emits = config.goal_derivations.contains(GoalDerivation::plain)
                            or config.goal_derivations.contains(
                               satisfied ? GoalDerivation::satisfied : GoalDerivation::unsatisfied
                            );
         if(emits and config.use_predicate_virtual_nodes) {
            (void) ensure_predicate_entity(prepared.context, literal.atom.predicate);
         }
      }
      if(config.use_predicate_virtual_nodes) {
         for(const auto& entry : prepared.context.history_entries) {
            for(const auto& literal : entry.literals) {
               const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
               if(not(config.ignore_zero_arity_relations and predicate.arity == 0)) {
                  (void) ensure_predicate_entity(prepared.context, literal.atom.predicate);
               }
            }
         }
      }
      return prepared;
   }

   void emit_relation_lane(
      const PreparedRelationGraph& prepared,
      RelationLane lane,
      FlatGraphContext& graph
   ) const
   {
      auto& context = prepared.context;
      const auto emit_atom =
         [&graph, &context, this](
            int relation_id, const SemanticAtom& atom, std::span< const int64_t > auxiliary = {}
         ) {
            const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               return;
            }
            const auto args = tuple_args(context, atom, auxiliary);
            graph.emit(required_relation_id(relation_id), args);
         };

      if(lane == RelationLane::facts) {
         const auto append = [&](const std::vector< SemanticAtom >& facts, bool emit_facts) {
            for(const auto& fact : facts) {
               const auto category = predicates.at(static_cast< size_t >(fact.predicate)).category;
               if(emit_facts
                  and (category != SemanticPredicateCategory::static_predicate or config.include_static)) {
                  emit_atom(state_relation_ids.at(static_cast< size_t >(fact.predicate)), fact);
               }
            }
         };
         append(semantic_static_facts(*prepared.input), config.include_static);
         append(prepared.input->state_facts, true);
         return;
      }

      if(lane == RelationLane::goals) {
         for(const auto& literal : prepared.grouped_goals) {
            const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
            if(kTopTypePredicates.contains(predicate.name)
               or (config.ignore_zero_arity_relations and predicate.arity == 0)) {
               continue;
            }
            const size_t level = semantic_goal_level(prepared.goal_levels, literal);
            if(config.goal_derivations.contains(GoalDerivation::plain)) {
               std::array< int64_t, 1 > auxiliary{};
               std::span< const int64_t > auxiliary_span;
               if(const auto source = source_for_goal_level(config, level); source.has_value()) {
                  auxiliary[0] = context.goal_entity_indices.at(
                     GoalEntityKey{*source, literal, level}
                  );
                  auxiliary_span = auxiliary;
               }
               emit_atom(
                  goal_relation_ids.at(static_cast< size_t >(literal.atom.predicate))
                     .at(level)
                     .by_polarity[literal.positive ? 1 : 0][goal_derivation_index(std::nullopt)],
                  literal.atom,
                  auxiliary_span
               );
            }
            const bool satisfied = prepared.fact_keys.contains(literal.atom) == literal.positive;
            const auto derivation = satisfied ? GoalDerivation::satisfied
                                              : GoalDerivation::unsatisfied;
            if(config.goal_derivations.contains(derivation)) {
               emit_atom(
                  goal_relation_ids.at(static_cast< size_t >(literal.atom.predicate))
                     .at(level)
                     .by_polarity[literal.positive ? 1 : 0][goal_derivation_index(derivation)],
                  literal.atom
               );
            }
         }
         return;
      }

      if(lane == RelationLane::actions) {
         for(const auto& action : context.unique_actions) {
            FlatTupleArguments args;
            args.reserve(action.arguments.size() + 1);
            args.push_back(context.action_entity_indices.at(action));
            args.insert(args.end(), action.arguments.begin(), action.arguments.end());
            graph.emit(
               required_relation_id(action_relation_ids.at(static_cast< size_t >(action.action))),
               args
            );
         }
         return;
      }

      for(const auto& entry : context.history_entries) {
         for(const auto& literal : entry.literals) {
            const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               continue;
            }
            std::array< int64_t, 2 > auxiliary{};
            std::span< const int64_t > auxiliary_span;
            if(has_anchor_entity_source(config, TargetSource::history)) {
               auxiliary[0] = context.history_target_entity_indices.at(
                  HistoryEntityKey{entry.dt, entry.entry_index, literal}
               );
               auxiliary[1] = entry.entity_index;
               auxiliary_span = auxiliary;
            } else {
               auxiliary[0] = entry.entity_index;
               auxiliary_span = std::span{auxiliary}.first(1);
            }
            emit_atom(
               history_relation_ids.at(
                  static_cast< size_t >(literal.atom.predicate)
               )[literal.positive ? 1 : 0],
               literal.atom,
               auxiliary_span
            );
         }
      }
   }

   void write_relation_fields(
      const PreparedRelationGraph& prepared,
      const std::vector< SemanticFlatFieldComponent::FieldDeclaration >& fields,
      const FlatGraphContext& graph,
      FlatFieldWriter& writer
   ) const
   {
      const auto& context = prepared.context;
      const int64_t node_size = context.entity_count;
      const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
      const int64_t history_size = static_cast< int64_t >(context.history_entity_indices.size());
      const int64_t target_entity_size = static_cast< int64_t >(
         context.target_entity_indices.size()
      );
      const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
      std::optional< FlatLGANFields > lgan;
      const auto values = [&](std::string_view key) -> std::span< const int64_t > {
         if(key == kNodeSizesField)
            return std::span{&node_size, size_t{1}};
         if(key == kObjectSizesField)
            return std::span{&object_size, size_t{1}};
         if(key == kObjectIndicesField)
            return context.object_indices;
         if(key == kEntityRoleIdsField)
            return context.entity_role_ids;
         if(key == kHistoryEntitySizesField)
            return std::span{&history_size, size_t{1}};
         if(key == kHistoryEntityIndicesField)
            return context.history_entity_indices;
         if(key == kHistoryEntityDtField)
            return context.history_entity_dt;
         if(key == kTargetEntitySizesField)
            return std::span{&target_entity_size, size_t{1}};
         if(key == kTargetEntityIndicesField)
            return context.target_entity_indices;
         if(key == kTargetEntityGroupIdsField)
            return context.target_entity_group_ids;
         if(key == kTargetSizesField)
            return std::span{&target_size, size_t{1}};
         if(key == kTargetPositionsField)
            return context.target_columns.positions;
         if(key == kTargetIndicesField)
            return context.target_columns.indices;
         if(key == kTargetCandidateIdsField)
            return context.target_columns.candidate_ids;
         if(key == kTargetGroupIdsField)
            return context.target_columns.group_ids;
         throw std::logic_error("unknown direct semantic flat field '" + std::string(key) + "'");
      };

      for(const auto& [key, spec] : fields) {
         if(not key.starts_with("lgan_")) {
            writer.set(key, values(key));
            continue;
         }
         if(not lgan.has_value()) {
            if(context.target_entity_indices.empty()) {
               throw std::invalid_argument(
                  "Semantic flat include_lgan_edges=true requires LGAN anchor entity rows"
               );
            }
            lgan = build_flat_lgan(graph.relations, context.target_entity_indices);
         }
         const int64_t tn_size = static_cast< int64_t >(lgan->tn_relation_indices.size());
         const int64_t nn_size = static_cast< int64_t >(lgan->nn_relation_indices.size());
         const int64_t rr_size = static_cast< int64_t >(lgan->rr_src_relation_indices.size());
         if(key == kLGANTNSizesField)
            writer.set(key, std::span{&tn_size, size_t{1}});
         else if(key == kLGANTNRelationIndicesField)
            writer.set(key, lgan->tn_relation_indices);
         else if(key == kLGANTNEntityIndicesField)
            writer.set(key, lgan->tn_entity_indices);
         else if(key == kLGANNNSizesField)
            writer.set(key, std::span{&nn_size, size_t{1}});
         else if(key == kLGANNNRelationIndicesField)
            writer.set(key, lgan->nn_relation_indices);
         else if(key == kLGANNNEntityIndicesField)
            writer.set(key, lgan->nn_entity_indices);
         else if(key == kLGANRRSizesField)
            writer.set(key, std::span{&rr_size, size_t{1}});
         else if(key == kLGANRRSrcRelationIndicesField) {
            writer.set(key, lgan->rr_src_relation_indices);
         } else if(key == kLGANRRDstRelationIndicesField) {
            writer.set(key, lgan->rr_dst_relation_indices);
         } else {
            throw std::logic_error("unknown direct semantic LGAN field '" + key + "'");
         }
      }
   }

   void
   write_relation_metadata(const PreparedRelationGraph& prepared, FlatMetadataWriter& writer) const
   {
      if(config.export_node_names) {
         writer.set_object_names(semantic_objects(*prepared.input));
      }
      if(not target_group_names.empty() and config.export_node_names) {
         if(prepared.context.target_columns.names.empty()) {
            if(not prepared.suppress_empty_target_names) {
               writer.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
            }
         } else {
            writer.add_lazy_target_names(prepared.context.target_columns.names);
         }
      }
   }

   void encode_into(
      const SemanticFlatRelationInput& input,
      BatchBuilder* builder,
      SemanticFlatCompositionInput* carrier,
      std::vector< std::string >& batch_target_names
   ) const
   {
      validate_input(input);
      const auto& objects = semantic_objects(input);
      const auto& goals = semantic_goals(input);
      const auto& static_facts = semantic_static_facts(input);

      const auto goal_levels = semantic_goal_levels(input);
      std::vector< SemanticLiteral > grouped_goals;
      for(const auto category : kCategoryOrder) {
         auto append_category = [&](const std::vector< SemanticLiteral >& literals) {
            for(const auto& literal : literals) {
               if(predicates.at(static_cast< size_t >(literal.atom.predicate)).category
                  == category) {
                  grouped_goals.push_back(literal);
               }
            }
         };
         append_category(goals);
         for(const auto& layer : input.subgoal_layers) {
            append_category(layer);
         }
      }

      auto context = make_context(input, grouped_goals, goal_levels);
      FlatRelationSink sink(schema_.size(), config.include_lgan_edges);
      auto emit =
         [&](int relation_id, const SemanticAtom& atom, std::span< const int64_t > auxiliary = {}) {
            const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               return;
            }
            const auto args = tuple_args(context, atom, auxiliary);
            sink.emit(required_relation_id(relation_id), args);
         };

      hash_set< SemanticAtom, SemanticAtomHash > fact_keys;
      const auto append_facts = [&](const std::vector< SemanticAtom >& facts, bool emit_facts) {
         for(const auto& fact : facts) {
            const auto category = predicates.at(static_cast< size_t >(fact.predicate)).category;
            if(emit_facts
               and (category != SemanticPredicateCategory::static_predicate or config.include_static)) {
               emit(state_relation_ids.at(static_cast< size_t >(fact.predicate)), fact);
            }
            fact_keys.emplace(fact);
         }
      };
      append_facts(static_facts, config.include_static);
      append_facts(input.state_facts, true);

      for(const auto& literal : grouped_goals) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(kTopTypePredicates.contains(predicate.name)
            or (config.ignore_zero_arity_relations and predicate.arity == 0)) {
            continue;
         }
         const size_t level = semantic_goal_level(goal_levels, literal);
         if(config.goal_derivations.contains(GoalDerivation::plain)) {
            std::array< int64_t, 1 > auxiliary{};
            std::span< const int64_t > auxiliary_span;
            if(const auto source = source_for_goal_level(config, level); source.has_value()) {
               auxiliary[0] = context.goal_entity_indices.at(
                  GoalEntityKey{*source, literal, level}
               );
               auxiliary_span = std::span{auxiliary};
            }
            emit(
               goal_relation_ids.at(static_cast< size_t >(literal.atom.predicate))
                  .at(level)
                  .by_polarity[literal.positive ? 1 : 0][goal_derivation_index(std::nullopt)],
               literal.atom,
               auxiliary_span
            );
         }
         const bool satisfied = fact_keys.contains(literal.atom) == literal.positive;
         const auto derivation = satisfied ? GoalDerivation::satisfied
                                           : GoalDerivation::unsatisfied;
         if(config.goal_derivations.contains(derivation)) {
            emit(
               goal_relation_ids.at(static_cast< size_t >(literal.atom.predicate))
                  .at(level)
                  .by_polarity[literal.positive ? 1 : 0][goal_derivation_index(derivation)],
               literal.atom
            );
         }
      }

      for(const auto& action : context.unique_actions) {
         FlatTupleArguments args;
         args.reserve(action.arguments.size() + 1);
         args.push_back(context.action_entity_indices.at(action));
         args.insert(args.end(), action.arguments.begin(), action.arguments.end());
         sink.emit(
            required_relation_id(action_relation_ids.at(static_cast< size_t >(action.action))), args
         );
      }

      for(const auto& entry : context.history_entries) {
         for(const auto& literal : entry.literals) {
            const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               continue;
            }
            std::array< int64_t, 2 > auxiliary{};
            std::span< const int64_t > auxiliary_span;
            if(has_anchor_entity_source(config, TargetSource::history)) {
               auxiliary[0] = context.history_target_entity_indices.at(
                  HistoryEntityKey{entry.dt, entry.entry_index, literal}
               );
               auxiliary[1] = entry.entity_index;
               auxiliary_span = std::span{auxiliary};
            } else {
               auxiliary[0] = entry.entity_index;
               auxiliary_span = std::span{auxiliary}.first(1);
            }
            emit(
               history_relation_ids.at(
                  static_cast< size_t >(literal.atom.predicate)
               )[literal.positive ? 1 : 0],
               literal.atom,
               auxiliary_span
            );
         }
      }

      if(builder != nullptr) {
         std::vector< float > zeros(static_cast< size_t >(context.entity_count), 0.0F);
         builder->add_node_features(std::string(kFlatEntityNodeType), "x", std::span{zeros}, 1);
         if(config.export_node_names) {
            builder->set_node_names(std::string(kFlatEntityNodeType), context.entity_names);
            builder->set_object_names(objects);
         }
      } else {
         carrier->composition.objects = context.entity_names;
         if(not config.export_node_names) {
            carrier->composition.objects.clear();
            carrier->composition.objects.reserve(static_cast< size_t >(context.entity_count));
            for(int64_t index = 0; index < context.entity_count; ++index) {
               carrier->composition.objects.push_back("entity:" + std::to_string(index));
            }
         } else {
            carrier->object_names = objects;
         }
      }

      const int64_t node_size = context.entity_count;
      const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
      const int64_t history_size = static_cast< int64_t >(context.history_entity_indices.size());
      const int64_t target_entity_size = static_cast< int64_t >(
         context.target_entity_indices.size()
      );
      auto set_field = [&](std::string_view key, std::span< const int64_t > values) {
         if(builder != nullptr) {
            builder->set_field(std::string(key), values);
         } else {
            set_semantic_carrier_field(*carrier, key, values);
         }
      };
      set_field(kNodeSizesField, std::span{&node_size, size_t{1}});
      set_field(kObjectSizesField, std::span{&object_size, size_t{1}});
      set_field(kObjectIndicesField, std::span{context.object_indices});
      set_field(kEntityRoleIdsField, std::span{context.entity_role_ids});
      set_field(kHistoryEntitySizesField, std::span{&history_size, size_t{1}});
      set_field(kHistoryEntityIndicesField, std::span{context.history_entity_indices});
      set_field(kHistoryEntityDtField, std::span{context.history_entity_dt});
      set_field(kTargetEntitySizesField, std::span{&target_entity_size, size_t{1}});
      set_field(kTargetEntityIndicesField, std::span{context.target_entity_indices});
      set_field(kTargetEntityGroupIdsField, std::span{context.target_entity_group_ids});

      if(not target_group_names.empty()) {
         const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
         set_field(kTargetSizesField, std::span{&target_size, size_t{1}});
         const TargetMetadataEmitConfig target_config{
            .position_node_type_id = std::string(kFlatEntityNodeType),
            .symbol_prefix = config.target_symbol_prefix,
            .include_depth = false,
            .include_group = true,
            .include_names = false,
            .groups = target_group_names,
            .parent_relation = std::nullopt,
         };
         if(builder != nullptr) {
            set_target_fields(*builder, context.target_columns, target_config);
            set_target_graph_attrs(*builder, context.target_columns, target_config);
         } else {
            set_field(kTargetPositionsField, std::span{context.target_columns.positions});
            set_field(kTargetIndicesField, std::span{context.target_columns.indices});
            set_field(kTargetCandidateIdsField, std::span{context.target_columns.candidate_ids});
            set_field(kTargetGroupIdsField, std::span{context.target_columns.group_ids});
            carrier->graph_attrs.emplace(
               std::string(kTargetSymbolPrefixAttr), config.target_symbol_prefix
            );
            carrier->graph_attrs.emplace(std::string(kTargetGroupsAttr), target_group_names);
         }
         if(config.export_node_names) {
            if(builder != nullptr) {
               batch_target_names.insert(
                  batch_target_names.end(),
                  context.target_columns.names.begin(),
                  context.target_columns.names.end()
               );
            } else {
               if(context.target_columns.names.empty()) {
                  carrier->graph_attrs.emplace(
                     std::string(kTargetNamesAttr), std::vector< std::string >{}
                  );
               } else {
                  carrier->lazy_target_name_strings = context.target_columns.names;
               }
            }
         }
      }

      if(builder != nullptr) {
         builder->set_field(std::string(kRelationCountsField), std::span{sink.relation_counts()});
         const int64_t relation_instance_size = sink.relation_instance_count();
         builder->set_field(
            std::string(kRelationInstanceSizesField), std::span{&relation_instance_size, size_t{1}}
         );
         builder->set_field(std::string(kRelationArgsField), std::span{sink.relation_args()});
      } else {
         append_semantic_carrier_relations(*carrier, schema_, sink);
      }

      if(config.include_lgan_edges) {
         if(context.target_entity_indices.empty()) {
            throw std::invalid_argument(
               "Semantic flat include_lgan_edges=true requires LGAN anchor entity rows"
            );
         }
         const auto lgan = build_flat_lgan(sink, std::span{context.target_entity_indices});
         const int64_t tn_size = static_cast< int64_t >(lgan.tn_relation_indices.size());
         const int64_t nn_size = static_cast< int64_t >(lgan.nn_relation_indices.size());
         const int64_t rr_size = static_cast< int64_t >(lgan.rr_src_relation_indices.size());
         set_field(kLGANTNSizesField, std::span{&tn_size, size_t{1}});
         set_field(kLGANTNRelationIndicesField, std::span{lgan.tn_relation_indices});
         set_field(kLGANTNEntityIndicesField, std::span{lgan.tn_entity_indices});
         set_field(kLGANNNSizesField, std::span{&nn_size, size_t{1}});
         set_field(kLGANNNRelationIndicesField, std::span{lgan.nn_relation_indices});
         set_field(kLGANNNEntityIndicesField, std::span{lgan.nn_entity_indices});
         set_field(kLGANRRSizesField, std::span{&rr_size, size_t{1}});
         set_field(kLGANRRSrcRelationIndicesField, std::span{lgan.rr_src_relation_indices});
         set_field(kLGANRRDstRelationIndicesField, std::span{lgan.rr_dst_relation_indices});
      }
   }

   PreparedHorizonGraph prepare_horizon_graph(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& horizon
   ) const
   {
      if(dag.predicates() != predicates or dag.actions() != actions) {
         throw std::invalid_argument(
            "Semantic flat Horizon DAG schema must exactly match the encoder schema"
         );
      }
      const auto& root = dag.root().state;
      const auto& root_objects = semantic_objects(root);
      if(root.subgoal_layers.size() > horizon.max_goal_level) {
         throw std::invalid_argument(
            "Semantic flat Horizon subgoal layer count exceeds max_goal_level"
         );
      }

      PreparedHorizonGraph prepared;
      prepared.dag = &dag;
      prepared.config = &horizon;
      auto& context = prepared.context;
      context.entity_count = static_cast< int64_t >(root_objects.size());
      if(horizon.export_node_names) {
         context.entity_names = root_objects;
      }
      context.entity_role_ids.assign(
         root_objects.size(), static_cast< int64_t >(FlatEntityRole::object)
      );
      context.object_indices.resize(root_objects.size());
      std::iota(context.object_indices.begin(), context.object_indices.end(), int64_t{0});
      context.predicate_entity_indices.assign(predicates.size(), -1);
      context.state_entity_indices.assign(dag.nodes().size(), -1);

      std::vector< TargetCandidateRow > target_rows;
      for(const auto& node : dag.nodes()) {
         const int64_t position = context.entity_count++;
         context.state_entity_indices.at(static_cast< size_t >(node.index)) = position;
         const bool public_root = root_in_public_carrier(horizon.root_policy) or node.index != 0;
         if(horizon.export_node_names) {
            context.entity_names.push_back(
               public_root ? horizon.target_symbol_prefix + std::to_string(node.index)
                           : "_root_state_"
            );
         }
         context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::state));
         if(not root_in_target_metadata(horizon.root_policy) and node.index == 0) {
            continue;
         }
         context.target_entity_indices.push_back(position);
         context.target_entity_group_ids.push_back(0);
         target_rows.push_back(
            TargetCandidateRow{
               .position = position,
               .index = node.index,
               .candidate_id = node.candidate_id,
               .depth = node.depth,
               .group_id = int64_t{0},
               .name = horizon.export_node_names
                          ? node.display_name.value_or("state:" + std::to_string(node.index))
                          : std::string{},
            }
         );
      }
      append_target_candidate_rows(
         context.target_columns,
         target_rows,
         TargetCandidateAppendConfig{
            .include_depth = true,
            .include_group = true,
            .include_names = horizon.export_node_names,
            .missing_candidate_id_prefix = "missing candidate_id for target node index ",
            .duplicate_candidate_id_prefix = "duplicate candidate_id ",
         }
      );

      prepared.goal_levels = semantic_goal_levels(root);
      for(const auto category : kCategoryOrder) {
         const auto append_category = [&](const std::vector< SemanticLiteral >& literals) {
            for(const auto& literal : literals) {
               if(predicates.at(static_cast< size_t >(literal.atom.predicate)).category
                  == category) {
                  prepared.goals.push_back(literal);
               }
            }
         };
         append_category(semantic_goals(root));
         for(const auto& layer : root.subgoal_layers) {
            append_category(layer);
         }
      }

      prepared.nodes.resize(dag.nodes().size());
      const auto collect_state =
         [&](
            const SemanticFlatRelationInput& state, bool include_static, PreparedHorizonNode& result
         ) {
            const auto append = [&](const std::vector< SemanticAtom >& atoms) {
               for(const auto& atom : atoms) {
                  const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
                  if((predicate.category == SemanticPredicateCategory::static_predicate
                      and not include_static)
                     or (horizon.ignore_zero_arity_relations and predicate.arity == 0)) {
                     continue;
                  }
                  result.fact_keys.emplace(atom);
               }
            };
            append(semantic_static_facts(state));
            append(state.state_facts);
         };
      collect_state(root, horizon.include_static, prepared.nodes.front());
      if(horizon.transition_mode == SemanticHorizonMode::full) {
         for(size_t index = 1; index < dag.nodes().size(); ++index) {
            collect_state(dag.nodes()[index].state, false, prepared.nodes[index]);
         }
      } else if(horizon.transition_mode == SemanticHorizonMode::delta) {
         std::set< SemanticAtom > root_fluent;
         std::set< SemanticAtom > root_derived;
         for(const auto& atom : root.state_facts) {
            const auto category = predicates.at(static_cast< size_t >(atom.predicate)).category;
            if(category == SemanticPredicateCategory::fluent)
               root_fluent.insert(atom);
            else if(category == SemanticPredicateCategory::derived)
               root_derived.insert(atom);
         }
         for(size_t index = 1; index < dag.nodes().size(); ++index) {
            const auto& node = dag.nodes()[index];
            auto& result = prepared.nodes[index];
            if(node.delta_literals.has_value()) {
               result.deltas = *node.delta_literals;
               for(const auto& literal : result.deltas) {
                  const auto category = predicates.at(static_cast< size_t >(literal.atom.predicate))
                                           .category;
                  auto* changed = category == SemanticPredicateCategory::fluent
                                     ? (literal.positive ? &result.added_fluent
                                                         : &result.removed_fluent)
                                  : category == SemanticPredicateCategory::derived
                                     ? (literal.positive ? &result.added_derived
                                                         : &result.removed_derived)
                                     : nullptr;
                  if(changed != nullptr)
                     changed->insert(literal.atom);
               }
               continue;
            }
            std::set< SemanticAtom > candidate_fluent;
            std::set< SemanticAtom > candidate_derived;
            for(const auto& atom : node.state.state_facts) {
               const auto category = predicates.at(static_cast< size_t >(atom.predicate)).category;
               if(category == SemanticPredicateCategory::fluent)
                  candidate_fluent.insert(atom);
               else if(category == SemanticPredicateCategory::derived) {
                  candidate_derived.insert(atom);
               }
            }
            for(const auto& atom : candidate_fluent) {
               if(not root_fluent.contains(atom)) {
                  result.added_fluent.insert(atom);
                  result.deltas.push_back({atom, true});
               }
            }
            for(const auto& atom : root_fluent) {
               if(not candidate_fluent.contains(atom)) {
                  result.removed_fluent.insert(atom);
                  result.deltas.push_back({atom, false});
               }
            }
            for(const auto& atom : candidate_derived) {
               if(not root_derived.contains(atom)) {
                  result.added_derived.insert(atom);
                  result.deltas.push_back({atom, true});
               }
            }
            for(const auto& atom : root_derived) {
               if(not candidate_derived.contains(atom)) {
                  result.removed_derived.insert(atom);
                  result.deltas.push_back({atom, false});
               }
            }
         }
      }

      if(horizon.use_predicate_virtual_nodes) {
         const auto plan_atom = [&](const SemanticAtom& atom) {
            const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
            if(not(horizon.ignore_zero_arity_relations and predicate.arity == 0)) {
               (void) ensure_predicate_entity(context, atom.predicate);
            }
         };
         const auto plan_state = [&](const SemanticFlatRelationInput& state, bool include_static) {
            const auto append = [&](const std::vector< SemanticAtom >& atoms) {
               for(const auto& atom : atoms) {
                  const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
                  if(predicate.category != SemanticPredicateCategory::static_predicate
                     or include_static) {
                     plan_atom(atom);
                  }
               }
            };
            append(semantic_static_facts(state));
            append(state.state_facts);
         };
         const auto plan_goal =
            [&](const SemanticLiteral& goal, const PreparedHorizonNode& node, bool plain) {
               const auto& predicate = predicates.at(static_cast< size_t >(goal.atom.predicate));
               if(kTopTypePredicates.contains(predicate.name)
                  or (horizon.ignore_zero_arity_relations and predicate.arity == 0))
                  return;
               const bool satisfied = node.fact_keys.contains(goal.atom) == goal.positive;
               if((plain and horizon.goal_derivations.contains(GoalDerivation::plain))
                  or horizon.goal_derivations.contains(
                     satisfied ? GoalDerivation::satisfied : GoalDerivation::unsatisfied
                  )) {
                  plan_atom(goal.atom);
               }
            };
         const bool root_anchor = root_in_state_relations(horizon.root_policy);
         (void) root_anchor;
         plan_state(root, horizon.include_static);
         for(const auto& goal : prepared.goals) {
            plan_goal(goal, prepared.nodes.front(), true);
         }
         for(size_t index = 1; index < dag.nodes().size(); ++index) {
            if(horizon.transition_mode == SemanticHorizonMode::full) {
               plan_state(dag.nodes()[index].state, false);
               for(const auto& goal : prepared.goals) {
                  plan_goal(goal, prepared.nodes[index], false);
               }
            } else if(horizon.transition_mode == SemanticHorizonMode::delta) {
               for(const auto& literal : prepared.nodes[index].deltas)
                  plan_atom(literal.atom);
               for(const auto& goal : prepared.goals) {
                  const auto& predicate = predicates.at(static_cast< size_t >(goal.atom.predicate));
                  if(kTopTypePredicates.contains(predicate.name)
                     or (horizon.ignore_zero_arity_relations and predicate.arity == 0))
                     continue;
                  const auto& node = prepared.nodes[index];
                  const auto category = predicate.category;
                  const auto derivation = category == SemanticPredicateCategory::fluent
                                             ? delta_goal_satisfaction_derivation(
                                                  goal.positive,
                                                  node.added_fluent.contains(goal.atom),
                                                  node.removed_fluent.contains(goal.atom)
                                               )
                                          : category == SemanticPredicateCategory::derived
                                             ? delta_goal_satisfaction_derivation(
                                                  goal.positive,
                                                  node.added_derived.contains(goal.atom),
                                                  node.removed_derived.contains(goal.atom)
                                               )
                                             : std::nullopt;
                  if(derivation.has_value() and horizon.goal_derivations.contains(*derivation))
                     plan_atom(goal.atom);
               }
            }
         }
      }
      return prepared;
   }

   void emit_horizon_semantics(const PreparedHorizonGraph& prepared, FlatGraphContext& graph) const
   {
      const auto& horizon = *prepared.config;
      const auto& dag = *prepared.dag;
      const auto& root = dag.root().state;
      auto& context = prepared.context;
      const auto state_position = [&](int64_t node_index) {
         if(node_index < 0
            or static_cast< size_t >(node_index) >= context.state_entity_indices.size()
            or context.state_entity_indices[static_cast< size_t >(node_index)] < 0) {
            throw std::invalid_argument(
               "Semantic flat Horizon encountered missing state carrier for node index "
               + std::to_string(node_index)
            );
         }
         return context.state_entity_indices[static_cast< size_t >(node_index)];
      };
      const auto emit_atom = [&](
                                int relation_id,
                                const SemanticAtom& atom,
                                std::optional< int64_t > state_anchor = std::nullopt
                             ) {
         const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
         if(horizon.ignore_zero_arity_relations and predicate.arity == 0)
            return;
         const std::array< int64_t, 1 > auxiliary = {state_anchor.value_or(0)};
         graph.emit(
            required_relation_id(relation_id),
            tuple_args(
               context,
               atom,
               state_anchor.has_value() ? std::span{auxiliary} : std::span< const int64_t >{}
            )
         );
      };
      const auto emit_state = [&](
                                 const SemanticFlatRelationInput& state,
                                 int64_t node_index,
                                 bool include_static,
                                 bool anchored
                              ) {
         const auto anchor = anchored ? std::optional(state_position(node_index)) : std::nullopt;
         const auto append = [&](const std::vector< SemanticAtom >& atoms) {
            for(const auto& atom : atoms) {
               const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
               if(predicate.category == SemanticPredicateCategory::static_predicate
                  and not include_static)
                  continue;
               const auto index = static_cast< size_t >(atom.predicate);
               emit_atom(
                  anchor.has_value() ? horizon_state_anchored_relation_ids.at(index)
                                     : horizon_state_relation_ids.at(index),
                  atom,
                  anchor
               );
            }
         };
         append(semantic_static_facts(state));
         append(state.state_facts);
      };
      const auto emit_goal = [&](
                                const SemanticLiteral& literal,
                                size_t level,
                                int64_t node_index,
                                bool anchored,
                                std::optional< GoalDerivation > derivation
                             ) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(kTopTypePredicates.contains(predicate.name)
            or (horizon.ignore_zero_arity_relations and predicate.arity == 0))
            return;
         const auto anchor = anchored ? std::optional(state_position(node_index)) : std::nullopt;
         const auto& ids = horizon_goal_relation_ids
                              .at(static_cast< size_t >(literal.atom.predicate))
                              .at(level);
         emit_atom(
            (anchor.has_value()
                ? ids.candidate
                : ids.root)[literal.positive ? 1 : 0][horizon_goal_derivation_index(derivation)],
            literal.atom,
            anchor
         );
      };

      const bool root_anchor = root_in_state_relations(horizon.root_policy);
      emit_state(root, 0, horizon.include_static, root_anchor);
      for(const auto& goal : prepared.goals) {
         const auto level = semantic_goal_level(prepared.goal_levels, goal);
         if(horizon.goal_derivations.contains(GoalDerivation::plain)) {
            emit_goal(goal, level, 0, root_anchor, std::nullopt);
         }
         const bool satisfied = prepared.nodes.front().fact_keys.contains(goal.atom)
                                == goal.positive;
         const auto derivation = satisfied ? GoalDerivation::satisfied
                                           : GoalDerivation::unsatisfied;
         if(horizon.goal_derivations.contains(derivation)) {
            emit_goal(goal, level, 0, root_anchor, derivation);
         }
      }
      const bool encode_actions = not horizon.ignore_actions
                                  or horizon.transition_mode == SemanticHorizonMode::action;
      for(size_t index = 1; index < dag.nodes().size(); ++index) {
         const auto& node = dag.nodes()[index];
         const auto emit_action = [&] {
            if(not encode_actions or not node.incoming_action.has_value())
               return;
            FlatTupleArguments args;
            args.push_back(state_position(node.index));
            args.insert(
               args.end(),
               node.incoming_action->arguments.begin(),
               node.incoming_action->arguments.end()
            );
            graph.emit(
               required_relation_id(horizon_action_relation_ids.at(
                  static_cast< size_t >(node.incoming_action->action)
               )),
               args
            );
         };
         if(horizon.transition_mode == SemanticHorizonMode::full) {
            emit_state(node.state, node.index, false, true);
            emit_action();
            for(const auto& goal : prepared.goals) {
               const bool satisfied = prepared.nodes[index].fact_keys.contains(goal.atom)
                                      == goal.positive;
               const auto derivation = satisfied ? GoalDerivation::satisfied
                                                 : GoalDerivation::unsatisfied;
               if(horizon.goal_derivations.contains(derivation)) {
                  emit_goal(
                     goal,
                     semantic_goal_level(prepared.goal_levels, goal),
                     node.index,
                     true,
                     derivation
                  );
               }
            }
            continue;
         }
         if(horizon.transition_mode == SemanticHorizonMode::action) {
            emit_action();
            continue;
         }
         for(const auto& literal : prepared.nodes[index].deltas) {
            emit_atom(
               horizon_literal_relation_ids.at(
                  static_cast< size_t >(literal.atom.predicate)
               )[literal.positive ? 1 : 0],
               literal.atom,
               state_position(node.index)
            );
         }
         emit_action();
         for(const auto& goal : prepared.goals) {
            const auto& node_plan = prepared.nodes[index];
            const auto category = predicates.at(static_cast< size_t >(goal.atom.predicate))
                                     .category;
            const auto derivation = category == SemanticPredicateCategory::fluent
                                       ? delta_goal_satisfaction_derivation(
                                            goal.positive,
                                            node_plan.added_fluent.contains(goal.atom),
                                            node_plan.removed_fluent.contains(goal.atom)
                                         )
                                    : category == SemanticPredicateCategory::derived
                                       ? delta_goal_satisfaction_derivation(
                                            goal.positive,
                                            node_plan.added_derived.contains(goal.atom),
                                            node_plan.removed_derived.contains(goal.atom)
                                         )
                                       : std::nullopt;
            if(derivation.has_value() and horizon.goal_derivations.contains(*derivation)) {
               emit_goal(
                  goal,
                  semantic_goal_level(prepared.goal_levels, goal),
                  node.index,
                  true,
                  derivation
               );
            }
         }
      }
   }

   void emit_horizon_topology(const PreparedHorizonGraph& prepared, FlatGraphContext& graph) const
   {
      const auto& horizon = *prepared.config;
      const auto& dag = *prepared.dag;
      const auto position = [&](int64_t index) {
         return prepared.context.state_entity_indices.at(static_cast< size_t >(index));
      };
      const bool exclude_root = horizon.root_policy == RootPolicy::exclude;
      if(horizon.enable_parent_relation) {
         for(const auto& [parent, child] : dag.edges()) {
            if(exclude_root and parent == 0)
               continue;
            const std::array< int64_t, 2 > args = {position(parent), position(child)};
            graph.emit(required_relation_id(horizon_parent_relation_id), args);
         }
      }
      std::vector< std::vector< int64_t > > children(dag.nodes().size());
      for(const auto& [parent, child] : dag.edges()) {
         if(not exclude_root or parent != 0) {
            children.at(static_cast< size_t >(parent)).push_back(child);
         }
      }
      const auto emit_pair = [&](int relation, int64_t source, int64_t target) {
         const std::array< int64_t, 2 > args = {position(source), position(target)};
         graph.emit(required_relation_id(relation), args);
      };
      struct PairHash {
         size_t operator()(const std::pair< int64_t, int64_t >& value) const noexcept
         {
            return std::hash< int64_t >{}(value.first)
                   ^ (std::hash< int64_t >{}(value.second) << 1);
         }
      };
      hash_set< std::pair< int64_t, int64_t >, PairHash > siblings;
      if(horizon.enable_sibling_relation) {
         for(auto& values : children) {
            std::ranges::sort(values);
            for(size_t lhs = 0; lhs < values.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < values.size(); ++rhs) {
                  if(siblings.emplace(values[lhs], values[rhs]).second) {
                     emit_pair(horizon_sibling_relation_id, values[lhs], values[rhs]);
                     emit_pair(horizon_sibling_relation_id, values[rhs], values[lhs]);
                  }
               }
            }
         }
      }
      if(horizon.enable_cousin_relation) {
         hash_set< std::pair< int64_t, int64_t >, PairHash > cousins;
         for(const auto& parents : children) {
            for(size_t lhs = 0; lhs < parents.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < parents.size(); ++rhs) {
                  const auto& left = children.at(static_cast< size_t >(parents[lhs]));
                  const auto& right = children.at(static_cast< size_t >(parents[rhs]));
                  for(const auto u : left) {
                     for(const auto v : right) {
                        if(u == v)
                           continue;
                        const auto pair = std::minmax(u, v);
                        if(siblings.contains(pair) or not cousins.emplace(pair).second)
                           continue;
                        emit_pair(horizon_cousin_relation_id, u, v);
                        emit_pair(horizon_cousin_relation_id, v, u);
                     }
                  }
               }
            }
         }
      }
   }

   void write_horizon_fields(
      const PreparedHorizonGraph& prepared,
      const std::vector< SemanticFlatFieldComponent::FieldDeclaration >& fields,
      const FlatGraphContext& graph,
      FlatFieldWriter& writer
   ) const
   {
      const auto& context = prepared.context;
      const int64_t node_size = context.entity_count;
      const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
      const int64_t target_entity_size = static_cast< int64_t >(
         context.target_entity_indices.size()
      );
      const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
      std::optional< FlatLGANFields > lgan;
      for(const auto& [key, spec] : fields) {
         (void) spec;
         if(key == kNodeSizesField)
            writer.set(key, std::span{&node_size, size_t{1}});
         else if(key == kObjectSizesField) {
            writer.set(key, std::span{&object_size, size_t{1}});
         } else if(key == kObjectIndicesField)
            writer.set(key, context.object_indices);
         else if(key == kEntityRoleIdsField)
            writer.set(key, context.entity_role_ids);
         else if(key == kTargetEntitySizesField) {
            writer.set(key, std::span{&target_entity_size, size_t{1}});
         } else if(key == kTargetEntityIndicesField) {
            writer.set(key, context.target_entity_indices);
         } else if(key == kTargetEntityGroupIdsField) {
            writer.set(key, context.target_entity_group_ids);
         } else if(key == kTargetSizesField) {
            writer.set(key, std::span{&target_size, size_t{1}});
         } else if(key == kTargetPositionsField)
            writer.set(key, context.target_columns.positions);
         else if(key == kTargetIndicesField)
            writer.set(key, context.target_columns.indices);
         else if(key == kTargetCandidateIdsField) {
            writer.set(key, context.target_columns.candidate_ids);
         } else if(key == kTargetDepthsField)
            writer.set(key, context.target_columns.depths);
         else if(key == kTargetGroupIdsField)
            writer.set(key, context.target_columns.group_ids);
         else {
            if(not lgan.has_value()) {
               if(context.target_columns.positions.empty()) {
                  throw std::invalid_argument(
                     "FlatHorizonEncoder include_lgan_edges=true requires surviving candidate "
                     "state rows, but none were encoded. Ensure the horizon DAG exposes at "
                     "least one selectable candidate state."
                  );
               }
               lgan = build_flat_lgan(graph.relations, context.target_columns.positions);
            }
            const int64_t tn_size = static_cast< int64_t >(lgan->tn_relation_indices.size());
            const int64_t nn_size = static_cast< int64_t >(lgan->nn_relation_indices.size());
            const int64_t rr_size = static_cast< int64_t >(lgan->rr_src_relation_indices.size());
            if(key == kLGANTNSizesField)
               writer.set(key, std::span{&tn_size, size_t{1}});
            else if(key == kLGANTNRelationIndicesField) {
               writer.set(key, lgan->tn_relation_indices);
            } else if(key == kLGANTNEntityIndicesField)
               writer.set(key, lgan->tn_entity_indices);
            else if(key == kLGANNNSizesField)
               writer.set(key, std::span{&nn_size, size_t{1}});
            else if(key == kLGANNNRelationIndicesField) {
               writer.set(key, lgan->nn_relation_indices);
            } else if(key == kLGANNNEntityIndicesField)
               writer.set(key, lgan->nn_entity_indices);
            else if(key == kLGANRRSizesField)
               writer.set(key, std::span{&rr_size, size_t{1}});
            else if(key == kLGANRRSrcRelationIndicesField) {
               writer.set(key, lgan->rr_src_relation_indices);
            } else if(key == kLGANRRDstRelationIndicesField) {
               writer.set(key, lgan->rr_dst_relation_indices);
            } else
               throw std::logic_error("unknown direct semantic Horizon field '" + key + "'");
         }
      }
   }

   void
   write_horizon_metadata(const PreparedHorizonGraph& prepared, FlatMetadataWriter& writer) const
   {
      if(not prepared.config->export_node_names)
         return;
      writer.set_object_names(semantic_objects(prepared.dag->root().state));
      if(prepared.context.target_columns.names.empty()) {
         if(not prepared.suppress_empty_target_names) {
            writer.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         }
      } else {
         writer.add_lazy_target_names(prepared.context.target_columns.names);
      }
   }

   void encode_horizon(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& horizon,
      BatchBuilder* builder,
      SemanticFlatCompositionInput* carrier
   ) const
   {
      if(dag.predicates() != predicates or dag.actions() != actions) {
         throw std::invalid_argument(
            "Semantic flat Horizon DAG schema must exactly match the encoder schema"
         );
      }
      const auto& root = dag.root().state;
      const auto& root_objects = semantic_objects(root);
      const auto& root_goals = semantic_goals(root);
      if(root.subgoal_layers.size() > horizon.max_goal_level) {
         throw std::invalid_argument(
            "Semantic flat Horizon subgoal layer count exceeds max_goal_level"
         );
      }

      SemanticEncodingContext context;
      context.entity_count = static_cast< int64_t >(root_objects.size());
      if(horizon.export_node_names) {
         context.entity_names = root_objects;
      }
      context.entity_role_ids.assign(
         root_objects.size(), static_cast< int64_t >(FlatEntityRole::object)
      );
      context.object_indices.resize(root_objects.size());
      std::iota(context.object_indices.begin(), context.object_indices.end(), int64_t{0});
      context.predicate_entity_indices.assign(predicates.size(), -1);
      context.state_entity_indices.assign(dag.nodes().size(), -1);

      std::vector< TargetCandidateRow > target_rows;
      for(const auto& node : dag.nodes()) {
         const int64_t position = context.entity_count++;
         context.state_entity_indices.at(static_cast< size_t >(node.index)) = position;
         const bool public_root = root_in_public_carrier(horizon.root_policy) or node.index != 0;
         if(horizon.export_node_names) {
            context.entity_names.push_back(
               public_root ? horizon.target_symbol_prefix + std::to_string(node.index)
                           : "_root_state_"
            );
         }
         context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::state));
         if(not root_in_target_metadata(horizon.root_policy) and node.index == 0) {
            continue;
         }
         context.target_entity_indices.push_back(position);
         context.target_entity_group_ids.push_back(0);
         target_rows.push_back(
            TargetCandidateRow{
               .position = position,
               .index = node.index,
               .candidate_id = node.candidate_id,
               .depth = node.depth,
               .group_id = int64_t{0},
               .name = horizon.export_node_names
                          ? node.display_name.value_or("state:" + std::to_string(node.index))
                          : std::string{},
            }
         );
      }
      append_target_candidate_rows(
         context.target_columns,
         target_rows,
         TargetCandidateAppendConfig{
            .include_depth = true,
            .include_group = true,
            .include_names = horizon.export_node_names,
            .missing_candidate_id_prefix = "missing candidate_id for target node index ",
            .duplicate_candidate_id_prefix = "duplicate candidate_id ",
         }
      );

      const auto goal_levels = semantic_goal_levels(root);
      std::vector< SemanticLiteral > goals;
      for(const auto category : kCategoryOrder) {
         auto append_category = [&](const std::vector< SemanticLiteral >& literals) {
            for(const auto& literal : literals) {
               if(predicates.at(static_cast< size_t >(literal.atom.predicate)).category
                  == category) {
                  goals.push_back(literal);
               }
            }
         };
         append_category(root_goals);
         for(const auto& layer : root.subgoal_layers) {
            append_category(layer);
         }
      }

      FlatRelationSink sink(schema_.size(), horizon.include_lgan_edges);
      auto state_position = [&](int64_t node_index) {
         if(node_index < 0
            or static_cast< size_t >(node_index) >= context.state_entity_indices.size()
            or context.state_entity_indices[static_cast< size_t >(node_index)] < 0) {
            throw std::invalid_argument(
               "Semantic flat Horizon encountered missing state carrier for node index "
               + std::to_string(node_index)
            );
         }
         return context.state_entity_indices[static_cast< size_t >(node_index)];
      };
      auto emit_atom = [&](
                          int relation_id,
                          const SemanticAtom& atom,
                          std::optional< int64_t > state_anchor = std::nullopt
                       ) {
         const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
         if(horizon.ignore_zero_arity_relations and predicate.arity == 0) {
            return;
         }
         const std::array< int64_t, 1 > auxiliary = {state_anchor.value_or(0)};
         sink.emit(
            required_relation_id(relation_id),
            tuple_args(
               context,
               atom,
               state_anchor.has_value() ? std::span{auxiliary} : std::span< const int64_t >{}
            )
         );
      };
      auto emit_state = [&](
                           const SemanticFlatRelationInput& input,
                           int64_t node_index,
                           bool include_static,
                           bool include_anchor
                        ) {
         hash_set< SemanticAtom, SemanticAtomHash > facts;
         const auto anchor = include_anchor ? std::optional(state_position(node_index))
                                            : std::nullopt;
         const auto append_facts = [&](const std::vector< SemanticAtom >& atoms) {
            for(const auto& atom : atoms) {
               const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
               if(predicate.category == SemanticPredicateCategory::static_predicate
                  and not include_static) {
                  continue;
               }
               if(horizon.ignore_zero_arity_relations and predicate.arity == 0) {
                  continue;
               }
               const auto predicate_index = static_cast< size_t >(atom.predicate);
               emit_atom(
                  anchor.has_value() ? horizon_state_anchored_relation_ids.at(predicate_index)
                                     : horizon_state_relation_ids.at(predicate_index),
                  atom,
                  anchor
               );
               facts.insert(atom);
            }
         };
         append_facts(semantic_static_facts(input));
         append_facts(input.state_facts);
         return facts;
      };
      auto emit_goal = [&](
                          const SemanticLiteral& literal,
                          size_t level,
                          int64_t node_index,
                          bool include_anchor,
                          std::optional< GoalDerivation > derivation
                       ) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(kTopTypePredicates.contains(predicate.name)
            or (horizon.ignore_zero_arity_relations and predicate.arity == 0)) {
            return;
         }
         const auto anchor = include_anchor ? std::optional(state_position(node_index))
                                            : std::nullopt;
         const auto& ids = horizon_goal_relation_ids
                              .at(static_cast< size_t >(literal.atom.predicate))
                              .at(level);
         const auto relation_id = (anchor.has_value() ? ids.candidate : ids.root)
            [literal.positive ? 1 : 0][horizon_goal_derivation_index(derivation)];
         emit_atom(relation_id, literal.atom, anchor);
      };

      const bool root_anchor = root_in_state_relations(horizon.root_policy);
      const auto root_facts = emit_state(root, 0, horizon.include_static, root_anchor);
      for(const auto& literal : goals) {
         const size_t level = semantic_goal_level(goal_levels, literal);
         if(horizon.goal_derivations.contains(GoalDerivation::plain)) {
            emit_goal(literal, level, 0, root_anchor, std::nullopt);
         }
         const bool satisfied = root_facts.contains(literal.atom) == literal.positive;
         const auto derivation = satisfied ? GoalDerivation::satisfied
                                           : GoalDerivation::unsatisfied;
         if(horizon.goal_derivations.contains(derivation)) {
            emit_goal(literal, level, 0, root_anchor, derivation);
         }
      }

      auto emit_action = [&](const SemanticGroundAction& action, int64_t node_index) {
         FlatTupleArguments args;
         args.reserve(action.arguments.size() + 1);
         args.push_back(state_position(node_index));
         args.insert(args.end(), action.arguments.begin(), action.arguments.end());
         sink.emit(
            required_relation_id(
               horizon_action_relation_ids.at(static_cast< size_t >(action.action))
            ),
            args
         );
      };
      auto emit_delta = [&](const SemanticLiteral& literal, int64_t node_index) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(horizon.ignore_zero_arity_relations and predicate.arity == 0) {
            return;
         }
         emit_atom(
            horizon_literal_relation_ids.at(
               static_cast< size_t >(literal.atom.predicate)
            )[literal.positive ? 1 : 0],
            literal.atom,
            state_position(node_index)
         );
      };
      auto emit_delta_satisfaction = [&](
                                        const SemanticLiteral& goal,
                                        size_t level,
                                        const std::set< SemanticAtom >& added,
                                        const std::set< SemanticAtom >& removed,
                                        int64_t node_index
                                     ) {
         const auto derivation = delta_goal_satisfaction_derivation(
            goal.positive, added.contains(goal.atom), removed.contains(goal.atom)
         );
         if(not derivation.has_value() or not horizon.goal_derivations.contains(*derivation)) {
            return;
         }
         emit_goal(goal, level, node_index, true, derivation);
      };

      std::set< SemanticAtom > root_fluent;
      std::set< SemanticAtom > root_derived;
      if(horizon.transition_mode == SemanticHorizonMode::delta) {
         for(const auto& atom : root.state_facts) {
            const auto category = predicates.at(static_cast< size_t >(atom.predicate)).category;
            if(category == SemanticPredicateCategory::fluent) {
               root_fluent.insert(atom);
            } else if(category == SemanticPredicateCategory::derived) {
               root_derived.insert(atom);
            }
         }
      }

      const bool encode_actions = not horizon.ignore_actions
                                  or horizon.transition_mode == SemanticHorizonMode::action;
      for(size_t index = 1; index < dag.nodes().size(); ++index) {
         const auto& node = dag.nodes()[index];
         if(horizon.transition_mode == SemanticHorizonMode::full) {
            const auto candidate_facts = emit_state(node.state, node.index, false, true);
            if(encode_actions and node.incoming_action.has_value()) {
               emit_action(*node.incoming_action, node.index);
            }
            for(const auto& goal : goals) {
               const bool satisfied = candidate_facts.contains(goal.atom) == goal.positive;
               const auto derivation = satisfied ? GoalDerivation::satisfied
                                                 : GoalDerivation::unsatisfied;
               if(horizon.goal_derivations.contains(derivation)) {
                  emit_goal(
                     goal, semantic_goal_level(goal_levels, goal), node.index, true, derivation
                  );
               }
            }
            continue;
         }
         if(horizon.transition_mode == SemanticHorizonMode::action) {
            if(encode_actions and node.incoming_action.has_value()) {
               emit_action(*node.incoming_action, node.index);
            }
            continue;
         }

         std::set< SemanticAtom > added_fluent;
         std::set< SemanticAtom > removed_fluent;
         std::set< SemanticAtom > added_derived;
         std::set< SemanticAtom > removed_derived;
         if(node.delta_literals.has_value()) {
            for(const auto& literal : *node.delta_literals) {
               emit_delta(literal, node.index);
               const auto category = predicates.at(static_cast< size_t >(literal.atom.predicate))
                                        .category;
               auto* changed = category == SemanticPredicateCategory::fluent
                                  ? (literal.positive ? &added_fluent : &removed_fluent)
                               : category == SemanticPredicateCategory::derived
                                  ? (literal.positive ? &added_derived : &removed_derived)
                                  : nullptr;
               if(changed != nullptr) {
                  changed->insert(literal.atom);
               }
            }
         } else {
            std::set< SemanticAtom > candidate_fluent;
            std::set< SemanticAtom > candidate_derived;
            for(const auto& atom : node.state.state_facts) {
               const auto category = predicates.at(static_cast< size_t >(atom.predicate)).category;
               if(category == SemanticPredicateCategory::fluent) {
                  candidate_fluent.insert(atom);
               } else if(category == SemanticPredicateCategory::derived) {
                  candidate_derived.insert(atom);
               }
            }
            for(const auto& atom : candidate_fluent) {
               if(not root_fluent.contains(atom)) {
                  added_fluent.insert(atom);
                  emit_delta(SemanticLiteral{atom, true}, node.index);
               }
            }
            for(const auto& atom : root_fluent) {
               if(not candidate_fluent.contains(atom)) {
                  removed_fluent.insert(atom);
                  emit_delta(SemanticLiteral{atom, false}, node.index);
               }
            }
            for(const auto& atom : candidate_derived) {
               if(not root_derived.contains(atom)) {
                  added_derived.insert(atom);
                  emit_delta(SemanticLiteral{atom, true}, node.index);
               }
            }
            for(const auto& atom : root_derived) {
               if(not candidate_derived.contains(atom)) {
                  removed_derived.insert(atom);
                  emit_delta(SemanticLiteral{atom, false}, node.index);
               }
            }
         }
         if(encode_actions and node.incoming_action.has_value()) {
            emit_action(*node.incoming_action, node.index);
         }
         for(const auto& goal : goals) {
            const auto category = predicates.at(static_cast< size_t >(goal.atom.predicate))
                                     .category;
            if(category == SemanticPredicateCategory::fluent) {
               emit_delta_satisfaction(
                  goal,
                  semantic_goal_level(goal_levels, goal),
                  added_fluent,
                  removed_fluent,
                  node.index
               );
            } else if(category == SemanticPredicateCategory::derived) {
               emit_delta_satisfaction(
                  goal,
                  semantic_goal_level(goal_levels, goal),
                  added_derived,
                  removed_derived,
                  node.index
               );
            }
         }
      }

      const bool exclude_root_topology = horizon.root_policy == RootPolicy::exclude;
      if(horizon.enable_parent_relation) {
         for(const auto& [parent, child] : dag.edges()) {
            if(exclude_root_topology and parent == 0) {
               continue;
            }
            const std::array< int64_t, 2 > args = {state_position(parent), state_position(child)};
            sink.emit(required_relation_id(horizon_parent_relation_id), args);
         }
      }
      std::vector< std::vector< int64_t > > children(dag.nodes().size());
      for(const auto& [parent, child] : dag.edges()) {
         if(not exclude_root_topology or parent != 0) {
            children[static_cast< size_t >(parent)].push_back(child);
         }
      }
      auto emit_pair = [&](int relation_id, int64_t source, int64_t target) {
         const std::array< int64_t, 2 > args = {state_position(source), state_position(target)};
         sink.emit(required_relation_id(relation_id), args);
      };
      struct PairHash {
         size_t operator()(const std::pair< int64_t, int64_t >& value) const noexcept
         {
            return std::hash< int64_t >{}(value.first)
                   ^ (std::hash< int64_t >{}(value.second) << 1);
         }
      };
      hash_set< std::pair< int64_t, int64_t >, PairHash > siblings;
      if(horizon.enable_sibling_relation) {
         for(auto& values : children) {
            std::ranges::sort(values);
            for(size_t lhs = 0; lhs < values.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < values.size(); ++rhs) {
                  if(siblings.emplace(values[lhs], values[rhs]).second) {
                     emit_pair(horizon_sibling_relation_id, values[lhs], values[rhs]);
                     emit_pair(horizon_sibling_relation_id, values[rhs], values[lhs]);
                  }
               }
            }
         }
      }
      if(horizon.enable_cousin_relation) {
         hash_set< std::pair< int64_t, int64_t >, PairHash > cousins;
         for(const auto& parents : children) {
            for(size_t lhs = 0; lhs < parents.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < parents.size(); ++rhs) {
                  const auto& left = children.at(static_cast< size_t >(parents[lhs]));
                  const auto& right = children.at(static_cast< size_t >(parents[rhs]));
                  for(const auto u : left) {
                     for(const auto v : right) {
                        if(u == v) {
                           continue;
                        }
                        const auto pair = std::minmax(u, v);
                        if(siblings.contains(pair) or not cousins.emplace(pair).second) {
                           continue;
                        }
                        emit_pair(horizon_cousin_relation_id, u, v);
                        emit_pair(horizon_cousin_relation_id, v, u);
                     }
                  }
               }
            }
         }
      }

      if(builder != nullptr) {
         std::vector< float > zeros(static_cast< size_t >(context.entity_count), 0.0F);
         builder->add_node_features(std::string(kFlatEntityNodeType), "x", std::span{zeros}, 1);
         if(horizon.export_node_names) {
            builder->set_node_names(std::string(kFlatEntityNodeType), context.entity_names);
            builder->set_object_names(root_objects);
         }
      } else {
         carrier->composition.objects = context.entity_names;
         if(not horizon.export_node_names) {
            carrier->composition.objects.clear();
            carrier->composition.objects.reserve(static_cast< size_t >(context.entity_count));
            for(int64_t index = 0; index < context.entity_count; ++index) {
               carrier->composition.objects.push_back("entity:" + std::to_string(index));
            }
         } else {
            carrier->object_names = root_objects;
         }
      }
      const int64_t node_size = context.entity_count;
      const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
      const int64_t target_entity_size = static_cast< int64_t >(
         context.target_entity_indices.size()
      );
      const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
      auto set_field = [&](std::string_view key, std::span< const int64_t > values) {
         if(builder != nullptr) {
            builder->set_field(std::string(key), values);
         } else {
            set_semantic_carrier_field(*carrier, key, values);
         }
      };
      set_field(kNodeSizesField, std::span{&node_size, size_t{1}});
      set_field(kObjectSizesField, std::span{&object_size, size_t{1}});
      set_field(kObjectIndicesField, std::span{context.object_indices});
      set_field(kEntityRoleIdsField, std::span{context.entity_role_ids});
      set_field(kTargetEntitySizesField, std::span{&target_entity_size, size_t{1}});
      set_field(kTargetEntityIndicesField, std::span{context.target_entity_indices});
      set_field(kTargetEntityGroupIdsField, std::span{context.target_entity_group_ids});
      set_field(kTargetSizesField, std::span{&target_size, size_t{1}});
      const std::vector< std::string > groups = {
         std::string(target_source_group_name(TargetSource::states))
      };
      const TargetMetadataEmitConfig target_config{
         .position_node_type_id = std::string(kFlatEntityNodeType),
         .symbol_prefix = horizon.target_symbol_prefix,
         .include_depth = true,
         .include_group = true,
         .include_names = false,
         .groups = groups,
         .parent_relation = horizon.parent_relation,
      };
      if(builder != nullptr) {
         set_target_fields(*builder, context.target_columns, target_config);
         set_target_graph_attrs(*builder, context.target_columns, target_config);
      } else {
         set_field(kTargetPositionsField, std::span{context.target_columns.positions});
         set_field(kTargetIndicesField, std::span{context.target_columns.indices});
         set_field(kTargetCandidateIdsField, std::span{context.target_columns.candidate_ids});
         set_field(kTargetDepthsField, std::span{context.target_columns.depths});
         set_field(kTargetGroupIdsField, std::span{context.target_columns.group_ids});
         carrier->graph_attrs.emplace(std::string(kTargetGroupsAttr), groups);
         carrier->graph_attrs.emplace(std::string(kParentRelationAttr), horizon.parent_relation);
      }
      if(horizon.export_node_names) {
         if(context.target_columns.names.empty()) {
            if(builder != nullptr) {
               builder->set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
            } else {
               carrier->graph_attrs.emplace(
                  std::string(kTargetNamesAttr), std::vector< std::string >{}
               );
            }
         } else {
            if(builder != nullptr) {
               builder->add_lazy_target_names(std::span{context.target_columns.names});
            } else {
               carrier->lazy_target_name_strings = context.target_columns.names;
            }
         }
      }
      if(builder != nullptr) {
         builder->set_field(std::string(kRelationCountsField), std::span{sink.relation_counts()});
         const int64_t relation_size = sink.relation_instance_count();
         builder->set_field(
            std::string(kRelationInstanceSizesField), std::span{&relation_size, size_t{1}}
         );
         builder->set_field(std::string(kRelationArgsField), std::span{sink.relation_args()});
      } else {
         append_semantic_carrier_relations(*carrier, schema_, sink);
      }

      if(horizon.include_lgan_edges) {
         if(context.target_columns.positions.empty()) {
            throw std::invalid_argument(
               "FlatHorizonEncoder include_lgan_edges=true requires surviving candidate state "
               "rows, but none were encoded. Ensure the horizon DAG exposes at least one "
               "selectable candidate state."
            );
         }
         const auto lgan = build_flat_lgan(sink, std::span{context.target_columns.positions});
         const int64_t tn_size = static_cast< int64_t >(lgan.tn_relation_indices.size());
         const int64_t nn_size = static_cast< int64_t >(lgan.nn_relation_indices.size());
         const int64_t rr_size = static_cast< int64_t >(lgan.rr_src_relation_indices.size());
         set_field(kLGANTNSizesField, std::span{&tn_size, size_t{1}});
         set_field(kLGANTNRelationIndicesField, std::span{lgan.tn_relation_indices});
         set_field(kLGANTNEntityIndicesField, std::span{lgan.tn_entity_indices});
         set_field(kLGANNNSizesField, std::span{&nn_size, size_t{1}});
         set_field(kLGANNNRelationIndicesField, std::span{lgan.nn_relation_indices});
         set_field(kLGANNNEntityIndicesField, std::span{lgan.nn_entity_indices});
         set_field(kLGANRRSizesField, std::span{&rr_size, size_t{1}});
         set_field(kLGANRRSrcRelationIndicesField, std::span{lgan.rr_src_relation_indices});
         set_field(kLGANRRDstRelationIndicesField, std::span{lgan.rr_dst_relation_indices});
      }
   }

   BatchBuilder::BatchEncoding encode_horizon_composed(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& horizon
   ) const
   {
      if(composition_plan == nullptr) {
         throw std::logic_error("semantic flat composition plan is not available");
      }
      const auto prepared = prepare_horizon_graph(dag, horizon);
      auto actual = composition_plan->encode(FlatInputView::from(prepared));
      finalize_horizon_encoding(actual, horizon);
      return actual;
   }

   void append_horizon_composed(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& horizon,
      BatchBuilder& builder
   ) const
   {
      if(composition_plan == nullptr) {
         throw std::logic_error("semantic flat composition plan is not available");
      }
      const auto prepared = prepare_horizon_graph(dag, horizon);
      composition_plan->append_graph(FlatInputView::from(prepared), builder);
   }

   BatchBuilder::BatchEncoding encode_horizon_composed_batch(
      const std::vector< SemanticTransitionDAG >& dags,
      const SemanticFlatHorizonEncoderConfig& horizon
   ) const
   {
      if(composition_plan == nullptr) {
         throw std::logic_error("semantic flat composition plan is not available");
      }

      std::vector< PreparedHorizonGraph > graphs;
      graphs.reserve(dags.size());
      for(const auto& dag : dags) {
         graphs.push_back(prepare_horizon_graph(dag, horizon));
      }
      if(horizon.export_node_names and std::ranges::any_of(graphs, [](const auto& graph) {
            return not graph.context.target_columns.names.empty();
         })) {
         for(auto& graph : graphs) {
            graph.suppress_empty_target_names = true;
         }
      }

      std::vector< FlatInputView > views;
      views.reserve(graphs.size());
      for(const auto& graph : graphs) {
         views.push_back(FlatInputView::from(graph));
      }
      auto actual = composition_plan->encode_batch(std::span{views});
      finalize_horizon_encoding(actual, horizon);
      return actual;
   }

   void finalize_horizon_encoding(
      BatchBuilder::BatchEncoding& encoding,
      const SemanticFlatHorizonEncoderConfig& horizon
   ) const
   {
      FlatRelationMajorWriter writer(
         std::span{schema_.arities()}, horizon.pack_relation_args_relation_major
      );
      writer.finalize(encoding);
   }

   BatchBuilder::BatchEncoding encode_many(
      std::span< const SemanticFlatRelationInput > inputs
   ) const
   {
      return compose_many(inputs);
   }

   void append_composed(const SemanticFlatRelationInput& input, BatchBuilder& builder) const
   {
      if(composition_plan == nullptr) {
         throw std::logic_error("semantic flat composition plan is not available");
      }
      const auto prepared = prepare_relation_graph(input);
      composition_plan->append_graph(FlatInputView::from(prepared), builder);
   }

   void encode_one_into(const SemanticFlatRelationInput& input, BatchBuilder& builder) const
   {
      prepare_builder(builder);
      std::vector< std::string > target_names;
      encode_into(input, &builder, nullptr, target_names);
      if(not target_group_names.empty() and config.export_node_names) {
         if(target_names.empty()) {
            builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         } else {
            builder.add_lazy_target_names(std::span{target_names});
         }
      }
   }

   BatchBuilder::BatchEncoding compose_many(
      std::span< const SemanticFlatRelationInput > inputs
   ) const
   {
      if(composition_plan == nullptr) {
         throw std::logic_error("semantic flat composition plan is not available");
      }
      std::vector< PreparedRelationGraph > graphs;
      graphs.reserve(inputs.size());
      for(const auto& input : inputs) {
         graphs.push_back(prepare_relation_graph(input));
      }
      if(not target_group_names.empty() and config.export_node_names
         and std::ranges::any_of(graphs, [](const auto& graph) {
                return not graph.context.target_columns.names.empty();
             })) {
         for(auto& graph : graphs) {
            graph.suppress_empty_target_names = true;
         }
      }
      std::vector< FlatInputView > views;
      views.reserve(graphs.size());
      for(const auto& graph : graphs) {
         views.push_back(FlatInputView::from(graph));
      }
      auto actual = composition_plan->encode_batch(std::span{views});
      finalize_batch_encoding(actual);
      return actual;
   }

   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
   {
      FlatRelationMajorWriter writer(
         std::span{schema_.arities()}, config.pack_relation_args_relation_major
      );
      writer.finalize(encoding);
   }
};

SemanticFlatRelationEncoderEngine::SemanticFlatRelationEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(predicates), std::move(actions), std::move(config)))
{
}

SemanticFlatRelationEncoderEngine::SemanticFlatRelationEncoderEngine(
   std::shared_ptr< const SemanticTaskContext > task_context,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(task_context), std::move(config)))
{
}

SemanticFlatRelationEncoderEngine::SemanticFlatRelationEncoderEngine(
   SemanticFlatRelationEncoderEngine&&
) noexcept = default;

SemanticFlatRelationEncoderEngine& SemanticFlatRelationEncoderEngine::operator=(
   SemanticFlatRelationEncoderEngine&&
) noexcept = default;

SemanticFlatRelationEncoderEngine::~SemanticFlatRelationEncoderEngine() = default;

BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode(
   const SemanticFlatRelationInput& input
) const
{
   return impl_->encode_many(std::span{&input, size_t{1}});
}

void SemanticFlatRelationEncoderEngine::encode(
   const SemanticFlatRelationInput& input,
   BatchBuilder& builder
) const
{
   impl_->append_composed(input, builder);
}

BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode_batch(
   const std::vector< SemanticFlatRelationInput >& inputs
) const
{
   return impl_->encode_many(std::span{inputs});
}

void SemanticFlatRelationEncoderEngine::finalize_batch_encoding(
   BatchBuilder::BatchEncoding& encoding
) const
{
   impl_->finalize_batch_encoding(encoding);
}

const SemanticFlatRelationEncoderEngine::Config&
SemanticFlatRelationEncoderEngine::get_config() const
{
   return impl_->config;
}

const std::shared_ptr< const SemanticTaskContext >&
SemanticFlatRelationEncoderEngine::get_task_context() const
{
   return impl_->task_context;
}

const std::vector< SemanticPredicateSpec >&
SemanticFlatRelationEncoderEngine::get_predicates() const
{
   return impl_->predicates;
}

const std::vector< SemanticActionSpec >& SemanticFlatRelationEncoderEngine::get_actions() const
{
   return impl_->actions;
}

const std::vector< std::string >& SemanticFlatRelationEncoderEngine::get_relation_names() const
{
   return impl_->schema_.names();
}

const std::vector< int64_t >& SemanticFlatRelationEncoderEngine::get_relation_arities() const
{
   return impl_->schema_.arities();
}

const std::vector< std::string >& SemanticFlatRelationEncoderEngine::get_relation_sources() const
{
   return impl_->schema_.sources();
}

const std::vector< int64_t >&
SemanticFlatRelationEncoderEngine::get_relation_logical_arities() const
{
   return impl_->schema_.logical_arities();
}

const std::vector< int64_t >&
SemanticFlatRelationEncoderEngine::get_relation_encoded_arities() const
{
   return impl_->schema_.encoded_arities();
}

const std::vector< int64_t >& SemanticFlatRelationEncoderEngine::get_relation_slot_roles() const
{
   return impl_->schema_.slot_roles();
}

const std::vector< int64_t >&
SemanticFlatRelationEncoderEngine::get_relation_slot_role_offsets() const
{
   return impl_->schema_.slot_role_offsets();
}

const std::vector< std::string >& SemanticFlatRelationEncoderEngine::get_slot_role_names() const
{
   return impl_->schema_.slot_role_names();
}

void SemanticFlatRelationEncoderEngine::configure_horizon(
   const SemanticFlatHorizonEncoderConfig& config
)
{
   impl_->configure_horizon(config);
}

void SemanticFlatRelationEncoderEngine::prepare_horizon_builder(
   BatchBuilder& builder,
   const SemanticFlatHorizonEncoderConfig& config
) const
{
   impl_->prepare_horizon_builder(builder, config);
}

void SemanticFlatRelationEncoderEngine::encode_horizon(
   const SemanticTransitionDAG& dag,
   const SemanticFlatHorizonEncoderConfig& config,
   BatchBuilder& builder
) const
{
   impl_->append_horizon_composed(dag, config, builder);
}

BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode_horizon_composed(
   const SemanticTransitionDAG& dag,
   const SemanticFlatHorizonEncoderConfig& config
) const
{
   return impl_->encode_horizon_composed(dag, config);
}

BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode_horizon_composed_batch(
   const std::vector< SemanticTransitionDAG >& dags,
   const SemanticFlatHorizonEncoderConfig& config
) const
{
   return impl_->encode_horizon_composed_batch(dags, config);
}

void SemanticFlatRelationEncoderEngine::finalize_horizon_encoding(
   BatchBuilder::BatchEncoding& encoding,
   const SemanticFlatHorizonEncoderConfig& config
) const
{
   impl_->finalize_horizon_encoding(encoding, config);
}

}  // namespace mifrost
