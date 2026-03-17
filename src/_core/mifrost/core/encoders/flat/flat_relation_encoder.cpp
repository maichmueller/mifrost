/**
 * @file flat_relation_encoder.cpp
 * @brief Main implementation file for the flat relation encoder.
 *
 * Tuple layout, schema setup, node-table helpers, and target metadata helpers
 * live in nearby files. This file keeps the relation encoder's own emit order
 * and component setup.
 */
#include "flat_relation_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <mimir/formalism/problem.hpp>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "flat_encoder_common.hpp"
#include "flat_lgan.hpp"
#include "flat_relation_context.hpp"
#include "flat_tuple_args.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

namespace {

FlatRelationConfigView relation_config_view(const FlatRelationEncoderEngine::Config& config)
{
   return FlatRelationConfigView{
      .include_lgan_edges = config.include_lgan_edges,
      .use_predicate_virtual_nodes = config.use_predicate_virtual_nodes,
      .ignore_zero_arity_relations = config.ignore_zero_arity_relations,
      .lgan_anchor_sources = config.lgan_anchor_sources,
      .target_sources = config.target_sources,
   };
}

template < typename AtomTag >
std::vector< int64_t > local_arg_rows_for_atom(
   const FlatRelationEncoderEngine& engine,
   FlatRelationEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAtom< AtomTag >& atom,
   std::span< const int64_t > auxiliary_args = {}
)
{
   return build_flat_atom_tuple_args(
      context,
      atom,
      auxiliary_args,
      engine.get_config().use_predicate_virtual_nodes,
      "Flat relation encoder encountered object not present in entity table: "
   );
}

template < typename ErrorMsgCallable >
   requires std::is_invocable_r_v< std::string, ErrorMsgCallable >
int64_t lookup_target_entity_index(
   const FlatRelationEncoderEngine::EncodingContext& context,
   const FlatTargetEntityKey& key,
   const ErrorMsgCallable& missing_message
)
{
   const auto it = context.target_entity_index_by_key.find(key);
   if(it == context.target_entity_index_by_key.end()) {
      throw std::invalid_argument(missing_message());
   }
   return it->second;
}

template < typename GoalTag >
int64_t lookup_goal_target_entity_index(
   const FlatRelationEncoderEngine::EncodingContext& context,
   TargetSource source,
   const mimir::formalism::GroundLiteral< GoalTag >& literal,
   const std::optional< size_t >& goal_level
)
{
   return lookup_target_entity_index(
      context, goal_target_entity_key(source, literal, goal_level), [&] {
         return std::string{
            "Flat relation encoder encountered goal literal without a target entity row: "
            + goal_target_display_name(literal, goal_level)
         };
      }
   );
}

template < typename HistoryTag >
int64_t lookup_history_target_entity_index(
   const FlatRelationEncoderEngine::EncodingContext& context,
   int dt,
   size_t entry_idx,
   const mimir::formalism::GroundLiteral< HistoryTag >& literal
)
{
   return lookup_target_entity_index(
      context, history_target_entity_key(dt, entry_idx, literal), [&] {
         return std::string{
            "Flat relation encoder encountered history literal without a target entity row: "
            + history_target_display_name(dt, entry_idx, literal)
         };
      }
   );
}

std::vector< int64_t > local_arg_rows_for_action(
   const FlatRelationEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAction& action
)
{
   const int64_t action_entity_index = lookup_target_entity_index(
      context, action_target_entity_key(action), [&] {
         return std::string{
            "Flat relation encoder encountered grounded action without a target entity row: "
            + RelationFormatter::format_action(action)
         };
      }
   );
   const std::array auxiliary_args{action_entity_index};
   return build_flat_action_tuple_args(
      context,
      action,
      std::span{auxiliary_args},
      "Flat relation encoder encountered action object not present in entity table: "
   );
}

template < typename GoalTag >
std::vector< int64_t > local_arg_rows_for_goal_literal(
   const FlatRelationEncoderEngine& engine,
   FlatRelationEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundLiteral< GoalTag >& literal,
   const std::optional< size_t >& goal_level
)
{
   if(const auto target_source = anchor_source_for_goal_level(
         relation_config_view(engine.get_config()), goal_level
      );
      target_source.has_value()) {
      const std::array auxiliary_args{
         lookup_goal_target_entity_index(context, *target_source, literal, goal_level)
      };
      return local_arg_rows_for_atom(
         engine, context, literal->get_atom(), std::span{auxiliary_args}
      );
   }
   return local_arg_rows_for_atom(engine, context, literal->get_atom());
}

template < typename HistoryTag >
std::vector< int64_t > local_arg_rows_for_history_literal(
   const FlatRelationEncoderEngine& engine,
   FlatRelationEncoderEngine::EncodingContext& context,
   int64_t history_entity_index,
   int dt,
   size_t entry_idx,
   const mimir::formalism::GroundLiteral< HistoryTag >& literal
)
{
   if(has_anchor_entity_source(relation_config_view(engine.get_config()), TargetSource::history)) {
      const std::array auxiliary_args{
         lookup_history_target_entity_index(context, dt, entry_idx, literal), history_entity_index
      };
      return local_arg_rows_for_atom(
         engine, context, literal->get_atom(), std::span{auxiliary_args}
      );
   }
   const std::array auxiliary_args{history_entity_index};
   return local_arg_rows_for_atom(engine, context, literal->get_atom(), std::span{auxiliary_args});
}

}  // namespace

class FlatRelationEncoderEngine::RelationComponent {
  public:
   virtual ~RelationComponent() = default;
   virtual void declare_schema(
      const FlatRelationEncoderEngine& engine,
      FlatRelationSchemaRegistry& registry
   ) const = 0;
   virtual void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const hash_set< uint64_t >& fact_keys,
      EncodingContext& context,
      FlatRelationSink& sink
   ) const = 0;
};

class FlatRelationEncoderEngine::StateFactsComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      FlatRelationSchemaRegistry& registry
   ) const override
   {
      for(const auto& spec : engine.predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         registry.add(
            spec.name,
            make_predicate_tuple_layout(spec.arity, {}, engine.config_.use_predicate_virtual_nodes),
            "state"
         );
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State& state,
      const GoalInputs&,
      const hash_set< uint64_t >&,
      EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      auto emit_atom = [&](const auto& atom) {
         const int arity = static_cast< int >(atom->get_predicate()->get_arity());
         if(engine.config_.ignore_zero_arity_relations and arity == 0) {
            return;
         }
         const auto relation_id = engine.relation_id_for(
            RelationFormatter::format_predicate(atom->get_predicate())
         );
         const auto args = local_arg_rows_for_atom(engine, context, atom);
         sink.emit(relation_id, args);
      };
      for_each_state_fact_atom(state, engine.config_.include_static, emit_atom);
   }
};

class FlatRelationEncoderEngine::GoalFactsComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      FlatRelationSchemaRegistry& registry
   ) const override
   {
      if(not includes_plain_goal_derivation(engine.config_.goal_derivations)) {
         return;
      }
      for(const auto& spec : engine.regular_predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         for(size_t level = 0; level <= engine.config_.max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               registry.add(
                  RelationFormatter::format_predicate(
                     spec.name, goal_level, std::nullopt, polarity
                  ),
                  goal_relation_layout(
                     relation_config_view(engine.get_config()), spec.arity, level
                  ),
                  "goal"
               );
            }
         }
         if(engine.config_.support_literals) {
            for(bool polarity : {true, false}) {
               registry.add(
                  RelationFormatter::format_predicate(
                     spec.name, std::nullopt, std::nullopt, polarity
                  ),
                  goal_relation_layout(
                     relation_config_view(engine.get_config()), spec.arity, std::nullopt
                  ),
                  "goal"
               );
            }
         }
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State&,
      const GoalInputs& goals,
      const hash_set< uint64_t >&,
      EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      if(not includes_plain_goal_derivation(engine.config_.goal_derivations)) {
         return;
      }
      emit_literals(std::span{goals.static_goals}, goals.static_goal_levels, engine, context, sink);
      emit_literals(std::span{goals.fluent_goals}, goals.fluent_goal_levels, engine, context, sink);
      emit_literals(
         std::span{goals.derived_goals}, goals.derived_goal_levels, engine, context, sink
      );
   }

  private:
   template < typename GoalTag >
   static void emit_literals(
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
      const auto& goal_levels,
      const FlatRelationEncoderEngine& engine,
      EncodingContext& context,
      FlatRelationSink& sink
   )
   {
      for(const auto& literal : literals) {
         const auto predicate = literal->get_atom()->get_predicate();
         const int arity = static_cast< int >(predicate->get_arity());
         if(engine.config_.ignore_zero_arity_relations and arity == 0) {
            continue;
         }

         const auto level = goal_level_for(goal_levels, literal);
         std::string relation_name;
         if(level.has_value()) {
            relation_name = RelationFormatter::format_predicate(
               predicate, GoalLevel(*level), std::nullopt, literal->get_polarity()
            );
         } else {
            relation_name = RelationFormatter::format_predicate(
               predicate, std::nullopt, std::nullopt, literal->get_polarity()
            );
         }

         const auto relation_id = engine.relation_id_for(relation_name);
         const auto args = local_arg_rows_for_goal_literal(engine, context, literal, level);
         sink.emit(relation_id, args);
      }
   }
};

class FlatRelationEncoderEngine::GoalDerivationComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      FlatRelationSchemaRegistry& registry
   ) const override
   {
      if(not has_non_plain_goal_derivations(engine.config_.goal_derivations)) {
         return;
      }
      for(const auto& spec : engine.regular_predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         for(const auto derivation :
             goal_satisfaction_derivations(engine.config_.goal_derivations)) {
            for(size_t level = 0; level <= engine.config_.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(bool polarity : {true, false}) {
                  registry.add(
                     RelationFormatter::format_predicate(
                        spec.name, goal_level, derivation, polarity
                     ),
                     make_predicate_tuple_layout(
                        spec.arity, {}, engine.config_.use_predicate_virtual_nodes
                     ),
                     "goal_derivation"
                  );
               }
            }
            if(engine.config_.support_literals) {
               for(bool polarity : {true, false}) {
                  registry.add(
                     RelationFormatter::format_predicate(
                        spec.name, std::nullopt, derivation, polarity
                     ),
                     make_predicate_tuple_layout(
                        spec.arity, {}, engine.config_.use_predicate_virtual_nodes
                     ),
                     "goal_derivation"
                  );
               }
            }
         }
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State&,
      const GoalInputs& goals,
      const hash_set< uint64_t >& fact_keys,
      EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      if(not has_non_plain_goal_derivations(engine.config_.goal_derivations)) {
         return;
      }
      emit_literals(
         std::span{goals.static_goals}, goals.static_goal_levels, fact_keys, engine, context, sink
      );
      emit_literals(
         std::span{goals.fluent_goals}, goals.fluent_goal_levels, fact_keys, engine, context, sink
      );
      emit_literals(
         std::span{goals.derived_goals}, goals.derived_goal_levels, fact_keys, engine, context, sink
      );
   }

  private:
   template < typename GoalTag >
   static void emit_literals(
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
      const auto& goal_levels,
      const hash_set< uint64_t >& fact_keys,
      const FlatRelationEncoderEngine& engine,
      EncodingContext& context,
      FlatRelationSink& sink
   )
   {
      for(const auto& literal : literals) {
         const auto predicate = literal->get_atom()->get_predicate();
         const int arity = static_cast< int >(predicate->get_arity());
         if(engine.config_.ignore_zero_arity_relations and arity == 0) {
            continue;
         }

         const uint64_t fact_key = state_fact_key_for_atom(literal->get_atom());
         const bool satisfied = fact_keys.contains(fact_key) == literal->get_polarity();
         const GoalDerivation satisfaction = satisfied ? GoalDerivation::satisfied
                                                       : GoalDerivation::unsatisfied;
         if(not engine.config_.goal_derivations.contains(satisfaction)) {
            continue;
         }

         const auto level = goal_level_for(goal_levels, literal);
         std::string relation_name;
         if(level.has_value()) {
            relation_name = RelationFormatter::format_predicate(
               predicate, GoalLevel(*level), satisfaction, literal->get_polarity()
            );
         } else {
            relation_name = RelationFormatter::format_predicate(
               predicate, std::nullopt, satisfaction, literal->get_polarity()
            );
         }

         const auto relation_id = engine.relation_id_for(relation_name);
         const auto args = local_arg_rows_for_atom(engine, context, literal->get_atom());
         sink.emit(relation_id, args);
      }
   }
};

class FlatRelationEncoderEngine::GroundActionsComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      FlatRelationSchemaRegistry& registry
   ) const override
   {
      for(const auto& spec : engine.action_specs_) {
         registry.add(
            spec.name,
            make_nonpredicate_tuple_layout(spec.arity - 1, {FlatSlotRole::action_slot}),
            "action"
         );
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State&,
      const GoalInputs&,
      const hash_set< uint64_t >&,
      EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      for(const auto& action : context.unique_actions) {
         const auto relation_id = engine.relation_id_for(
            RelationFormatter::format_action_schema(*action->get_action())
         );
         const auto args = local_arg_rows_for_action(context, action);
         sink.emit(relation_id, args);
      }
   }
};

class FlatRelationEncoderEngine::HistoryFactsComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      FlatRelationSchemaRegistry& registry
   ) const override
   {
      for(const auto& spec : engine.predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         for(bool polarity : {true, false}) {
            registry.add(
               RelationFormatter::format_predicate(
                  spec.name, std::nullopt, std::nullopt, polarity, "[hist]"
               ),
               history_relation_layout(relation_config_view(engine.get_config()), spec.arity),
               "history"
            );
         }
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State&,
      const GoalInputs&,
      const hash_set< uint64_t >&,
      EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      for(const auto& entry : context.history_entries) {
         for(const auto& literal_variant : entry.literals) {
            std::visit(
               [&]< typename HistoryTag >(
                  const mimir::formalism::GroundLiteral< HistoryTag >& literal
               ) {
                  const auto predicate = literal->get_atom()->get_predicate();
                  const int arity = static_cast< int >(predicate->get_arity());
                  if(engine.config_.ignore_zero_arity_relations and arity == 0) {
                     return;
                  }
                  const auto relation_id = engine.relation_id_for(
                     history_relation_name(predicate, literal->get_polarity())
                  );
                  const auto args = local_arg_rows_for_history_literal(
                     engine, context, entry.entity_index, entry.dt, entry.entry_idx, literal
                  );
                  sink.emit(relation_id, args);
               },
               literal_variant
            );
         }
      }
   }
};

FlatRelationEncoderEngine::FlatRelationEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : FlatRelationEncoderEngine(domain, Config{})
{
}

FlatRelationEncoderEngine::FlatRelationEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : domain_(domain), config_(std::move(config))
{
   validate_config();
   initialize_from_domain();
}

FlatRelationEncoderEngine::FlatRelationEncoderEngine(mimir::formalism::Domain domain)
    : FlatRelationEncoderEngine(std::move(domain), Config{})
{
}

FlatRelationEncoderEngine::FlatRelationEncoderEngine(mimir::formalism::Domain domain, Config config)
    : domain_holder_(std::move(domain)), domain_(*domain_holder_), config_(std::move(config))
{
   validate_config();
   initialize_from_domain();
}

FlatRelationEncoderEngine::~FlatRelationEncoderEngine() = default;

void FlatRelationEncoderEngine::validate_config() const
{
   auto validate_sources = [](const std::ranges::range auto& sources, std::string_view field_name) {
      for(const TargetSource source : sources) {
         if(source == TargetSource::actions or source == TargetSource::goals
            or source == TargetSource::subgoals or source == TargetSource::history) {
            continue;
         }
         throw std::invalid_argument(
            fmt::format(
               "FlatRelationEncoder currently supports {}={{'action', 'goal', "
               "'subgoal', 'history'}} only; 'state' is reserved for the upcoming flat "
               "successor/horizon encoders",
               field_name
            )
         );
      }
   };
   validate_sources(config_.target_sources, "target_sources");
   validate_sources(config_.lgan_anchor_sources, "lgan_anchor_sources");
}

void FlatRelationEncoderEngine::initialize_from_domain()
{
   predicate_specs_.clear();
   regular_predicate_specs_.clear();
   action_specs_.clear();

   RelationDictConfig rel_config;
   rel_config.max_goal_level = static_cast< int >(config_.max_goal_level);
   rel_config.support_literals = config_.support_literals;
   rel_config.goal_derivations = config_.goal_derivations;

   std::vector< mimir::formalism::Action > actions;
   actions.assign(domain_.get_actions().begin(), domain_.get_actions().end());

   const auto top_type_predicates = rel_config.top_type_predicates;
   auto collect_predicates = [&]< typename Tag >(Tag) {
      for(const auto predicate : domain_.get_predicates< Tag >()) {
         PredicateSpec spec{
            .name = RelationFormatter::format_predicate(predicate),
            .arity = static_cast< int >(predicate->get_arity()),
         };
         predicate_specs_.push_back(spec);
         if(not top_type_predicates.contains(predicate->get_name())) {
            regular_predicate_specs_.push_back(spec);
         }
      }
   };
   collect_predicates(mimir::formalism::StaticTag{});
   collect_predicates(mimir::formalism::FluentTag{});
   collect_predicates(mimir::formalism::DerivedTag{});

   auto predicate_order = [](const PredicateSpec& lhs, const PredicateSpec& rhs) {
      return lhs.name < rhs.name;
   };
   std::ranges::sort(predicate_specs_, predicate_order);
   std::ranges::sort(regular_predicate_specs_, predicate_order);

   for(const auto& action : actions) {
      const auto layout = make_nonpredicate_tuple_layout(
         action->get_arity(), {FlatSlotRole::action_slot}
      );
      action_specs_.push_back(
         PredicateSpec{
            .name = RelationFormatter::format_action_schema(*action),
            .arity = layout.encoded_arity(),
         }
      );
   }
   std::ranges::sort(action_specs_, predicate_order);

   target_entity_group_names_.clear();
   target_entity_group_ids_.clear();
   target_metadata_group_names_.clear();
   target_metadata_group_ids_.clear();

   auto append_group = [](std::vector< std::string >& group_names,
                          std::map< TargetSource, int64_t >& group_ids,
                          TargetSource source) {
      if(group_ids.contains(source)) {
         return;
      }
      const auto next_id = static_cast< int64_t >(group_names.size());
      group_ids.emplace(source, next_id);
      group_names.emplace_back(target_source_group_name(source));
   };

   for(const auto source : kCanonicalTargetSourceOrder) {
      if(source == TargetSource::actions or has_anchor_entity_source(source)) {
         if(source == TargetSource::states) {
            continue;
         }
         append_group(target_entity_group_names_, target_entity_group_ids_, source);
      }
      if(has_target_source(source)) {
         append_group(target_metadata_group_names_, target_metadata_group_ids_, source);
      }
   }

   components_.clear();
   components_.push_back(std::make_unique< StateFactsComponent >());
   components_.push_back(std::make_unique< GoalFactsComponent >());
   components_.push_back(std::make_unique< GoalDerivationComponent >());
   components_.push_back(std::make_unique< GroundActionsComponent >());
   components_.push_back(std::make_unique< HistoryFactsComponent >());

   rebuild_schema();
}

void FlatRelationEncoderEngine::rebuild_schema()
{
   FlatRelationSchemaRegistry registry;
   for(const auto& component : components_) {
      component->declare_schema(*this, registry);
   }

   const auto metadata = build_flat_relation_schema_metadata(
      registry,
      static_cast< int >(config_.max_goal_level),
      config_.support_literals,
      config_.goal_derivations,
      "FlatRelationEncoderEngine did not derive any relation types for this domain/config"
   );

   relation_dict_ = metadata.relation_dict;
   relation_names_ = metadata.relation_names;
   relation_arities_ = metadata.relation_arities;
   relation_sources_ = metadata.relation_sources;
   relation_logical_arities_ = metadata.relation_logical_arities;
   relation_encoded_arities_ = metadata.relation_encoded_arities;
   relation_slot_roles_ = metadata.relation_slot_roles;
   relation_slot_role_offsets_ = metadata.relation_slot_role_offsets;
   slot_role_names_ = metadata.slot_role_names;
   relation_name_to_id_ = metadata.relation_name_to_id;
}

void FlatRelationEncoderEngine::prepare_builder(BatchBuilder& builder) const
{
   const FlatRelationSchemaMetadata metadata{
      .relation_dict = relation_dict_,
      .relation_names = relation_names_,
      .relation_arities = relation_arities_,
      .relation_sources = relation_sources_,
      .relation_logical_arities = relation_logical_arities_,
      .relation_encoded_arities = relation_encoded_arities_,
      .relation_slot_roles = relation_slot_roles_,
      .relation_slot_role_offsets = relation_slot_role_offsets_,
      .slot_role_names = slot_role_names_,
      .relation_name_to_id = relation_name_to_id_,
   };
   set_flat_graph_attrs(
      builder,
      metadata,
      FlatBuilderGraphConfig{
         .include_lgan_edges = config_.include_lgan_edges,
         .use_predicate_virtual_nodes = config_.use_predicate_virtual_nodes,
         .target_sources = source_names_for(config_.target_sources),
         .lgan_anchor_sources = source_names_for(config_.lgan_anchor_sources),
         .target_symbol_prefix = config_.target_symbol_prefix,
         .target_entity_group_names = target_entity_group_names_,
         .lgan_tn_edge_pos = config_.lgan_tn_edge_pos,
         .lgan_nn_edge_pos = config_.lgan_nn_edge_pos,
         .lgan_rr_edge_pos = config_.lgan_rr_edge_pos,
      }
   );

   register_flat_entity_fields(builder);
   register_flat_history_entity_fields(builder);
   register_flat_target_entity_fields(builder);
   if(supports_target_metadata()) {
      builder.register_field(
         std::string(kTargetSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
      const TargetMetadataEmitConfig target_emit_config{
         .position_node_type_id = std::string(kFlatEntityNodeType),
         .symbol_prefix = config_.target_symbol_prefix,
         .include_depth = false,
         .include_group = true,
         .include_names = false,
         .groups = target_metadata_group_names_,
         .parent_relation = std::nullopt,
      };
      register_target_fields(builder, target_emit_config);
      builder.set_graph_attr(std::string(kTargetGroupsAttr), target_metadata_group_names_);
      builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   }
   register_flat_relation_instance_fields(builder, static_cast< int >(relation_names_.size()));
   if(config_.include_lgan_edges) {
      register_flat_lgan_fields(builder);
   }
}

FlatRelationEncoderEngine::EncodingContext FlatRelationEncoderEngine::make_context(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps
) const
{
   return build_flat_relation_encoding_context(
      state,
      goals,
      actions,
      history_subgoals,
      history_max_steps,
      FlatRelationContextBuildConfig{
         .relation_config = relation_config_view(config_),
         .ignore_zero_arity_relations = config_.ignore_zero_arity_relations,
         .supports_target_metadata = supports_target_metadata(),
         .predicate_symbol_capacity = predicate_specs_.size(),
         .target_entity_group_ids = &target_entity_group_ids_,
         .target_metadata_group_ids = &target_metadata_group_ids_,
      }
   );
}

int FlatRelationEncoderEngine::relation_id_for(const std::string& name) const
{
   const auto it = relation_name_to_id_.find(name);
   if(it == relation_name_to_id_.end()) {
      throw std::invalid_argument("Unknown flat relation name '" + name + "'");
   }
   return it->second;
}

bool FlatRelationEncoderEngine::has_target_source(TargetSource source) const
{
   return config_.target_sources.contains(source);
}

bool FlatRelationEncoderEngine::has_lgan_anchor_source(TargetSource source) const
{
   return config_.include_lgan_edges and config_.lgan_anchor_sources.contains(source);
}

bool FlatRelationEncoderEngine::has_anchor_entity_source(TargetSource source) const
{
   return has_target_source(source) or has_lgan_anchor_source(source);
}

bool FlatRelationEncoderEngine::supports_target_metadata() const
{
   return not target_metadata_group_names_.empty();
}

int64_t FlatRelationEncoderEngine::target_entity_group_id(TargetSource source) const
{
   const auto it = target_entity_group_ids_.find(source);
   if(it == target_entity_group_ids_.end()) {
      throw std::invalid_argument(
         "FlatRelationEncoder does not define a target-entity group for source '"
         + std::string(target_source_group_name(source)) + "'"
      );
   }
   return it->second;
}

int64_t FlatRelationEncoderEngine::target_metadata_group_id(TargetSource source) const
{
   const auto it = target_metadata_group_ids_.find(source);
   if(it == target_metadata_group_ids_.end()) {
      throw std::invalid_argument(
         "FlatRelationEncoder does not define target metadata for source '"
         + std::string(target_source_group_name(source)) + "'"
      );
   }
   return it->second;
}

void FlatRelationEncoderEngine::encode_default_goals(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder
)
{
   encode_impl(
      state,
      default_goal_inputs_for_state(state),
      actions,
      history_subgoals,
      history_max_steps,
      builder
   );
}

void FlatRelationEncoderEngine::encode(const mimir::search::State& state, BatchBuilder& builder)
{
   encode_default_goals(state, {}, {}, std::nullopt, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   encode_default_goals(state, actions, {}, std::nullopt, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, {}, {}, std::nullopt, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, actions, {}, std::nullopt, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, actions, history_subgoals, history_max_steps, builder);
}

void FlatRelationEncoderEngine::encode_impl(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder,
   std::vector< std::string >* batch_target_names,
   bool prepare_builder_once
)
{
   // Summary:
   // 1. Build the per-graph entity and target tables.
   // 2. Emit all relation tuples into a flat sink.
   // 3. Write node rows, target rows, tuple data, and optional LGAN edges.
   // Phase 1: prepare shared schema state once, then build the per-graph context.
   if(prepare_builder_once) {
      prepare_builder(builder);
   }
   auto context = make_context(state, goals, actions, history_subgoals, history_max_steps);
   FlatRelationSink sink(relation_names_.size(), config_.include_lgan_edges);

   // Phase 2: collect state facts so later goal-derivation emission can test satisfaction.
   hash_set< uint64_t > fact_keys;
   collect_state_fact_keys(state, config_.include_static, fact_keys);

   // Phase 3: let each configured component write its relation tuples into the sink.
   for(const auto& component : components_) {
      component->emit(*this, state, goals, fact_keys, context, sink);
   }

   // Phase 4: export node rows and per-node metadata for this graph.
   std::vector< float > zeros(context.entity_names.size(), 0.0f);
   builder.add_node_features(
      std::string(kFlatEntityNodeType), "x", std::span< const float >(zeros.data(), zeros.size()), 1
   );

   if(config_.export_node_names) {
      builder.set_node_names(std::string(kFlatEntityNodeType), context.entity_names);
      builder.set_object_names(context.object_names);
   }

   const int64_t node_size = static_cast< int64_t >(context.entity_names.size());
   const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
   const int64_t history_entity_size = static_cast< int64_t >(
      context.history_entity_indices.size()
   );
   const int64_t target_entity_size = static_cast< int64_t >(context.target_entity_indices.size());
   builder.set_field(std::string(kNodeSizesField), std::span< const int64_t >(&node_size, 1));
   builder.set_field(std::string(kObjectSizesField), std::span< const int64_t >(&object_size, 1));
   builder.set_field(
      std::string(kObjectIndicesField),
      std::span< const int64_t >(context.object_indices.data(), context.object_indices.size())
   );
   builder.set_field(
      std::string(kEntityRoleIdsField),
      std::span< const int64_t >(context.entity_role_ids.data(), context.entity_role_ids.size())
   );
   builder.set_field(
      std::string(kHistoryEntitySizesField), std::span< const int64_t >(&history_entity_size, 1)
   );
   builder.set_field(
      std::string(kHistoryEntityIndicesField),
      std::span< const int64_t >(
         context.history_entity_indices.data(), context.history_entity_indices.size()
      )
   );
   builder.set_field(
      std::string(kHistoryEntityDtField),
      std::span< const int64_t >(context.history_entity_dt.data(), context.history_entity_dt.size())
   );
   builder.set_field(
      std::string(kTargetEntitySizesField), std::span< const int64_t >(&target_entity_size, 1)
   );
   builder.set_field(
      std::string(kTargetEntityIndicesField),
      std::span< const int64_t >(
         context.target_entity_indices.data(), context.target_entity_indices.size()
      )
   );
   builder.set_field(
      std::string(kTargetEntityGroupIdsField),
      std::span< const int64_t >(
         context.target_entity_group_ids.data(), context.target_entity_group_ids.size()
      )
   );
   if(supports_target_metadata()) {
      // Target rows point back into the flat node table. Readout should gather the
      // target embeddings from these positions instead of inferring them from tuple slots.
      const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
      builder.set_field(
         std::string(kTargetSizesField), std::span< const int64_t >(&target_size, 1)
      );
      const TargetMetadataEmitConfig target_emit_config{
         .position_node_type_id = std::string(kFlatEntityNodeType),
         .symbol_prefix = config_.target_symbol_prefix,
         .include_depth = false,
         .include_group = true,
         .include_names = false,
         .groups = target_metadata_group_names_,
         .parent_relation = std::nullopt,
      };
      set_target_fields(builder, context.target_columns, target_emit_config);
      set_target_graph_attrs(builder, context.target_columns, target_emit_config);
      if(config_.export_node_names) {
         if(batch_target_names != nullptr) {
            if(not context.target_columns.names.empty()) {
               batch_target_names->insert(
                  batch_target_names->end(),
                  context.target_columns.names.begin(),
                  context.target_columns.names.end()
               );
            }
         } else if(context.target_columns.names.empty()) {
            builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         } else {
            builder.add_lazy_target_names(std::span(context.target_columns.names));
         }
      }
   }
   // Phase 5: export the relation tuples after all components have finished writing.
   builder.set_field(
      std::string(kRelationCountsField),
      std::span< const int64_t >(sink.relation_counts().data(), sink.relation_counts().size())
   );
   const int64_t relation_instance_size = sink.relation_instance_count();
   builder.set_field(
      std::string(kRelationInstanceSizesField),
      std::span< const int64_t >(&relation_instance_size, 1)
   );
   builder.set_field(
      std::string(kRelationArgsField),
      std::span< const int64_t >(sink.relation_args().data(), sink.relation_args().size())
   );
   if(config_.include_lgan_edges) {
      // Phase 6: derive LGAN helper edges from the emitted flat tuples and target rows.
      if(context.target_entity_indices.empty()) {
         throw std::invalid_argument(
            "FlatRelationEncoder include_lgan_edges=true requires LGAN anchor entity rows, "
            "but none were encoded. Encode explicit actions or enable anchor-row-emitting "
            "lgan_anchor_sources/target_sources such as 'goal', 'subgoal', or 'history'."
         );
      }
      const auto lgan = build_flat_lgan(sink, std::span{context.target_entity_indices});
      const int64_t tn_size = static_cast< int64_t >(lgan.tn_relation_indices.size());
      const int64_t nn_size = static_cast< int64_t >(lgan.nn_relation_indices.size());
      const int64_t rr_size = static_cast< int64_t >(lgan.rr_src_relation_indices.size());
      builder.set_field(std::string(kLGANTNSizesField), std::span< const int64_t >(&tn_size, 1));
      builder.set_field(
         std::string(kLGANTNRelationIndicesField),
         std::span< const int64_t >(
            lgan.tn_relation_indices.data(), lgan.tn_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANTNEntityIndicesField),
         std::span< const int64_t >(lgan.tn_entity_indices.data(), lgan.tn_entity_indices.size())
      );
      builder.set_field(std::string(kLGANNNSizesField), std::span< const int64_t >(&nn_size, 1));
      builder.set_field(
         std::string(kLGANNNRelationIndicesField),
         std::span< const int64_t >(
            lgan.nn_relation_indices.data(), lgan.nn_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANNNEntityIndicesField),
         std::span< const int64_t >(lgan.nn_entity_indices.data(), lgan.nn_entity_indices.size())
      );
      builder.set_field(std::string(kLGANRRSizesField), std::span< const int64_t >(&rr_size, 1));
      builder.set_field(
         std::string(kLGANRRSrcRelationIndicesField),
         std::span< const int64_t >(
            lgan.rr_src_relation_indices.data(), lgan.rr_src_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANRRDstRelationIndicesField),
         std::span< const int64_t >(
            lgan.rr_dst_relation_indices.data(), lgan.rr_dst_relation_indices.size()
         )
      );
   }
}

BatchBuilder::BatchEncoding FlatRelationEncoderEngine::encode_batch(
   const batch_input::parsed::FlatBatchInputs& inputs,
   std::optional< int > history_max_steps
)
{
   // Summary:
   // 1. Read one batch item at a time and normalize optional goals, actions, and history.
   // 2. Encode each item into the shared builder as one flat graph.
   // 3. Write batch-level target metadata after all graphs are appended.
   BatchBuilder builder;
   builder.set_graph_kind("flat");
   prepare_builder(builder);

   const size_t state_count = inputs.states.states.size();
   std::vector< std::string > batch_target_names;
   for(size_t idx = 0; idx < state_count; ++idx) {
      // Phase 1: collect the per-state inputs from the parsed batch.
      const auto& state_entry = inputs.states.states[idx];
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& actions_entry = inputs.actions.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);
      const auto& history_entry = inputs.history_subgoals.at(idx);

      // Phase 2: normalize optional goal payloads into one GoalInputs object.
      GoalInputs goal_inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         goal_inputs = batch_input::compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         goal_inputs = batch_input::default_goal_inputs_for_batch_state(state_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               goal_inputs.extend(layer, level);
               ++level;
            }
         }
      }

      const auto actions_span = actions_entry.has_value()
                                   ? std::span< const mimir::formalism::GroundAction >(
                                        *actions_entry
                                     )
                                   : std::span< const mimir::formalism::GroundAction >{};
      const auto history_span = history_entry.has_value()
                                   ? std::span< const HistorySubgoal >(*history_entry)
                                   : std::span< const HistorySubgoal >{};
      // Phase 3: encode one graph into the shared builder, then advance to the next slot.
      encode_impl(
         state_entry.state,
         goal_inputs,
         actions_span,
         history_span,
         history_max_steps,
         builder,
         &batch_target_names,
         /*prepare_builder_once=*/false
      );
      builder.next_graph();
   }

   if(supports_target_metadata()) {
      // Phase 4: write batch-level target metadata once all graphs have been appended.
      if(config_.export_node_names) {
         if(batch_target_names.empty()) {
            builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         } else {
            builder.add_lazy_target_names(std::span(batch_target_names));
         }
      }
      builder.set_graph_attr(std::string(kTargetGroupsAttr), target_metadata_group_names_);
      builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   }

   return builder.build();
}

}  // namespace mifrost
