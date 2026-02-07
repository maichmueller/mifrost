#pragma once

#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "batch_builder.hpp"
#include "common_types.hpp"
#include "default_relations.hpp"
#include "goal_inputs.hpp"
#include "relation_dict.hpp"
#include "relation_formatter.hpp"
#include "stream_encoder_base.hpp"

namespace mifrost {

/**
 * @brief Core heterogeneous graph encoder engine.
 *
 * Encodes states/goals/actions into relation-typed node/edge structures using
 * ``BatchBuilder`` output conventions.
 */
class HGraphEncoderEngine {
  public:
   using HistorySubgoal = std::pair< int, std::vector< GoalInputs::AnyGoalLiteral > >;

   /// Runtime configuration for relation/node/edge derivation behavior.
   struct Config {
      std::string symbol_type_id = defaults::symbol_type_id;
      std::string nullary_object_name = "![nullary_symbol]!";
      std::string lgan_nn_edge_pos = defaults::lgan_nn_edge_pos;
      std::string history_link_relation = defaults::history_link_relation;
      int max_goal_level = 0;
      bool support_literals = false;
      bool add_nullary_predicates = false;
      bool ignore_actions = true;
      bool include_lgan_edges = false;
      bool include_static = true;
      bool include_empty_edge_types = true;
      std::set< GoalSatisfaction > goal_satisfaction_derivations = {GoalSatisfaction::satisfied};
   };

   /// Construct from borrowed domain implementation reference.
   explicit HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
   HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);

   /// Construct from owning domain handle.
   explicit HGraphEncoderEngine(mimir::formalism::Domain domain);
   HGraphEncoderEngine(mimir::formalism::Domain domain, Config config);

   ~HGraphEncoderEngine() = default;

   /// Encode a state-only step into an existing builder.
   void encode_state(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state_impl(state, builder);
   }

   /// Encode a state with typed goals/actions into an existing builder.
   template < typename GoalTag >
   void encode_step(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      encode_step_impl(state, goals, actions, builder);
   }

   /// Encode with fully split goal inputs and explicit actions.
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      encode_impl(state, goals, actions, builder);
   }

   /// Encode with history-subgoal inputs.
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      const std::vector< HistorySubgoal >& history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder
   )
   {
      encode_impl_core(state, goals, actions, history_subgoals, history_max_steps, builder);
   }

   /// Encode state-only (goals inferred from problem if needed by caller).
   void encode(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state(state, builder);
   }

   /// Return effective engine config.
   const Config& get_config() const { return config_; }

  protected:
   /// Initialize relation dictionary and precomputed relation metadata from domain config.
   void initialize_from_domain();

   /// Internal typed goal/action encode implementation.
   template < typename GoalTag >
   void encode_step_impl(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   /// Internal state-only encode implementation.
   void encode_state_impl(const mimir::search::State& state, BatchBuilder& builder);

   /// Internal full encode implementation.
   void encode_impl(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   /// Internal encode with history support (shared implementation).
   void encode_impl_core(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder
   );

   /// Encode object/symbol node type and register node indices.
   void encode_objects(
      const mimir::search::State& state,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      std::span< const std::string > extra_objects = {}
   );

   /// Encode fact atoms and return formatted fact keys.
   hash_set< std::string > encode_facts(
      const mimir::search::State& state,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
      std::span< const std::string > extra_objects = {}
   );

   /// Encode goal literals as nodes and connect relation edges.
   template < typename GoalTag >
   void encode_literals(
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
      std::span< const std::string > extra_objects = {}
   );

   /// Encode grounded actions as nodes and connect relation edges.
   void encode_actions(
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
      std::span< const std::string > extra_objects = {}
   );

   /// Encode goal satisfaction derivations (sat/unsat variants).
   template < typename GoalTag >
   void encode_goal_satisfaction(
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
      const hash_set< std::string >& fact_keys,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
      std::string_view suffix = "",
      std::span< const std::string > extra_objects = {}
   );

   /// Add LGAN nearest-neighbor style edges where configured.
   void add_lgan_nn_edges(
      BatchBuilder& builder,
      const hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      const hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      const hash_map< std::string, hash_set< std::string > >& symbol_to_relations
   );

   /// Encode history subgoal nodes and edges.
   void encode_history(
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      hash_map< std::string, hash_set< std::string > >& symbol_to_relations
   );

   /// Ensure configured edge types exist in output even when empty.
   void ensure_empty_edge_types(BatchBuilder& builder) const;
   /// Ensure node feature dims are present for synthesized empty x tensors.
   void ensure_node_feature_dims(BatchBuilder& builder) const;

   /// Append one directed edge.
   static void append_edges(
      BatchBuilder& builder,
      const std::string& src_type,
      const std::string& rel_type,
      const std::string& dst_type,
      int64_t src,
      int64_t dst
   );

   /// Build stable relation-key string from node type and formatted node key.
   static std::string relation_key(const std::string& node_type, const std::string& node_key);

   /// Get/create one node index and keep names/index map aligned.
   static int64_t get_or_add_node(
      const std::string& node_type,
      const std::string& node_key,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names
   );

   /// Optional owning domain storage for handle-based construction.
   mimir::formalism::Domain domain_holder_;
   /// Active domain implementation reference.
   const mimir::formalism::DomainImpl& domain_;
   /// Effective runtime config.
   Config config_;
   /// Derived relation arity metadata.
   RelationDict relation_dict_;
   /// Precomputed edge types used when include_empty_edge_types is enabled.
   std::vector< std::tuple< std::string, std::string, std::string > > all_edge_types_;
};

/**
 * @brief Payload for one streaming HGraph encode step.
 */
struct HGraphStepInput {
   const mimir::search::State* state = nullptr;
   const GoalInputs* goals = nullptr;
   const std::vector< mimir::formalism::GroundAction >* actions = nullptr;
   const std::vector< HGraphEncoderEngine::HistorySubgoal >* history = nullptr;
   std::optional< int > history_max_steps;
};

/**
 * @brief Streaming HGraph encoder with static dispatch.
 */
class HGraphStreamEncoder: public StreamEncoderBase< HGraphStreamEncoder, HGraphStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "hetero"; }

   explicit HGraphStreamEncoder(HGraphEncoderEngine& engine) : engine_(&engine) { reset(); }

   int64_t append(const mimir::search::State& state)
   {
      HGraphStepInput step;
      step.state = &state;
      return StreamEncoderBase::append(step);
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      HGraphStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      return StreamEncoderBase::append(step);
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
      std::optional< int > history_max_steps
   )
   {
      HGraphStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      step.history = &history;
      step.history_max_steps = history_max_steps;
      return StreamEncoderBase::append(step);
   }

   void update(int64_t id, const mimir::search::State& state)
   {
      HGraphStepInput step;
      step.state = &state;
      StreamEncoderBase::update(id, step);
   }

   void update(
      int64_t id,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      HGraphStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      StreamEncoderBase::update(id, step);
   }

   void update(
      int64_t id,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
      std::optional< int > history_max_steps
   )
   {
      HGraphStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      step.history = &history;
      step.history_max_steps = history_max_steps;
      StreamEncoderBase::update(id, step);
   }

   void encode_step(const HGraphStepInput& step, BatchBuilder& builder)
   {
      if(engine_ == nullptr or step.state == nullptr) {
         throw std::invalid_argument("HGraphStreamEncoder requires a valid engine/state");
      }

      const auto& state = *step.state;
      if(step.goals == nullptr) {
         if(step.history != nullptr and not step.history->empty()) {
            throw std::invalid_argument("History encoding requires explicit GoalInputs");
         }
         engine_->encode(state, builder);
         return;
      }

      std::span< const mimir::formalism::GroundAction > actions_span;
      if(step.actions != nullptr) {
         actions_span = std::span< const mimir::formalism::GroundAction >(*step.actions);
      }

      if(step.history != nullptr and not step.history->empty()) {
         engine_->encode(
            state, *step.goals, actions_span, *step.history, step.history_max_steps, builder
         );
      } else {
         engine_->encode(state, *step.goals, actions_span, builder);
      }
   }

  private:
   HGraphEncoderEngine* engine_ = nullptr;
};

template < typename GoalTag >
void HGraphEncoderEngine::encode_step_impl(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   GoalInputs inputs;
   for(const auto& goal : goals) {
      if constexpr(std::is_same_v< GoalTag, mimir::formalism::StaticTag >) {
         inputs.static_goals.emplace_back(goal);
         inputs.static_goal_levels[goal] = 0;
      } else if constexpr(std::is_same_v< GoalTag, mimir::formalism::FluentTag >) {
         inputs.fluent_goals.emplace_back(goal);
         inputs.fluent_goal_levels[goal] = 0;
      } else {
         inputs.derived_goals.emplace_back(goal);
         inputs.derived_goal_levels[goal] = 0;
      }
   }
   encode_impl(state, inputs, actions, builder);
}

template < typename GoalTag >
void HGraphEncoderEngine::encode_literals(
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   for(const auto& literal : goals) {
      const auto atom = literal->get_atom();
      const auto predicate = atom->get_predicate();
      const std::optional< int > goal_level = goal_levels.contains(literal)
                                                 ? std::optional< int >(goal_levels.at(literal))
                                                 : std::nullopt;

      std::string node_type;
      std::string node_key;
      if(goal_level.has_value()) {
         const GoalLevel level(*goal_level);
         node_type = RelationFormatter::format_predicate(
            predicate, level, std::nullopt, literal->get_polarity()
         );
         node_key = RelationFormatter::format_literal< GoalTag >(literal, level);
      } else {
         node_type = RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, literal->get_polarity()
         );
         node_key = RelationFormatter::format_literal< GoalTag >(literal, std::nullopt);
      }

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         if(not config_.add_nullary_predicates) {
            continue;
         }
         object_keys.emplace_back(config_.nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(RelationFormatter::format_object(*obj));
         }
      }
      const auto relation_idx = get_or_add_node(
         node_type, node_key, builder, node_indices, node_names
      );

      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto& obj_key = object_keys[pos];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& obj_key = extra_objects[i];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(object_keys.size() + i);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      auto& symbols = relation_to_symbols[rel_key];
      for(const auto& obj_key : object_keys) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
      for(const auto& obj_key : extra_objects) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
   }
}

template < typename GoalTag >
void HGraphEncoderEngine::encode_goal_satisfaction(
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
   const hash_set< std::string >& fact_keys,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::string_view suffix,
   std::span< const std::string > extra_objects
)
{
   for(const auto& goal : goals) {
      const auto atom = goal->get_atom();
      const auto predicate = atom->get_predicate();
      const auto key = RelationFormatter::format_atom< GoalTag >(atom);
      const bool satisfied = fact_keys.contains(key) == goal->get_polarity();
      const GoalSatisfaction sat = satisfied ? GoalSatisfaction::satisfied
                                             : GoalSatisfaction::unsatisfied;
      if(not relation_dict_.goal_satisfaction_derivations.contains(sat)) {
         continue;
      }

      std::optional< int > goal_level = goal_levels.contains(goal)
                                           ? std::optional< int >(goal_levels.at(goal))
                                           : std::nullopt;

      std::string node_type;
      std::string node_key;
      if(goal_level.has_value()) {
         const GoalLevel level(*goal_level);
         node_type = RelationFormatter::format_predicate(
            predicate, level, sat, goal->get_polarity(), suffix
         );
         node_key = RelationFormatter::format_literal< GoalTag >(
            goal, level, sat, std::nullopt, suffix
         );
      } else {
         node_type = RelationFormatter::format_predicate(
            predicate, std::nullopt, sat, goal->get_polarity(), suffix
         );
         node_key = RelationFormatter::format_literal< GoalTag >(
            goal, std::nullopt, sat, std::nullopt, suffix
         );
      }

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         if(not config_.add_nullary_predicates) {
            continue;
         }
         object_keys.emplace_back(config_.nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(RelationFormatter::format_object(*obj));
         }
      }
      const auto relation_idx = get_or_add_node(
         node_type, node_key, builder, node_indices, node_names
      );

      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto& obj_key = object_keys[pos];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& obj_key = extra_objects[i];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(object_keys.size() + i);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      auto& symbols = relation_to_symbols[rel_key];
      for(const auto& obj_key : object_keys) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
      for(const auto& obj_key : extra_objects) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
   }
}

}  // namespace mifrost
