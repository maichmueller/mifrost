#pragma once

#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <set>
#include <span>
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
class HGraphEncoderEngine: public StreamEncoderBase< HGraphEncoderEngine > {
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

   virtual ~HGraphEncoderEngine() = default;

   /// StreamEncoderInterface entrypoint.
   void encode_state(const mimir::search::State& state, BatchBuilder& builder) override
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
   friend class StreamEncoderBase< HGraphEncoderEngine >;

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

   /// Internal full encode implementation (overridable by specialized encoders).
   virtual void encode_impl(
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
   virtual void encode_objects(
      const mimir::search::State& state,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      std::span< const std::string > extra_objects = {}
   );

   /// Encode fact atoms and return formatted fact keys.
   virtual hash_set< std::string > encode_facts(
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
   virtual void encode_actions(
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

}  // namespace mifrost
