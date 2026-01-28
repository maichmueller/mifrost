#pragma once

#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "batch_builder.hpp"
#include "common_types.hpp"
#include "goal_inputs.hpp"
#include "relation_dict.hpp"
#include "relation_formatter.hpp"
#include "stream_encoder_base.hpp"

namespace mifrost {

class HGraphEncoderEngine: public StreamEncoderBase< HGraphEncoderEngine > {
  public:
   struct Config {
      std::string symbol_type_id = "symbol";
      std::string nullary_object_name = "null";
      std::string lgan_nn_edge_pos = "lgan_nn";
      int max_goal_level = 0;
      bool support_literals = true;
      bool add_nullary_predicates = true;
      bool ignore_actions = false;
      bool include_lgan_edges = true;
      bool include_static = true;
      std::set< GoalSatisfaction > goal_satisfaction_derivations = {
         GoalSatisfaction::True,
         GoalSatisfaction::False
      };
   };

   // domain stays alive for the duration of the object
   explicit HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
   HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);

   // domain enters domain holder
   explicit HGraphEncoderEngine(mimir::formalism::Domain domain);
   HGraphEncoderEngine(mimir::formalism::Domain domain, Config config);

   virtual ~HGraphEncoderEngine() = default;

   void encode_state(const mimir::search::State& state, BatchBuilder& builder) override
   {
      encode_state_impl(state, builder);
   }

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

   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      encode_impl(state, goals, actions, builder);
   }

   void encode(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state(state, builder);
   }

   const Config& get_config() const { return config_; }

  protected:
   friend class StreamEncoderBase< HGraphEncoderEngine >;

   void initialize_from_domain();

   template < typename GoalTag >
   void encode_step_impl(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   void encode_state_impl(const mimir::search::State& state, BatchBuilder& builder);

   virtual void encode_impl(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   virtual void encode_objects(
      const mimir::search::State& state,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      std::span< const std::string > extra_objects = {}
   );

   virtual hash_set< std::string > encode_facts(
      const mimir::search::State& state,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
      std::span< const std::string > extra_objects = {}
   );

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

   virtual void encode_actions(
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
      std::span< const std::string > extra_objects = {}
   );

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
      std::span< const std::string > extra_objects = {}
   );

   void add_lgan_nn_edges(
      BatchBuilder& builder,
      const hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      const hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
      const hash_map< std::string, hash_set< std::string > >& symbol_to_relations
   );

   void ensure_empty_edge_types(BatchBuilder& builder) const;
   void ensure_node_feature_dims(BatchBuilder& builder) const;

   static void append_edges(
      BatchBuilder& builder,
      const std::string& src_type,
      const std::string& rel_type,
      const std::string& dst_type,
      int64_t src,
      int64_t dst
   );

   static std::string relation_key(const std::string& node_type, const std::string& node_key);

   static int64_t get_or_add_node(
      const std::string& node_type,
      const std::string& node_key,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names
   );

   mimir::formalism::Domain domain_holder_;
   const mimir::formalism::DomainImpl& domain_;
   Config config_;
   RelationDict relation_dict_;
   std::vector< std::tuple< std::string, std::string, std::string > > all_edge_types_;
};

}  // namespace mifrost
