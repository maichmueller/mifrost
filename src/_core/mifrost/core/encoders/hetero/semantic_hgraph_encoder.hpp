/** Planner-neutral heterogeneous graph encoder. */
#pragma once

#include <boost/describe.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/goal_derivation.hpp"
#include "mifrost/core/encoders/common/target_source.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/views/semantic_preparation.hpp"

namespace mifrost {

class SemanticSuccessorHGraphEncoderEngine;
class SemanticHorizonHGraphEncoderEngine;
class SemanticTransitionDAG;
struct SemanticHorizonHGraphEncoderConfig;

/** Runtime policy for `SemanticHGraphEncoderEngine`. */
struct SemanticHGraphEncoderConfig {
   std::string symbol_type_id = defaults::symbol_type_id;
   std::string target_symbol_prefix = "target:";
   std::string nullary_object_name = "![nullary_symbol]!";
   std::string lgan_tn_edge_pos = defaults::lgan_tn_edge_pos;
   std::string lgan_nn_edge_pos = defaults::lgan_nn_edge_pos;
   std::string lgan_rr_edge_pos = defaults::lgan_rr_edge_pos;
   std::string history_link_relation = defaults::history_link_relation;
   size_t max_goal_level = 0;
   bool support_literals = false;
   bool add_nullary_predicates = false;
   bool ignore_actions = true;
   bool include_lgan_edges = false;
   bool include_static = true;
   bool include_empty_edge_types = true;
   bool export_node_names = true;
   bool allow_subgoal_layers_beyond_max_goal_level = false;
   std::set< TargetSource > lgan_anchor_sources = {};
   std::set< TargetSource > target_sources = {};
   std::set< GoalDerivation > goal_derivations = {
      GoalDerivation::plain,
      GoalDerivation::satisfied,
   };
};

BOOST_DESCRIBE_STRUCT(
   SemanticHGraphEncoderConfig,
   (),
   (symbol_type_id,
    target_symbol_prefix,
    nullary_object_name,
    lgan_tn_edge_pos,
    lgan_nn_edge_pos,
    lgan_rr_edge_pos,
    history_link_relation,
    max_goal_level,
    support_literals,
    add_nullary_predicates,
    ignore_actions,
    include_lgan_edges,
    include_static,
    include_empty_edge_types,
    export_node_names,
    allow_subgoal_layers_beyond_max_goal_level,
    lgan_anchor_sources,
    target_sources,
    goal_derivations)
)

/**
 * Encode `SemanticFlatRelationInput` as the legacy heterogeneous graph schema.
 *
 * Predicate, action, object, atom, and literal indices are local to the owned
 * semantic input and schema. No planning-library types cross this interface.
 */
class MIFROST_API SemanticHGraphEncoderEngine {
  public:
   using Config = SemanticHGraphEncoderConfig;

   SemanticHGraphEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticHGraphEncoderEngine(
      std::shared_ptr< const SemanticTaskContext > task_context,
      Config config = {}
   );
   SemanticHGraphEncoderEngine(const SemanticHGraphEncoderEngine&) = delete;
   SemanticHGraphEncoderEngine& operator=(const SemanticHGraphEncoderEngine&) = delete;
   SemanticHGraphEncoderEngine(SemanticHGraphEncoderEngine&&) noexcept;
   SemanticHGraphEncoderEngine& operator=(SemanticHGraphEncoderEngine&&) noexcept;
   ~SemanticHGraphEncoderEngine();

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationInput& input) const;
   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(const State& state, Actions&& actions) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding
   encode(const State& state, Goals&& goals, Actions&& actions) const;

   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions,
      views::HistoryRange History >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions,
      History&& history,
      std::optional< int64_t > history_max_steps = std::nullopt
   ) const;

   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;
   template < views::StateView State, views::GroundActionRange Actions >
   void encode(const State& state, Actions&& actions, BatchBuilder& builder) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   void encode(const State& state, Goals&& goals, Actions&& actions, BatchBuilder& builder) const;

   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions,
      views::HistoryRange History >
   void encode(
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions,
      History&& history,
      std::optional< int64_t > history_max_steps,
      BatchBuilder& builder
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
   ) const;

   [[nodiscard]] const std::shared_ptr< const SemanticTaskContext >& get_task_context() const;
   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;
   [[nodiscard]] const std::vector< SemanticActionSpec >& get_actions() const;
   [[nodiscard]] const std::map< std::string, int >& get_relation_arities() const;
   void update_relations(std::map< std::string, int > relation_arities);

  private:
   friend class SemanticSuccessorHGraphEncoderEngine;
   friend class SemanticHorizonHGraphEncoderEngine;

   void configure_horizon(const SemanticHorizonHGraphEncoderConfig& config);
   void encode_horizon(
      const SemanticTransitionDAG& dag,
      const SemanticHorizonHGraphEncoderConfig& config,
      BatchBuilder& builder
   ) const;

   void encode_successor(
      const SemanticFlatRelationInput& current,
      const SemanticFlatRelationInput& successor,
      bool delta_mode,
      std::string_view successor_suffix,
      bool include_successor_goal_satisfaction,
      BatchBuilder& builder
   ) const;
   void encode_successor(
      const canonical::detail::ViewPreparation& current,
      const canonical::detail::ViewPreparation& successor,
      bool delta_mode,
      std::string_view successor_suffix,
      bool include_successor_goal_satisfaction,
      BatchBuilder& builder
   ) const;

   void encode_view_preparation(
      const canonical::detail::ViewPreparation& input,
      BatchBuilder& builder
   ) const;

   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost

namespace mifrost {

template < views::StateView State, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding
SemanticHGraphEncoderEngine::encode(const State& state, Actions&& actions) const
{
   BatchBuilder builder;
   encode(state, std::forward< Actions >(actions), builder);
   builder.next_graph();
   return builder.build();
}

template < views::StateView State, views::GroundActionRange Actions >
void SemanticHGraphEncoderEngine::encode(
   const State& state,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_hgraph_view_preparation(
      get_task_context(), state, std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding
SemanticHGraphEncoderEngine::encode(const State& state, Goals&& goals, Actions&& actions) const
{
   BatchBuilder builder;
   encode(state, std::forward< Goals >(goals), std::forward< Actions >(actions), builder);
   builder.next_graph();
   return builder.build();
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
void SemanticHGraphEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_hgraph_view_preparation(
      get_task_context(), state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
BatchBuilder::BatchEncoding SemanticHGraphEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps
) const
{
   BatchBuilder builder;
   encode(
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions),
      std::forward< History >(history),
      history_max_steps,
      builder
   );
   builder.next_graph();
   return builder.build();
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
void SemanticHGraphEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_hgraph_view_preparation(
      get_task_context(),
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions),
      std::forward< History >(history),
      history_max_steps
   );
   encode_view_preparation(preparation, builder);
}

}  // namespace mifrost
