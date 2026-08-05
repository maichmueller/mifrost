/** Planner-neutral homogeneous color encoder. */
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

namespace mifrost {

struct SemanticColorEncoderConfig {
   bool edge_features = false;
   bool enable_global_predicate_nodes = false;
   bool export_node_names = true;
};

namespace detail {

/** Encoder-local preparation for direct semantic View encoding. */
struct ColorViewPreparation {
   std::shared_ptr< const SemanticTaskContext > task_context;
   std::vector< SemanticAtom > state_facts;
   std::vector< SemanticLiteral > goals;
   bool use_default_goals = false;
   std::vector< SemanticGroundAction > actions;
   std::vector< std::vector< SemanticLiteral > > subgoal_layers;
   std::vector< SemanticHistoryEntry > history;
   std::optional< int64_t > history_max_steps = std::nullopt;
};

[[nodiscard]] inline const std::vector< std::string >& semantic_objects(
   const ColorViewPreparation& input
)
{
   if(not input.task_context) {
      throw std::invalid_argument("semantic View preparation requires a task context");
   }
   return input.task_context->objects;
}

[[nodiscard]] inline const std::vector< SemanticLiteral >& semantic_goals(
   const ColorViewPreparation& input
)
{
   if(input.task_context and input.use_default_goals) {
      return input.task_context->default_goals;
   }
   return input.goals;
}

[[nodiscard]] inline const std::vector< SemanticAtom >& semantic_static_facts(
   const ColorViewPreparation& input
)
{
   static const std::vector< SemanticAtom > empty;
   return input.task_context ? input.task_context->static_facts : empty;
}

}  // namespace detail

class MIFROST_API SemanticColorEncoderEngine {
  public:
   SemanticColorEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      SemanticColorEncoderConfig config = {}
   );
   SemanticColorEncoderEngine(
      std::shared_ptr< const SemanticTaskContext > task_context,
      SemanticColorEncoderConfig config = {}
   );

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
      views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions
   ) const;

   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
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
      views::GroundActionRange Actions >
   void encode(
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions,
      BatchBuilder& builder
   ) const;

   [[nodiscard]] const std::shared_ptr< const SemanticTaskContext >& get_task_context() const
   {
      return task_context_;
   }
   [[nodiscard]] const SemanticColorEncoderConfig& get_config() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;

  private:
   void
   encode_view_preparation(const detail::ColorViewPreparation& input, BatchBuilder& builder) const;

   std::shared_ptr< const SemanticTaskContext > task_context_;
   const std::vector< SemanticPredicateSpec >& predicates_;
   SemanticColorEncoderConfig config_;
};

}  // namespace mifrost

namespace mifrost {

template < views::StateView State, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding
SemanticColorEncoderEngine::encode(const State& state, Actions&& actions) const
{
   BatchBuilder builder;
   encode(state, std::forward< Actions >(actions), builder);
   builder.next_graph();
   return builder.build();
}

template < views::StateView State, views::GroundActionRange Actions >
void SemanticColorEncoderEngine::encode(
   const State& state,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_preparation< detail::ColorViewPreparation >(
      get_task_context(), state, std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding
SemanticColorEncoderEngine::encode(const State& state, Goals&& goals, Actions&& actions) const
{
   BatchBuilder builder;
   encode(state, std::forward< Goals >(goals), std::forward< Actions >(actions), builder);
   builder.next_graph();
   return builder.build();
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
void SemanticColorEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_preparation< detail::ColorViewPreparation >(
      get_task_context(), state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions >
BatchBuilder::BatchEncoding SemanticColorEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions
) const
{
   BatchBuilder builder;
   encode(
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions),
      builder
   );
   builder.next_graph();
   return builder.build();
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions >
void SemanticColorEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_preparation< detail::ColorViewPreparation >(
      get_task_context(),
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

}  // namespace mifrost
