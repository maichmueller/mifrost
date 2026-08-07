/** Planner-neutral immediate-successor heterogeneous graph encoder. */
#pragma once

#include <boost/describe.hpp>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/views/semantic_preparation.hpp"

namespace mifrost {

enum class SemanticSuccessorMode {
   full,
   delta,
};

/** Runtime policy for `SemanticSuccessorHGraphEncoderEngine`. */
struct SemanticSuccessorHGraphEncoderConfig: SemanticHGraphEncoderConfig {
   SemanticSuccessorMode successor_mode = SemanticSuccessorMode::full;
   std::string successor_suffix = "[suc]";
   bool include_successor_goal_satisfaction = false;
};

BOOST_DESCRIBE_STRUCT(
   SemanticSuccessorHGraphEncoderConfig,
   (SemanticHGraphEncoderConfig),
   (successor_mode, successor_suffix, include_successor_goal_satisfaction)
)

/**
 * Encode aligned current/successor semantic state inputs.
 *
 * The current input supplies objects and every optional HGraph lane. The
 * successor input supplies the successor state facts; its object table must
 * exactly match the current input. No planning-library types cross this API.
 */
class MIFROST_API SemanticSuccessorHGraphEncoderEngine {
  public:
   using Config = SemanticSuccessorHGraphEncoderConfig;

   SemanticSuccessorHGraphEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticSuccessorHGraphEncoderEngine(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config config = {}
   );
   SemanticSuccessorHGraphEncoderEngine(const SemanticSuccessorHGraphEncoderEngine&) = delete;
   SemanticSuccessorHGraphEncoderEngine& operator=(const SemanticSuccessorHGraphEncoderEngine&) =
      delete;
   SemanticSuccessorHGraphEncoderEngine(SemanticSuccessorHGraphEncoderEngine&&) noexcept;
   SemanticSuccessorHGraphEncoderEngine& operator=(SemanticSuccessorHGraphEncoderEngine&&) noexcept;
   ~SemanticSuccessorHGraphEncoderEngine();

   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const SemanticFlatRelationInput& current,
      const SemanticFlatRelationInput& successor
   ) const;
   template <
      views::StateView CurrentState,
      views::GroundActionRange CurrentActions,
      views::StateView SuccessorState,
      views::GroundActionRange SuccessorActions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const CurrentState& current_state,
      CurrentActions&& current_actions,
      const SuccessorState& successor_state,
      SuccessorActions&& successor_actions
   ) const;
   void encode(
      const SemanticFlatRelationInput& current,
      const SemanticFlatRelationInput& successor,
      BatchBuilder& builder
   ) const;
   template <
      views::StateView CurrentState,
      views::GroundActionRange CurrentActions,
      views::StateView SuccessorState,
      views::GroundActionRange SuccessorActions >
   void encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const CurrentState& current_state,
      CurrentActions&& current_actions,
      const SuccessorState& successor_state,
      SuccessorActions&& successor_actions,
      BatchBuilder& builder
   ) const;
   template <
      views::StateView CurrentState,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange CurrentActions,
      views::StateView SuccessorState,
      views::GroundActionRange SuccessorActions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const CurrentState& current_state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      CurrentActions&& current_actions,
      const SuccessorState& successor_state,
      SuccessorActions&& successor_actions
   ) const;
   template <
      views::StateView CurrentState,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange CurrentActions,
      views::StateView SuccessorState,
      views::GroundActionRange SuccessorActions >
   void encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const CurrentState& current_state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      CurrentActions&& current_actions,
      const SuccessorState& successor_state,
      SuccessorActions&& successor_actions,
      BatchBuilder& builder
   ) const;
   /**
    * Prepare the two lanes of one transition without encoding it.
    *
    * The lanes are deliberately not symmetric: the successor side reads only
    * the object table and the successor state facts, so preparing it as a full
    * graph would materialize goals the algorithm never inspects.
    *
    * Both lanes of one transition take the same problem context -- a state and
    * its successor necessarily belong to one instance -- but two transitions in
    * the same batch may come from different problems.
    */
   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] canonical::detail::ViewPreparation prepare_current(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Actions&& actions
   ) const;

   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions >
   [[nodiscard]] canonical::detail::ViewPreparation prepare_current(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions
   ) const;

   template < views::StateView State >
   [[nodiscard]] canonical::detail::ViewPreparation prepare_successor(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state
   ) const;

   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& currents,
      const std::vector< SemanticFlatRelationInput >& successors
   ) const;
   /**
    * Encode a batch of already prepared aligned current/successor graphs.
    *
    * A `ViewPreparation` owns its compact pools and borrows nothing from the
    * backend state it was built from, so a caller may build one pair per
    * transition and hold them until the batch is flushed.
    */
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const canonical::detail::ViewPreparation* const > currents,
      std::span< const canonical::detail::ViewPreparation* const > successors
   ) const;

   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::shared_ptr< const SemanticSchemaContext >& get_schema_context() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;
   [[nodiscard]] const std::vector< SemanticActionSpec >& get_actions() const;
   [[nodiscard]] const std::map< std::string, int >& get_relation_arities() const;
   void update_relations(std::map< std::string, int > relation_arities);

  private:
   void encode_views(
      const canonical::detail::ViewPreparation& current,
      const canonical::detail::ViewPreparation& successor,
      BatchBuilder& builder
   ) const;

   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost

namespace mifrost {

template < views::StateView State, views::GroundActionRange Actions >
canonical::detail::ViewPreparation SemanticSuccessorHGraphEncoderEngine::prepare_current(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Actions&& actions
) const
{
   return canonical::detail::make_hgraph_view_preparation(
      problem_context, state, std::forward< Actions >(actions)
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions >
canonical::detail::ViewPreparation SemanticSuccessorHGraphEncoderEngine::prepare_current(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions
) const
{
   return canonical::detail::make_hgraph_view_preparation(
      problem_context,
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions)
   );
}

template < views::StateView State >
canonical::detail::ViewPreparation SemanticSuccessorHGraphEncoderEngine::prepare_successor(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state
) const
{
   return canonical::detail::make_state_only_view_preparation(problem_context, state);
}

template <
   views::StateView CurrentState,
   views::GroundActionRange CurrentActions,
   views::StateView SuccessorState,
   views::GroundActionRange SuccessorActions >
BatchBuilder::BatchEncoding SemanticSuccessorHGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const CurrentState& current_state,
   CurrentActions&& current_actions,
   const SuccessorState& successor_state,
   SuccessorActions&& successor_actions
) const
{
   BatchBuilder builder;
   encode(
      problem_context,
      current_state,
      std::forward< CurrentActions >(current_actions),
      successor_state,
      std::forward< SuccessorActions >(successor_actions),
      builder
   );
   builder.next_graph();
   return builder.build();
}

template <
   views::StateView CurrentState,
   views::GroundActionRange CurrentActions,
   views::StateView SuccessorState,
   views::GroundActionRange SuccessorActions >
void SemanticSuccessorHGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const CurrentState& current_state,
   CurrentActions&& current_actions,
   const SuccessorState& successor_state,
   SuccessorActions&& successor_actions,
   BatchBuilder& builder
) const
{
   // Accepted for signature symmetry with the current-state lane, but the
   // successor-graph algorithm never encodes successor actions.
   (void) successor_actions;
   encode_views(
      canonical::detail::make_hgraph_view_preparation(
         problem_context, current_state, std::forward< CurrentActions >(current_actions)
      ),
      // The successor side reads only the object table and the successor state
      // facts; goals, subgoal layers, actions and history are never inspected
      // there. Prepare only what the algorithm consumes instead of
      // materializing the problem context's default goals to discard them.
      canonical::detail::make_state_only_view_preparation(problem_context, successor_state),
      builder
   );
}

template <
   views::StateView CurrentState,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange CurrentActions,
   views::StateView SuccessorState,
   views::GroundActionRange SuccessorActions >
BatchBuilder::BatchEncoding SemanticSuccessorHGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const CurrentState& current_state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   CurrentActions&& current_actions,
   const SuccessorState& successor_state,
   SuccessorActions&& successor_actions
) const
{
   BatchBuilder builder;
   encode(
      problem_context,
      current_state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< CurrentActions >(current_actions),
      successor_state,
      std::forward< SuccessorActions >(successor_actions),
      builder
   );
   builder.next_graph();
   return builder.build();
}

template <
   views::StateView CurrentState,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange CurrentActions,
   views::StateView SuccessorState,
   views::GroundActionRange SuccessorActions >
void SemanticSuccessorHGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const CurrentState& current_state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   CurrentActions&& current_actions,
   const SuccessorState& successor_state,
   SuccessorActions&& successor_actions,
   BatchBuilder& builder
) const
{
   // Accepted for signature symmetry with the current-state lane, but the
   // successor-graph algorithm never encodes successor actions.
   (void) successor_actions;
   encode_views(
      canonical::detail::make_hgraph_view_preparation(
         problem_context,
         current_state,
         std::forward< Goals >(goals),
         std::forward< SubgoalLayers >(subgoal_layers),
         std::forward< CurrentActions >(current_actions)
      ),
      // The successor side reads only the object table and the successor state
      // facts; goals, subgoal layers, actions and history are never inspected
      // there. Prepare only what the algorithm consumes instead of
      // materializing the problem context's default goals to discard them.
      canonical::detail::make_state_only_view_preparation(problem_context, successor_state),
      builder
   );
}

}  // namespace mifrost
