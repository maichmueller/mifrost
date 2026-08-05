/** Planner-neutral immediate-successor heterogeneous graph encoder. */
#pragma once

#include <boost/describe.hpp>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"

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
      std::shared_ptr< const SemanticTaskContext > task_context,
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
      const CurrentState& current_state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      CurrentActions&& current_actions,
      const SuccessorState& successor_state,
      SuccessorActions&& successor_actions,
      BatchBuilder& builder
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& currents,
      const std::vector< SemanticFlatRelationInput >& successors
   ) const;

   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::shared_ptr< const SemanticTaskContext >& get_task_context() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;
   [[nodiscard]] const std::vector< SemanticActionSpec >& get_actions() const;
   [[nodiscard]] const std::map< std::string, int >& get_relation_arities() const;
   void update_relations(std::map< std::string, int > relation_arities);

  private:
   void encode_views(
      const canonical::detail::GraphInput& current,
      const canonical::detail::GraphInput& successor,
      BatchBuilder& builder
   ) const;

   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost

namespace mifrost {

template <
   views::StateView CurrentState,
   views::GroundActionRange CurrentActions,
   views::StateView SuccessorState,
   views::GroundActionRange SuccessorActions >
BatchBuilder::BatchEncoding SemanticSuccessorHGraphEncoderEngine::encode(
   const CurrentState& current_state,
   CurrentActions&& current_actions,
   const SuccessorState& successor_state,
   SuccessorActions&& successor_actions
) const
{
   BatchBuilder builder;
   encode(
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
   const CurrentState& current_state,
   CurrentActions&& current_actions,
   const SuccessorState& successor_state,
   SuccessorActions&& successor_actions,
   BatchBuilder& builder
) const
{
   encode_views(
      canonical::detail::make_graph_input(
         get_task_context(), current_state, std::forward< CurrentActions >(current_actions)
      ),
      canonical::detail::make_graph_input(
         get_task_context(), successor_state, std::forward< SuccessorActions >(successor_actions)
      ),
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
   const CurrentState& current_state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   CurrentActions&& current_actions,
   const SuccessorState& successor_state,
   SuccessorActions&& successor_actions,
   BatchBuilder& builder
) const
{
   encode_views(
      canonical::detail::make_graph_input(
         get_task_context(),
         current_state,
         std::forward< Goals >(goals),
         std::forward< SubgoalLayers >(subgoal_layers),
         std::forward< CurrentActions >(current_actions)
      ),
      canonical::detail::make_graph_input(
         get_task_context(), successor_state, std::forward< SuccessorActions >(successor_actions)
      ),
      builder
   );
}

}  // namespace mifrost
