/** Planner-neutral homogeneous color encoder. */
#pragma once

#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/views/semantic_preparation.hpp"

namespace mifrost {

struct SemanticColorEncoderConfig {
   bool edge_features = false;
   bool enable_global_predicate_nodes = false;
   bool export_node_names = true;
};

class MIFROST_API SemanticColorEncoderEngine {
  public:
   /// Matches the other family engines, so generic code can name the policy type.
   using Config = SemanticColorEncoderConfig;

   SemanticColorEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      SemanticColorEncoderConfig config = {}
   );
   SemanticColorEncoderEngine(
      std::shared_ptr< const SemanticSchemaContext > schema,
      SemanticColorEncoderConfig config = {}
   );

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationInput& input) const;
   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Actions&& actions
   ) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      Actions&& actions
   ) const;

   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions
   ) const;

   /**
    * Prepare one direct-View graph without encoding it.
    *
    * The family owns which preparation its algorithm consumes; a caller that
    * needs to hold graphs (a batch, a stream) asks the engine rather than
    * picking a preparation helper itself. Color has no history lane.
    *
    * The problem the graph belongs to is passed in per call, not held by the
    * engine: preparations from different problems of this domain may be mixed
    * freely in one `encode_batch`.
    */
   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] canonical::detail::ViewPreparation prepare(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Actions&& actions
   ) const;

   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions >
   [[nodiscard]] canonical::detail::ViewPreparation prepare(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions
   ) const;

   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
   ) const;
   /**
    * Encode a batch of already prepared direct-View graphs.
    *
    * A `ViewPreparation` owns its compact pools and borrows nothing from the
    * backend state it was built from, so a caller may build one per state and
    * hold them until the batch is flushed. The preparations need not come from
    * one problem: each is closed with `BatchBuilder::next_graph()` and carries
    * its own object table, so only their schemas have to agree with this
    * engine's.
    */
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const canonical::detail::ViewPreparation* const > preparations
   ) const;
   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;
   template < views::StateView State, views::GroundActionRange Actions >
   void encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Actions&& actions,
      BatchBuilder& builder
   ) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   void encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      Actions&& actions,
      BatchBuilder& builder
   ) const;

   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions >
   void encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions,
      BatchBuilder& builder
   ) const;

   [[nodiscard]] const std::shared_ptr< const SemanticSchemaContext >& get_schema_context() const
   {
      return schema_context_;
   }
   [[nodiscard]] const SemanticColorEncoderConfig& get_config() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;

  private:
   void encode_view_preparation(
      const canonical::detail::ViewPreparation& input,
      BatchBuilder& builder
   ) const;

   std::shared_ptr< const SemanticSchemaContext > schema_context_;
   const std::vector< SemanticPredicateSpec >& predicates_;
   SemanticColorEncoderConfig config_;
};

}  // namespace mifrost

namespace mifrost {

template < views::StateView State, views::GroundActionRange Actions >
canonical::detail::ViewPreparation SemanticColorEncoderEngine::prepare(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Actions&& actions
) const
{
   return canonical::detail::make_color_view_preparation(
      problem_context, state, std::forward< Actions >(actions)
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions >
canonical::detail::ViewPreparation SemanticColorEncoderEngine::prepare(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions
) const
{
   return canonical::detail::make_color_view_preparation(
      problem_context,
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions)
   );
}

template < views::StateView State, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding SemanticColorEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Actions&& actions
) const
{
   BatchBuilder builder;
   encode(problem_context, state, std::forward< Actions >(actions), builder);
   builder.next_graph();
   return builder.build();
}

template < views::StateView State, views::GroundActionRange Actions >
void SemanticColorEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_color_view_preparation(
      problem_context, state, std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding SemanticColorEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   Actions&& actions
) const
{
   BatchBuilder builder;
   encode(
      problem_context,
      state,
      std::forward< Goals >(goals),
      std::forward< Actions >(actions),
      builder
   );
   builder.next_graph();
   return builder.build();
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
void SemanticColorEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_color_view_preparation(
      problem_context, state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions >
BatchBuilder::BatchEncoding SemanticColorEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions
) const
{
   BatchBuilder builder;
   encode(
      problem_context,
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
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_color_view_preparation(
      problem_context,
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

}  // namespace mifrost
