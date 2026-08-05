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
   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationSink& sink) const;
   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(const State& state, Actions&& actions) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding
   encode(const State& state, Goals&& goals, Actions&& actions) const;

   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
   ) const;
   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;
   void encode(const SemanticFlatRelationSink& sink, BatchBuilder& builder) const;
   template < views::StateView State, views::GroundActionRange Actions >
   void encode(const State& state, Actions&& actions, BatchBuilder& builder) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   void encode(const State& state, Goals&& goals, Actions&& actions, BatchBuilder& builder) const;

   [[nodiscard]] const std::shared_ptr< const SemanticTaskContext >& get_task_context() const
   {
      return task_context_;
   }
   [[nodiscard]] const SemanticColorEncoderConfig& get_config() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;

  private:
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
   encode(
      canonical::make_semantic_flat_relation_sink(
         get_task_context(), state, std::forward< Actions >(actions)
      ),
      builder
   );
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
   encode(
      canonical::make_semantic_flat_relation_sink(
         get_task_context(), state, std::forward< Goals >(goals), std::forward< Actions >(actions)
      ),
      builder
   );
}

}  // namespace mifrost
