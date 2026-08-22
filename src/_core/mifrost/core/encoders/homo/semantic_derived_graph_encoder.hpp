/** Planner-neutral derived-graph encoder for vanilla GNN consumption. */
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/views/concepts.hpp"
#include "mifrost/core/views/semantic_preparation.hpp"

namespace mifrost {

/** Node-universe policy for the derived-graph family. */
enum class DerivedNodeUniverse : int64_t {
   /// Objects plus one node per encoded grounded atom/literal/action (reified).
   objects_and_atoms = 0,
   /// Domain objects only; atom structure is folded into directed object-object edges.
   objects_only = 1,
};

/** How one grounded atom's argument tuple becomes directed edges. */
enum class DerivedAtomExpansion : int64_t {
   /// One fact node connected to every argument object at its position (strategy 1).
   star = 0,
   /// All ordered argument pairs among i < j positions (strategy 2).
   clique = 1,
   /// Directed path following argument order, position i -> position i + 1 (strategy 2).
   chain = 2,
   /// First argument broadcasts to every other argument, position 0 -> position j (strategy 2).
   star_first = 3,
};

struct SemanticDerivedGraphEncoderConfig {
   DerivedNodeUniverse node_universe = DerivedNodeUniverse::objects_and_atoms;
   DerivedAtomExpansion atom_expansion = DerivedAtomExpansion::star;
   bool include_reverse_edges = true;
   bool export_node_names = true;
   bool include_line_graph = false;
   int64_t line_graph_max_degree = 32;
};

class MIFROST_API SemanticDerivedGraphEncoderEngine {
  public:
   using Config = SemanticDerivedGraphEncoderConfig;

   SemanticDerivedGraphEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      SemanticDerivedGraphEncoderConfig config = {}
   );
   SemanticDerivedGraphEncoderEngine(
      std::shared_ptr< const SemanticSchemaContext > schema,
      SemanticDerivedGraphEncoderConfig config = {}
   );

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationInput& input) const;
   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;

   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Actions&& actions
   ) const;

   template < views::StateView State, views::GroundActionRange Actions >
   void encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Actions&& actions,
      BatchBuilder& builder
   ) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      Actions&& actions
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
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions
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

   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions,
      views::HistoryRange History >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions,
      History&& history,
      std::optional< int64_t > history_max_steps = std::nullopt
   ) const;

   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions,
      views::HistoryRange History >
   void encode(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions,
      History&& history,
      std::optional< int64_t > history_max_steps,
      BatchBuilder& builder
   ) const;

   /**
    * Prepare one direct-View graph without encoding it (full lanes).
    *
    * Mirrors the other families so generic stream/batch code can hold
    * preparations from different problems of one domain in a single batch.
    */
   template <
      views::StateView State,
      views::LiteralRange Goals,
      views::LiteralLayerRange SubgoalLayers,
      views::GroundActionRange Actions,
      views::HistoryRange History >
   [[nodiscard]] canonical::detail::ViewPreparation prepare(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Goals&& goals,
      SubgoalLayers&& subgoal_layers,
      Actions&& actions,
      History&& history,
      std::optional< int64_t > history_max_steps = std::nullopt
   ) const;

   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] canonical::detail::ViewPreparation prepare(
      const std::shared_ptr< const SemanticProblemContext >& problem_context,
      const State& state,
      Actions&& actions
   ) const;

   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
   ) const;

   /**
    * Encode a batch of already prepared direct-View graphs.
    *
    * Preparations may come from different problems of one domain; only their
    * schemas have to agree with this engine's.
    */
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const canonical::detail::ViewPreparation* const > preparations
   ) const;

   void encode_view_preparation(
      const canonical::detail::ViewPreparation& preparation,
      BatchBuilder& builder
   ) const;

   [[nodiscard]] const std::shared_ptr< const SemanticSchemaContext >& get_schema_context() const
   {
      return schema_context_;
   }
   [[nodiscard]] const Config& get_config() const { return config_; }
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;

  private:
   std::shared_ptr< const SemanticSchemaContext > schema_context_;
   SemanticDerivedGraphEncoderConfig config_;
};

}  // namespace mifrost

namespace mifrost {

template < views::StateView State, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding SemanticDerivedGraphEncoderEngine::encode(
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
void SemanticDerivedGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = prepare(problem_context, state, std::forward< Actions >(actions));
   encode_view_preparation(preparation, builder);
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding SemanticDerivedGraphEncoderEngine::encode(
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
void SemanticDerivedGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = prepare(
      problem_context, state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions >
BatchBuilder::BatchEncoding SemanticDerivedGraphEncoderEngine::encode(
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
void SemanticDerivedGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   auto preparation = canonical::detail::make_derived_view_preparation(
      problem_context,
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions)
   );
   encode_view_preparation(preparation, builder);
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
BatchBuilder::BatchEncoding SemanticDerivedGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
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
      problem_context,
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
void SemanticDerivedGraphEncoderEngine::encode(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps,
   BatchBuilder& builder
) const
{
   auto preparation = prepare(
      problem_context,
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions),
      std::forward< History >(history),
      history_max_steps
   );
   encode_view_preparation(preparation, builder);
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
canonical::detail::ViewPreparation SemanticDerivedGraphEncoderEngine::prepare(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps
) const
{
   return canonical::detail::make_derived_view_preparation(
      problem_context,
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions),
      std::forward< History >(history),
      history_max_steps
   );
}

template < views::StateView State, views::GroundActionRange Actions >
canonical::detail::ViewPreparation SemanticDerivedGraphEncoderEngine::prepare(
   const std::shared_ptr< const SemanticProblemContext >& problem_context,
   const State& state,
   Actions&& actions
) const
{
   return canonical::detail::make_derived_view_preparation(
      problem_context, state, std::forward< Actions >(actions)
   );
}

}  // namespace mifrost
