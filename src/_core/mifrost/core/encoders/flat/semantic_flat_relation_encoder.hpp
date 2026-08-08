/**
 * @file semantic_flat_relation_encoder.hpp
 * @brief Planning-backend-neutral native flat relation encoder.
 */
#pragma once

#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <compare>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "flat_relation_config.hpp"
#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/semantic_assembly.hpp"
#include "mifrost/core/semantic/records.hpp"
#include "mifrost/core/views/concepts.hpp"

namespace mifrost {

class SemanticFlatHorizonEncoderEngine;
class SemanticFlatHorizonInput;
class SemanticFlatRelationAssemblyBuilder;
class SemanticTransitionDAG;
struct SemanticFlatHorizonEncoderConfig;
class FlatEmitterComponent;
namespace canonical::detail {
struct ViewPreparation;
}
namespace detail {
struct SemanticFlatHorizonRelationEngineAccess;
struct SemanticFlatRelationPreparedGraphData;
struct SemanticFlatRelationPreparedGraphAccess;
}  // namespace detail

/*
 * The semantic record and key definitions (SemanticAtom, SemanticLiteral,
 * SemanticGroundAction, SemanticHistoryEntry, SemanticSchemaContext, SemanticProblemContext,
 * SemanticFlatRelationInput and their hash/ordering helpers) live in
 * core/semantic/records.hpp so that the View layer can describe borrowed inputs
 * without including this encoder header. They are re-exported here unchanged.
 */

/** Compatibility DTO carrier with backend-neutral graph-local annotations. */
class MIFROST_API SemanticFlatRelationGraphInput:
    public SemanticEncoderInput< SemanticFlatRelationInput > {
  public:
   using Base = SemanticEncoderInput< SemanticFlatRelationInput >;

   explicit SemanticFlatRelationGraphInput(
      const SemanticFlatRelationInput& input,
      SemanticAnnotations annotations = {}
   );
   SemanticFlatRelationGraphInput(SemanticFlatRelationInput&&, SemanticAnnotations = {}) = delete;
   SemanticFlatRelationGraphInput(const SemanticFlatRelationInput&&, SemanticAnnotations = {}) =
      delete;
   explicit SemanticFlatRelationGraphInput(
      std::shared_ptr< const SemanticFlatRelationInput > input,
      SemanticAnnotations annotations = {}
   );

   [[nodiscard]] const SemanticFlatRelationInput& input() const { return source(); }
};

/**
 * Stable read-only semantic relation view shared by canonical and extensions.
 *
 * Accessors borrow the engine's one compact graph preparation and are valid for
 * the duration of the current component callback. Components must not retain
 * this view or any returned span after the callback returns.
 */
class MIFROST_API SemanticFlatRelationPreparedGraph {
  public:
   SemanticFlatRelationPreparedGraph(const SemanticFlatRelationPreparedGraph&) = delete;
   SemanticFlatRelationPreparedGraph& operator=(const SemanticFlatRelationPreparedGraph&) = delete;
   SemanticFlatRelationPreparedGraph(SemanticFlatRelationPreparedGraph&&) noexcept = default;
   SemanticFlatRelationPreparedGraph& operator=(SemanticFlatRelationPreparedGraph&&) noexcept =
      default;
   ~SemanticFlatRelationPreparedGraph();

   [[nodiscard]] const FlatRelationEncoderConfig& config() const;
   [[nodiscard]] const SemanticAnnotations& annotations() const;
   [[nodiscard]] std::span< const std::string > objects() const;
   [[nodiscard]] std::span< const SemanticAtom > static_facts() const;
   [[nodiscard]] std::span< const SemanticAtom > state_facts() const;
   [[nodiscard]] std::span< const SemanticGoalLevel > goal_levels() const;
   [[nodiscard]] const std::shared_ptr< const SemanticProblemContext >& problem_context() const;
   [[nodiscard]] bool contains_fact(const SemanticAtom& atom) const;
   [[nodiscard]] int64_t entity_count() const;
   [[nodiscard]] std::span< const std::string > entity_names() const;
   [[nodiscard]] std::span< const int64_t > entity_role_ids() const;
   [[nodiscard]] std::span< const int64_t > object_entity_indices() const;
   [[nodiscard]] std::span< const int64_t > predicate_entity_indices() const;
   [[nodiscard]] std::span< const SemanticGroundAction > action_identities() const;
   [[nodiscard]] std::span< const int64_t > history_entity_indices() const;
   [[nodiscard]] std::span< const int64_t > history_steps() const;
   [[nodiscard]] std::span< const int64_t > target_entity_indices() const;
   [[nodiscard]] std::span< const int64_t > target_entity_group_ids() const;
   [[nodiscard]] std::span< const int64_t > target_positions() const;
   [[nodiscard]] std::span< const int64_t > target_indices() const;
   [[nodiscard]] std::span< const int64_t > target_candidate_ids() const;
   [[nodiscard]] std::span< const int64_t > target_depths() const;
   [[nodiscard]] std::span< const int64_t > target_group_ids() const;
   [[nodiscard]] std::span< const std::string > target_names() const;

  private:
   friend struct detail::SemanticFlatRelationPreparedGraphAccess;
   explicit SemanticFlatRelationPreparedGraph(
      std::shared_ptr< const detail::SemanticFlatRelationPreparedGraphData > data
   );
   std::shared_ptr< const detail::SemanticFlatRelationPreparedGraphData > data_;
};

/**
 * @brief Encode owned semantic values directly to native BatchEncoding.
 *
 * The engine owns only predicate/action schema and encoding policy. It never
 * accepts or retains a Mimir, Tyr, or other planning-repository view.
 */
class MIFROST_API SemanticFlatRelationEncoderEngine {
  public:
   using Config = FlatRelationEncoderConfig;

   SemanticFlatRelationEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticFlatRelationEncoderEngine(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config config = {}
   );
   SemanticFlatRelationEncoderEngine(const SemanticFlatRelationEncoderEngine&) = delete;
   SemanticFlatRelationEncoderEngine& operator=(const SemanticFlatRelationEncoderEngine&) = delete;
   SemanticFlatRelationEncoderEngine(SemanticFlatRelationEncoderEngine&&) noexcept;
   SemanticFlatRelationEncoderEngine& operator=(SemanticFlatRelationEncoderEngine&&) noexcept;
   ~SemanticFlatRelationEncoderEngine();

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationInput& input) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const SemanticFlatRelationGraphInput& input
   ) const;
   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;
   void encode(const SemanticFlatRelationGraphInput& input, BatchBuilder& builder) const;

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
    * Prepare one direct-View graph without encoding it.
    *
    * The family owns which preparation its algorithm consumes; a caller that
    * needs to hold graphs (a batch, a stream) asks the engine rather than
    * picking a preparation helper itself.
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

   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const SemanticFlatRelationGraphInput > inputs
   ) const;

   /**
    * Encode a batch of already prepared direct-View graphs.
    *
    * A `ViewPreparation` owns its compact pools and borrows nothing from the
    * backend state it was built from, so it is the natural unit for a batch or
    * a stream: the caller may build one per state and hold them until flush.
    *
    * This is not a loop over the single-graph append: the flat batch has a
    * genuine cross-graph pass (target-name suppression and shared batch
    * constants), so appending graph by graph would silently diverge from
    * `encode_batch(inputs)`.
    *
    * The preparations need not come from one problem: each is closed with
    * `BatchBuilder::next_graph()` and carries its own object table, so only
    * their schemas have to agree with this engine's.
    */
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const canonical::detail::ViewPreparation* const > preparations
   ) const;
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::shared_ptr< const SemanticSchemaContext >& get_schema_context() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;
   [[nodiscard]] const std::vector< SemanticActionSpec >& get_actions() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_names() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_arities() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_sources() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_logical_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_encoded_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_roles() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_role_offsets() const;
   [[nodiscard]] const std::vector< std::string >& get_slot_role_names() const;

  private:
   friend class SemanticFlatHorizonEncoderEngine;
   friend class SemanticFlatRelationAssemblyBuilder;
   friend struct detail::SemanticFlatHorizonRelationEngineAccess;

   SemanticFlatRelationEncoderEngine(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config config,
      bool compile_relation_plan
   );
   SemanticFlatRelationEncoderEngine(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config config,
      std::vector< std::shared_ptr< FlatEmitterComponent > > components
   );

   void configure_horizon(
      const SemanticFlatHorizonEncoderConfig& config,
      std::vector< std::shared_ptr< FlatEmitterComponent > > components = {}
   );
   void encode_horizon(
      const SemanticFlatHorizonInput& input,
      const SemanticFlatHorizonEncoderConfig& config,
      BatchBuilder& builder
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_horizon_composed(
      const SemanticFlatHorizonInput& input,
      const SemanticFlatHorizonEncoderConfig& config
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_horizon_composed_batch(
      std::span< const SemanticFlatHorizonInput > inputs,
      const SemanticFlatHorizonEncoderConfig& config
   ) const;
   void finalize_horizon_encoding(
      BatchBuilder::BatchEncoding& encoding,
      const SemanticFlatHorizonEncoderConfig& config
   ) const;

   void encode_view_preparation(
      const canonical::detail::ViewPreparation& input,
      BatchBuilder& builder
   ) const;

   struct Impl;
   [[nodiscard]] Impl* impl() noexcept;
   [[nodiscard]] const Impl* impl() const noexcept;
   // Keep the engine's established one-pointer object layout while components
   // receive weak ownership of the shared implementation.
   std::unique_ptr< std::shared_ptr< Impl > > impl_;
};

/** Mutable canonical semantic relation assembly; `compile()` freezes it. */
class MIFROST_API SemanticFlatRelationAssemblyBuilder {
  public:
   using Config = FlatRelationEncoderConfig;

   SemanticFlatRelationAssemblyBuilder(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticFlatRelationAssemblyBuilder(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config config = {}
   );
   SemanticFlatRelationAssemblyBuilder(const SemanticFlatRelationAssemblyBuilder&) = delete;
   SemanticFlatRelationAssemblyBuilder& operator=(const SemanticFlatRelationAssemblyBuilder&) =
      delete;
   SemanticFlatRelationAssemblyBuilder(SemanticFlatRelationAssemblyBuilder&&) noexcept;
   SemanticFlatRelationAssemblyBuilder& operator=(SemanticFlatRelationAssemblyBuilder&&) noexcept;
   ~SemanticFlatRelationAssemblyBuilder();

   void add_component(std::unique_ptr< FlatEmitterComponent > component);

   template < typename Component, typename... Args >
   void emplace_component(Args&&... args)
   {
      add_component(std::make_unique< Component >(std::forward< Args >(args)...));
   }

   [[nodiscard]] SemanticFlatRelationEncoderEngine compile() &&;

  private:
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

namespace detail {
struct SemanticFlatHorizonRelationEngineAccess {
   [[nodiscard]] static SemanticFlatRelationEncoderEngine make(
      std::shared_ptr< const SemanticSchemaContext > schema,
      SemanticFlatRelationEncoderEngine::Config config
   );
};
}  // namespace detail

}  // namespace mifrost

#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
