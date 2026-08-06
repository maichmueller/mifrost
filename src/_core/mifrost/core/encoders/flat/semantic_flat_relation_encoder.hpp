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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "flat_relation_config.hpp"
#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/semantic/records.hpp"
#include "mifrost/core/views/concepts.hpp"

namespace mifrost {

class SemanticFlatHorizonEncoderEngine;
class SemanticTransitionDAG;
struct SemanticFlatHorizonEncoderConfig;
namespace canonical::detail {
struct ViewPreparation;
}

/*
 * The semantic record and key definitions (SemanticAtom, SemanticLiteral,
 * SemanticGroundAction, SemanticHistoryEntry, SemanticTaskContext,
 * SemanticFlatRelationInput and their hash/ordering helpers) live in
 * core/semantic/records.hpp so that the View layer can describe borrowed inputs
 * without including this encoder header. They are re-exported here unchanged.
 */

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
      std::shared_ptr< const SemanticTaskContext > task_context,
      Config config = {}
   );
   SemanticFlatRelationEncoderEngine(const SemanticFlatRelationEncoderEngine&) = delete;
   SemanticFlatRelationEncoderEngine& operator=(const SemanticFlatRelationEncoderEngine&) = delete;
   SemanticFlatRelationEncoderEngine(SemanticFlatRelationEncoderEngine&&) noexcept;
   SemanticFlatRelationEncoderEngine& operator=(SemanticFlatRelationEncoderEngine&&) noexcept;
   ~SemanticFlatRelationEncoderEngine();

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationInput& input) const;
   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;

   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(const State& state, Actions&& actions) const;

   template < views::StateView State, views::GroundActionRange Actions >
   void encode(const State& state, Actions&& actions, BatchBuilder& builder) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding
   encode(const State& state, Goals&& goals, Actions&& actions) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   void encode(const State& state, Goals&& goals, Actions&& actions, BatchBuilder& builder) const;

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
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::shared_ptr< const SemanticTaskContext >& get_task_context() const;
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

   void configure_horizon(const SemanticFlatHorizonEncoderConfig& config);
   void encode_horizon(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& config,
      BatchBuilder& builder
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_horizon_composed(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& config
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_horizon_composed_batch(
      const std::vector< SemanticTransitionDAG >& dags,
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
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost

#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
