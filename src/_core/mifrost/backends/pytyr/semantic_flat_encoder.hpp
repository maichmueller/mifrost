/**
 * @file semantic_flat_encoder.hpp
 * @brief Native PyTyr task converter for backend-neutral semantic encoders.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <tyr/formalism/planning/ground_action_view.hpp>
#include <tyr/formalism/planning/planning_task.hpp>
#include <tyr/planning/ground/state_view.hpp>
#include <tyr/planning/lifted/state_view.hpp>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

namespace mifrost::pytyr {

/**
 * Convert one Tyr planning task and its states/actions directly into compact
 * semantic records. The copied planning task keeps its repository-backed views
 * and compact Tyr-index lookup tables valid for the adapter lifetime. It owns
 * no family-specific neutral engine.
 */
class MIFROST_API SemanticPlanningTaskAdapter {
  public:
   explicit SemanticPlanningTaskAdapter(const tyr::formalism::planning::PlanningTask& task);
   SemanticPlanningTaskAdapter(const SemanticPlanningTaskAdapter&) = delete;
   SemanticPlanningTaskAdapter& operator=(const SemanticPlanningTaskAdapter&) = delete;
   SemanticPlanningTaskAdapter(SemanticPlanningTaskAdapter&&) noexcept;
   SemanticPlanningTaskAdapter& operator=(SemanticPlanningTaskAdapter&&) noexcept;
   ~SemanticPlanningTaskAdapter();

   [[nodiscard]] SemanticFlatRelationInput make_input(
      const tyr::planning::StateView< tyr::planning::LiftedTag >& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions = {}
   ) const;
   [[nodiscard]] SemanticFlatRelationInput make_input(
      const tyr::planning::StateView< tyr::planning::GroundTag >& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions = {}
   ) const;
   [[nodiscard]] SemanticLiteral make_raw_literal(
      int64_t category,
      int64_t predicate_index,
      const std::vector< int64_t >& object_indices,
      bool positive
   ) const;

   [[nodiscard]] std::shared_ptr< const SemanticTaskContext > get_task_context() const;

  private:
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost::pytyr
