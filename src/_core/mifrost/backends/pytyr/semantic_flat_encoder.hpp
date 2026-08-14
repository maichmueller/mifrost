/**
 * @file semantic_flat_encoder.hpp
 * @brief Native PyTyr task converter for backend-neutral semantic encoders.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <ranges>
#include <tyr/formalism/planning/ground_action_view.hpp>
#include <tyr/formalism/planning/planning_task.hpp>
#include <tyr/planning/ground/state_view.hpp>
#include <tyr/planning/lifted/state_view.hpp>
#include <vector>

#include "mifrost/backends/pytyr/views.hpp"
#include "mifrost/core/api.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/views/ranges.hpp"

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
      const tyr::planning::StateView< tyr::LiftedTag >& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions = {}
   ) const;
   [[nodiscard]] SemanticFlatRelationInput make_input(
      const tyr::planning::StateView< tyr::GroundTag >& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions = {}
   ) const;
   [[nodiscard]] SemanticLiteral make_raw_literal(
      int64_t category,
      int64_t predicate_index,
      const std::vector< int64_t >& object_indices,
      bool positive
   ) const;

   [[nodiscard]] std::shared_ptr< const SemanticProblemContext > get_problem_context() const;
   /** The domain-level schema, shared by every task of this domain. */
   [[nodiscard]] std::shared_ptr< const SemanticSchemaContext > get_schema_context() const;
   [[nodiscard]] const views::Context& get_view_context() const noexcept;
   [[nodiscard]] views::StateView< tyr::planning::StateView< tyr::LiftedTag >, tyr::LiftedTag >
   make_view(const tyr::planning::StateView< tyr::LiftedTag >& state) const;
   [[nodiscard]] views::StateView< tyr::planning::StateView< tyr::GroundTag >, tyr::GroundTag >
   make_view(const tyr::planning::StateView< tyr::GroundTag >& state) const;

   /**
    * Borrow a range of Tyr ground actions as granular action Views.
    *
    * The returned range holds no action of its own; the caller's container must
    * outlive it, exactly as for `make_view`.
    */
   template < std::ranges::input_range Actions >
   [[nodiscard]] auto make_action_views(Actions&& actions) const
   {
      using NativeAction = std::remove_cvref_t< std::ranges::range_value_t< Actions > >;
      return mifrost::views::TransformRange{
         std::forward< Actions >(actions), [context = &get_view_context()](const auto& action) {
            return views::GroundActionView< NativeAction >{action, *context};
         }
      };
   }

  private:
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost::pytyr
