/**
 * @file semantic_flat_encoder.hpp
 * @brief Native PyTyr adapter for the backend-neutral semantic flat encoder.
 */
#pragma once

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
 * semantic records. The planning task is copied into the adapter so all
 * repository-backed views used to build cached schema metadata remain alive.
 */
class MIFROST_API SemanticFlatRelationEncoder {
  public:
   using Config = FlatRelationEncoderConfig;

   SemanticFlatRelationEncoder(
      const tyr::formalism::planning::PlanningTask& task,
      Config config = {}
   );
   SemanticFlatRelationEncoder(const SemanticFlatRelationEncoder&) = delete;
   SemanticFlatRelationEncoder& operator=(const SemanticFlatRelationEncoder&) = delete;
   SemanticFlatRelationEncoder(SemanticFlatRelationEncoder&&) noexcept;
   SemanticFlatRelationEncoder& operator=(SemanticFlatRelationEncoder&&) noexcept;
   ~SemanticFlatRelationEncoder();

   [[nodiscard]] SemanticFlatRelationInput make_input(
      const tyr::planning::StateView< tyr::planning::LiftedTag >& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions = {}
   ) const;
   [[nodiscard]] SemanticFlatRelationInput make_input(
      const tyr::planning::StateView< tyr::planning::GroundTag >& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions = {}
   ) const;

   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const tyr::planning::StateView< tyr::planning::LiftedTag >& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions = {}
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const tyr::planning::StateView< tyr::planning::GroundTag >& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions = {}
   ) const;

   [[nodiscard]] const SemanticFlatRelationEncoderEngine& get_engine() const;

  private:
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost::pytyr
