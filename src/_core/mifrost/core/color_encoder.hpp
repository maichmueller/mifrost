#pragma once

#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_builder.hpp"
#include "common_types.hpp"
#include "goal_inputs.hpp"
#include "relation_formatter.hpp"
#include "stream_encoder_base.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct ColorBatchInputs;
}
}  // namespace batch_input

/**
 * @brief Color-encoding engine producing homogeneous graph payloads.
 */
class ColorEncoderEngine {
  public:
   /// Runtime config for color encoding behavior.
   struct Config {
      /// Emit edge feature tensors.
      bool edge_features = false;
      /// Emit global predicate nodes in addition to atom/object nodes.
      bool enable_global_predicate_nodes = false;
   };

   explicit ColorEncoderEngine(const mimir::formalism::DomainImpl& domain);
   ColorEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);

   explicit ColorEncoderEngine(mimir::formalism::Domain domain);
   ColorEncoderEngine(mimir::formalism::Domain domain, Config config);

   /// Encode state-only into an existing builder.
   void encode_state(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state_impl(state, builder);
   }

   /// Encode state-only into an existing builder.
   void encode(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state(state, builder);
   }

   /// Encode state + goals into an existing builder.
   void encode(const mimir::search::State& state, const GoalInputs& goals, BatchBuilder& builder)
   {
      encode_impl(state, goals, {}, builder);
   }

   /// Encode state + goals + actions into an existing builder.
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      encode_impl(state, goals, actions, builder);
   }

   /// Encode a parsed batch input plan into one batch encoding.
   BatchBuilder::BatchEncoding encode_batch(const batch_input::parsed::ColorBatchInputs& inputs);

   /// Return effective engine config.
   const Config& get_config() const { return config_; }

  private:
   /// Internal state-only encode implementation.
   void encode_state_impl(const mimir::search::State& state, BatchBuilder& builder);

   /// Internal full encode implementation.
   void encode_impl(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   /// Optional owning domain storage for handle-based construction.
   std::optional< mimir::formalism::Domain > domain_holder_;
   /// Active domain implementation reference.
   const mimir::formalism::DomainImpl& domain_;
   /// Effective runtime config.
   Config config_;
};

/**
 * @brief Payload for one streaming color encode step.
 */
struct ColorStepInput {
   const mimir::search::State* state = nullptr;
   const GoalInputs* goals = nullptr;
};

/**
 * @brief Streaming color encoder with static dispatch.
 */
class ColorStreamEncoder: public StreamEncoderBase< ColorStreamEncoder, ColorStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "homo"; }

   explicit ColorStreamEncoder(ColorEncoderEngine& engine) : engine_(&engine) { reset(); }

   int64_t append(const mimir::search::State& state)
   {
      ColorStepInput step;
      step.state = &state;
      return StreamEncoderBase::append(step);
   }

   int64_t append(const mimir::search::State& state, const GoalInputs& goals)
   {
      ColorStepInput step;
      step.state = &state;
      step.goals = &goals;
      return StreamEncoderBase::append(step);
   }

   void update(int64_t id, const mimir::search::State& state)
   {
      ColorStepInput step;
      step.state = &state;
      StreamEncoderBase::update(id, step);
   }

   void update(int64_t id, const mimir::search::State& state, const GoalInputs& goals)
   {
      ColorStepInput step;
      step.state = &state;
      step.goals = &goals;
      StreamEncoderBase::update(id, step);
   }

   void encode_step(const ColorStepInput& step, BatchBuilder& builder)
   {
      if(engine_ == nullptr or step.state == nullptr) {
         throw std::invalid_argument("ColorStreamEncoder requires a valid engine/state");
      }
      if(step.goals == nullptr) {
         engine_->encode(*step.state, builder);
      } else {
         engine_->encode(*step.state, *step.goals, builder);
      }
   }

  private:
   ColorEncoderEngine* engine_ = nullptr;
};

}  // namespace mifrost
