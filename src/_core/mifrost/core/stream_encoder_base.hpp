#pragma once

#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <span>

#include "batch_builder.hpp"

namespace mifrost {

/**
 * @brief Runtime interface for stream encoders (for Python trampolines).
 */
class StreamEncoderInterface {
  public:
   virtual ~StreamEncoderInterface() = default;

   /**
    * @brief Encode a single state into a provided builder.
    *
    * Implementations append graph content to the open graph in @p builder.
    */
   virtual void encode_state(const mimir::search::State& state, BatchBuilder& builder) = 0;
};

/**
 * @brief CRTP base for stream encoders with static dispatch.
 *
 * This base preserves one runtime-virtual entrypoint (`encode_state`) for
 * interface usage while delegating concrete logic to derived implementations.
 */
template < typename Derived >
class StreamEncoderBase: public StreamEncoderInterface {
  public:
   /// Runtime interface entrypoint delegating to Derived::encode_state_impl.
   void encode_state(const mimir::search::State& state, BatchBuilder& builder) override
   {
      static_cast< Derived* >(this)->encode_state_impl(state, builder);
   }

   /// Static-dispatch encode step with goals/actions for derived stream encoders.
   template < typename GoalTag >
   void encode_step(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      static_cast< Derived* >(this)->template encode_step_impl< GoalTag >(
         state, goals, actions, builder
      );
   }
};

}  // namespace mifrost
