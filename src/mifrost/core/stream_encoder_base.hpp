#pragma once

#include "batch_builder.hpp"

#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <span>

namespace mifrost
{

/**
 * @brief Runtime interface for stream encoders (for Python trampolines).
 */
class StreamEncoderInterface
{
public:
    virtual ~StreamEncoderInterface() = default;

    virtual void encode_state(const mimir::search::State& state, BatchBuilder& builder) = 0;
};

/**
 * @brief CRTP base for stream encoders, providing a unified interface.
 */
template<typename Derived>
class StreamEncoderBase : public StreamEncoderInterface
{
public:
    void encode_state(const mimir::search::State& state, BatchBuilder& builder) override { static_cast<Derived*>(this)->encode_state_impl(state, builder); }

    template<typename GoalTag>
    void encode_step(const mimir::search::State& state,
                     std::span<const mimir::formalism::GroundLiteral<GoalTag>> goals,
                     std::span<const mimir::formalism::GroundAction> actions,
                     BatchBuilder& builder)
    {
        static_cast<Derived*>(this)->template encode_step_impl<GoalTag>(state, goals, actions, builder);
    }
};

}  // namespace mifrost
