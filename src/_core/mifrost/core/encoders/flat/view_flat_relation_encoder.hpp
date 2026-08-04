/**
 * @file view_flat_relation_encoder.hpp
 * @brief Canonical flat-lane traversal over backend Views.
 */
#pragma once

#include <utility>

#include "mifrost/core/views/canonical.hpp"

namespace mifrost::canonical {

enum class FlatLane {
   state,
   goal,
   action,
};

/**
 * Visit the logical flat-relation lanes without materializing backend values.
 *
 * The callback receives the lane tag and one View value. It is intentionally
 * the only backend-facing part of this algorithm; relation schema and tensor
 * emission remain owned by the neutral flat runtime.
 */
template <
   views::StateView State,
   views::LiteralRange Goals,
   views::GroundActionRange Actions,
   typename Callback >
void visit_flat_lanes(const State& state, Goals&& goals, Actions&& actions, Callback&& callback)
{
   auto&& visitor = callback;
   for_each_state_atom(state, [&](const auto& atom) { visitor(FlatLane::state, atom); });
   for(auto&& goal : goals) {
      visitor(FlatLane::goal, goal);
   }
   for(auto&& action : actions) {
      visitor(FlatLane::action, action);
   }
}

}  // namespace mifrost::canonical
