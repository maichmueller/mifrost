/**
 * @file goal_derivation.hpp
 * @brief Planning-backend-neutral goal-derivation modes.
 */
#pragma once

#include <algorithm>
#include <optional>
#include <ranges>

namespace mifrost {

/**
 * @brief Goal-derived relation variants exposed by encoder configs.
 *
 * This enum describes an encoding choice, not a planning-library object.  It
 * lives outside relation_formatter.hpp so semantic encoders can use it without
 * importing Mimir formalism headers.
 */
enum class GoalDerivation {
   plain,
   satisfied,
   unsatisfied,
   added_satisfied,
   added_unsatisfied,
};

template < typename Range >
bool has_non_plain_goal_derivations(const Range& derivations)
{
   return std::ranges::any_of(derivations, [](GoalDerivation derivation) {
      return derivation != GoalDerivation::plain;
   });
}

template < typename Range >
bool includes_plain_goal_derivation(const Range& derivations)
{
   return std::ranges::find(derivations, GoalDerivation::plain) != std::ranges::end(derivations);
}

template < typename Range >
auto goal_satisfaction_derivations(const Range& derivations)
{
   return derivations | std::views::filter([](GoalDerivation derivation) {
             return derivation != GoalDerivation::plain;
          });
}

inline std::optional< GoalDerivation >
delta_goal_satisfaction_derivation(bool goal_polarity, bool added_match, bool removed_match)
{
   if(goal_polarity ? added_match : removed_match) {
      return GoalDerivation::added_satisfied;
   }
   if(goal_polarity ? removed_match : added_match) {
      return GoalDerivation::added_unsatisfied;
   }
   return std::nullopt;
}

}  // namespace mifrost
