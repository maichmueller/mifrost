#pragma once

#include <ankerl/unordered_dense.h>

#include <mimir/formalism/ground_literal.hpp>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mifrost {

/**
 * @brief Tagged goal container shared by all encoders.
 *
 * Stores goals separated by tag (static/fluent/derived) and optional goal-level
 * maps used by layered-goal encodings.
 */
struct GoalInputs {
   using FluentLiteral = mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >;
   using DerivedLiteral = mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag >;
   using StaticLiteral = mimir::formalism::GroundLiteral< mimir::formalism::StaticTag >;
   /// Order matters for nanobind casting: first matching alternative wins.
   using AnyGoalLiteral = std::variant< FluentLiteral, DerivedLiteral, StaticLiteral >;

   /// Static goals.
   mimir::formalism::GroundLiteralList< mimir::formalism::StaticTag > static_goals;
   /// Fluent goals.
   mimir::formalism::GroundLiteralList< mimir::formalism::FluentTag > fluent_goals;
   /// Derived goals.
   mimir::formalism::GroundLiteralList< mimir::formalism::DerivedTag > derived_goals;
   /// Optional per-goal layer map for static goals.
   ankerl::unordered_dense::
      map< mimir::formalism::GroundLiteral< mimir::formalism::StaticTag >, int >
         static_goal_levels;
   /// Optional per-goal layer map for fluent goals.
   ankerl::unordered_dense::
      map< mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >, int >
         fluent_goal_levels;
   /// Optional per-goal layer map for derived goals.
   ankerl::unordered_dense::
      map< mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag >, int >
         derived_goal_levels;

   GoalInputs() = default;

   /// Convenience ctor for Python conversion from list[variant].
   explicit GoalInputs(const std::vector< AnyGoalLiteral >& goals) { append(goals, 0); }

   /// Convenience ctor for Python conversion with explicit goal level.
   GoalInputs(const std::vector< AnyGoalLiteral >& goals, int level) { append(goals, level); }

  private:
   void append(const std::vector< AnyGoalLiteral >& goals, int level)
   {
      for(const auto& goal : goals) {
         std::visit(
            [&](const auto& literal) {
               using LiteralT = std::decay_t< decltype(literal) >;
               if constexpr(std::is_same_v< LiteralT, FluentLiteral >) {
                  fluent_goals.emplace_back(literal);
                  fluent_goal_levels[literal] = level;
               } else if constexpr(std::is_same_v< LiteralT, DerivedLiteral >) {
                  derived_goals.emplace_back(literal);
                  derived_goal_levels[literal] = level;
               } else {
                  static_goals.emplace_back(literal);
                  static_goal_levels[literal] = level;
               }
            },
            goal
         );
      }
   }
};

}  // namespace mifrost
