#pragma once

#include <ankerl/unordered_dense.h>

#include <concepts>
#include <mimir/formalism/ground_literal.hpp>
#include <range/v3/view/enumerate.hpp>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "utils/macro.hpp"
#include "utils/type_traits.hpp"

namespace mifrost {
using FluentLiteral = mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >;
using DerivedLiteral = mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag >;
using StaticLiteral = mimir::formalism::GroundLiteral< mimir::formalism::StaticTag >;
/// Order matters for nanobind casting: first matching alternative wins.
using AnyGoalLiteral = std::variant< FluentLiteral, DerivedLiteral, StaticLiteral >;

template < typename T >
concept AnyGoalLiteralLike = std::constructible_from< AnyGoalLiteral, T >;

template < typename R >
concept AnyGoalLiteralRange = std::ranges::input_range< R >
                              && AnyGoalLiteralLike<
                                 detail::raw_t< std::ranges::range_reference_t< R > > >;

template < typename RR >
concept AnyGoalLiteralRangeRange = std::ranges::input_range< RR >
                                   && AnyGoalLiteralRange< std::ranges::range_reference_t< RR > >;

template < typename T >
concept PairLike = requires(T t) {
   std::get< 0 >(t);
   std::get< 1 >(t);
};

template < typename LR >
concept LeveledGoalLayers = std::ranges::input_range< LR >
                            && PairLike< detail::raw_t< std::ranges::range_reference_t< LR > > >
                            && std::convertible_to<
                               detail::raw_t< decltype(std::get< 0 >(
                                  std::declval< std::ranges::range_reference_t< LR > >()
                               )) >,
                               size_t >
                            && AnyGoalLiteralRange< decltype(std::get< 1 >(
                               std::declval< std::ranges::range_reference_t< LR > >()
                            )) >;

/**
 * @brief Tagged goal container shared by all encoders.
 *
 * Stores goals separated by tag (static/fluent/derived) and optional goal-level
 * maps used by layered-goal encodings.
 */
struct GoalInputs {
   /// Static goals.
   mimir::formalism::GroundLiteralList< mimir::formalism::StaticTag > static_goals;
   /// Fluent goals.
   mimir::formalism::GroundLiteralList< mimir::formalism::FluentTag > fluent_goals;
   /// Derived goals.
   mimir::formalism::GroundLiteralList< mimir::formalism::DerivedTag > derived_goals;
   /// Optional per-goal layer map for static goals.
   ankerl::unordered_dense::
      map< mimir::formalism::GroundLiteral< mimir::formalism::StaticTag >, size_t >
         static_goal_levels;
   /// Optional per-goal layer map for fluent goals.
   ankerl::unordered_dense::
      map< mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >, size_t >
         fluent_goal_levels;
   /// Optional per-goal layer map for derived goals.
   ankerl::unordered_dense::
      map< mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag >, size_t >
         derived_goal_levels;

   GoalInputs() = default;

   /// ctor for Python conversion from list[variant] (with optional goal level)
   explicit GoalInputs(const std::vector< AnyGoalLiteral >& goals, const size_t level = 0)
   {
      append_range(goals, level);
   }

  private:
   template < typename R >
      requires AnyGoalLiteralRange< R >
   void append_range(R&& goals, size_t level)
   {
      for(auto&& goal : goals) {
         std::visit(
            [&]< typename LiteralT >(const LiteralT& literal) {
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
            FWD(goal)
         );
      }
   }

  public:
   /// Construct from any iterable/range of goals (implicit single level).
   template < typename R >
      requires(AnyGoalLiteralRange< R > && ! std::same_as< detail::raw_t< R >, GoalInputs >)
   explicit GoalInputs(R&& goals, size_t level = 0)
   {
      append_range(std::forward< R >(goals), level);
   }

   /// Construct from any iterable/range of goal layers (levels counted up per layer).
   template < typename RR >
      requires(AnyGoalLiteralRangeRange< RR > && ! std::same_as< detail::raw_t< RR >, GoalInputs >)
   explicit GoalInputs(RR&& goal_layers)
   {
      size_t level = 0;
      for(auto&& layer : goal_layers) {
         append_range(layer, level++);
      }
   }

   /// Construct from any iterable/range of (level, goals) entries.
   template < typename LR >
      requires(LeveledGoalLayers< LR > && ! std::same_as< detail::raw_t< LR >, GoalInputs >)
   explicit GoalInputs(LR&& goal_layers)
   {
      for(auto&& entry : goal_layers) {
         const size_t level = static_cast< size_t >(std::get< 0 >(entry));
         auto&& goals = std::get< 1 >(entry);
         append_range(goals, level);
      }
   }

   /// Append more goals from any iterable/range with a given goal level.
   template < typename R >
      requires(AnyGoalLiteralRange< R > && ! std::same_as< detail::raw_t< R >, GoalInputs >)
   void extend(R&& goals, const size_t level)
   {
      append_range(std::forward< R >(goals), level);
   }
};

}  // namespace mifrost
