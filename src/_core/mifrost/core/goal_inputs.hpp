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

namespace mifrost {  //

using FluentLiteral = mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >;
using DerivedLiteral = mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag >;
using StaticLiteral = mimir::formalism::GroundLiteral< mimir::formalism::StaticTag >;
/// Order matters for nanobind casting: first matching alternative wins.
using LiteralVariant = std::variant< FluentLiteral, DerivedLiteral, StaticLiteral >;

template < typename R >
concept AnyLiteralRange = std::ranges::input_range< R >
                          and detail::is_any_v<
                             std::ranges::range_value_t< R >,
                             LiteralVariant,
                             FluentLiteral,
                             DerivedLiteral,
                             StaticLiteral >;

template < typename RR >
concept AnyLiteralRangeRange = std::ranges::input_range< RR >
                               and AnyLiteralRange< std::ranges::range_reference_t< RR > >;

template < class T >
concept pair_like = requires(T t) {
   // must follow the tuple protocol
   typename std::tuple_size< detail::raw_t< T > >::type;
   requires(std::tuple_size_v< detail::raw_t< T > > == 2);

   // elements must be accessible
   std::get< 0 >(t);
   std::get< 1 >(t);

   // (optional but useful) element types must exist
   typename std::tuple_element_t< 0, std::remove_cvref_t< T > >;
   typename std::tuple_element_t< 1, std::remove_cvref_t< T > >;
};

template < typename LR >
concept LeveledGoalLayers = std::ranges::input_range< LR >
                            and pair_like< std::ranges::range_value_t< LR > >
                            and std::convertible_to<
                               detail::raw_t< decltype(std::get< 0 >(
                                  std::declval< std::ranges::range_reference_t< LR > >()
                               )) >,
                               size_t >
                            and AnyLiteralRange< decltype(std::get< 1 >(
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

   template < typename R >
      requires AnyLiteralRange< R >
   explicit GoalInputs(R&& goals, const size_t level = 0)
   {
      extend(FWD(goals), level);
   }

   /// Append a single goal with a given level
   template < typename LiteralT >
   void append(LiteralT&& literal, size_t level)
   {
      if constexpr(std::is_same_v< detail::raw_t< LiteralT >, FluentLiteral >) {
         fluent_goals.emplace_back(FWD(literal));
         fluent_goal_levels[literal] = level;
      } else if constexpr(std::is_same_v< detail::raw_t< LiteralT >, DerivedLiteral >) {
         derived_goals.emplace_back(FWD(literal));
         derived_goal_levels[literal] = level;
      } else {
         static_goals.emplace_back(FWD(literal));
         static_goal_levels[literal] = level;
      }
   }

   /// Append more goals from any iterable/range with a given goal level.
   template < typename R >
      requires(AnyLiteralRange< R > and not std::same_as< detail::raw_t< R >, GoalInputs >)
   void extend(R&& goals, const size_t level)
   {
      for(auto&& goal : goals) {
         using LiteralT = detail::raw_t< decltype(goal) >;
         if constexpr(not std::same_as< LiteralT, LiteralVariant >) {
            append(FWD(goal), level);
         } else {
            std::visit([&](auto&& literal) { append(FWD(literal), level); }, FWD(goal));
         }
      }
   }

   /// Construct from any iterable/range of goals (implicit single level).
   template < typename R >
      requires(AnyLiteralRange< R > and not std::same_as< detail::raw_t< R >, GoalInputs >)
   explicit GoalInputs(R&& goals, size_t level = 0)
   {
      extend(std::forward< R >(goals), level);
   }

   /// Construct from any iterable/range of goal layers (levels counted up per layer).
   template < typename RR >
      requires(AnyLiteralRangeRange< RR > and not std::same_as< detail::raw_t< RR >, GoalInputs >)
   explicit GoalInputs(RR&& goal_layers)
   {
      size_t level = 0;
      for(auto&& layer : goal_layers) {
         append_range(layer, level++);
      }
   }

   /// Construct from any iterable/range of (level, goals) entries.
   template < typename LR >
      requires(LeveledGoalLayers< LR > and not std::same_as< detail::raw_t< LR >, GoalInputs >)
   explicit GoalInputs(LR&& goal_layers)
   {
      for(auto&& entry : goal_layers) {
         const size_t level = static_cast< size_t >(std::get< 0 >(entry));
         auto&& goals = std::get< 1 >(entry);
         extend(goals, level);
      }
   }
};

}  // namespace mifrost
