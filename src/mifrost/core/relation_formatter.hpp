#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <mimir/common/formatter.hpp>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/formatter.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_atom.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/formalism/object.hpp>
#include <mimir/formalism/predicate.hpp>
#include <optional>
#include <string>
#include <strong_type/strong_type.hpp>
#include <type_traits>

#include "utils/type_traits.hpp"

namespace mifrost {

enum class GoalSatisfaction {
   none,
   satisfied,
   unsatisfied,
   added_satisfied,
   added_unsatisfied,
};

enum class LiteralPrefix {
   positive,
   negative,
};

struct GoalLevel: strong::type< std::size_t, GoalLevel, strong::regular > {
   using base = strong::type< std::size_t, GoalLevel, strong::regular >;
   using base::base;  // keep the normal constructors

   template < std::integral I >
      requires(! std::same_as< detail::raw_t< I >, bool >)
   explicit constexpr GoalLevel(I v) : base(static_cast< std::size_t >(v))
   {
   }
};

struct RelationFormatter {
   static constexpr auto kPositivePrefix = "[+]";
   static constexpr auto kNegativePrefix = "[-]";
   static constexpr auto kGoalSuffixes = std::array{"[g]", "[sg]", "[ssg]", "[sssg]"};
   static constexpr auto kGoalSatisfiedSuffix = "[sat]";
   static constexpr auto kGoalUnsatisfiedSuffix = "[unsat]";
   static constexpr auto kGoalSatisfiedAddedSuffix = "[sat+]";
   static constexpr auto kGoalSatisfiedRemovedSuffix = "[sat-]";
   static constexpr auto kDefaultNullarySymbolName = "![nullary_symbol]!";

   static std::string_view goal_level_suffix(std::nullopt_t) { return ""; }

   static std::string_view goal_level_suffix(const GoalLevel& level)
   {
      return kGoalSuffixes.at(level.value_of());
   }

   static std::string_view goal_satisfaction_suffix(std::nullopt_t) { return ""; }

   static std::string_view goal_satisfaction_suffix(GoalSatisfaction satisfaction)
   {
      switch(satisfaction) {
         case GoalSatisfaction::none: return "";
         case GoalSatisfaction::satisfied: return kGoalSatisfiedSuffix;
         case GoalSatisfaction::unsatisfied: return kGoalUnsatisfiedSuffix;
         case GoalSatisfaction::added_satisfied: return kGoalSatisfiedAddedSuffix;
         case GoalSatisfaction::added_unsatisfied: return kGoalSatisfiedRemovedSuffix;
      }
      return "";
   }

   static std::string_view polarity_prefix(std::nullopt_t) { return ""; }

   static std::string_view polarity_prefix(bool polarity)
   {
      return polarity ? kPositivePrefix : kNegativePrefix;
   }

   static std::string
   format_predicate(const std::string_view name, const std::string_view suffix = "")
   {
      return fmt::format("{}{}", name, suffix);
   }

   static std::string format_predicate(
      const std::string_view name,
      const GoalLevel goal_level,
      const std::string& suffix = ""
   )
   {
      return fmt::format("{}{}{}", name, suffix, goal_level_suffix(goal_level));
   }

   template <
      typename GoalLevelArg = std::nullopt_t,
      typename SatisfactionArg = std::nullopt_t,
      typename PolarityArg = std::nullopt_t >
      requires detail::is_any_v< GoalLevelArg, GoalLevel, std::nullopt_t >
               and detail::is_any_v< SatisfactionArg, GoalSatisfaction, std::nullopt_t >
               and detail::is_any_v< PolarityArg, bool, std::nullopt_t >
   static std::string format_predicate(
      const std::string_view name,
      GoalLevelArg goal_level = std::nullopt,
      SatisfactionArg satisfaction = std::nullopt,
      PolarityArg polarity = std::nullopt,
      const std::string_view suffix = ""
   )
   {
      // If the caller passes `std::nullopt` (or omits the argument), overload resolution picks
      // the `std::nullopt_t` overloads above. This removes any optional-related logic at
      // compile-time, avoids indirections.
      return fmt::format(
         "{}{}{}{}{}",
         polarity_prefix(polarity),
         name,
         suffix,
         goal_level_suffix(goal_level),
         goal_satisfaction_suffix(satisfaction)
      );
   }

   template < typename P >
   static std::string
   format_atom(mimir::formalism::GroundAtom< P > atom, const std::string& suffix = "")
   {
      return fmt::format("{}{}", mimir::to_string(atom), suffix);
   }

   template <
      typename P,
      typename GoalLevelArg = std::nullopt_t,
      typename SatisfactionArg = std::nullopt_t,
      typename PolarityArg = std::nullopt_t >
      requires detail::is_any_v< GoalLevelArg, GoalLevel, std::nullopt_t >
               and detail::is_any_v< SatisfactionArg, GoalSatisfaction, std::nullopt_t >
               and detail::is_any_v< PolarityArg, bool, std::nullopt_t >
   static std::string format_literal(
      mimir::formalism::GroundLiteral< P > literal,
      GoalLevelArg goal_level = std::nullopt,
      SatisfactionArg satisfaction = std::nullopt,
      [[maybe_unused]] PolarityArg polarity = std::nullopt,
      const std::string& suffix = ""
   )
   {
      // Optional arguments are resolved at compile time via overloads on std::nullopt_t.
      return fmt::format(
         "{}{}{}{}",
         mimir::to_string(literal),
         suffix,
         goal_level_suffix(goal_level),
         goal_satisfaction_suffix(satisfaction)
      );
   }

   static std::string format_action_schema(const mimir::formalism::ActionImpl& action)
   {
      return action.get_name();
   }

   static std::string format_action(mimir::formalism::GroundAction action)
   {
      return fmt::format(
         "({} {})",
         action->get_action()->get_name(),
         fmt::join(
            std::views::transform(
               action->get_objects(),
               [](const mimir::formalism::Object& obj) { return obj->get_name(); }
            ),
            " "
         )
      );
   }

   static std::string format_object(const mimir::formalism::ObjectImpl& object)
   {
      return object.get_name();
   }
};

}  // namespace mifrost
