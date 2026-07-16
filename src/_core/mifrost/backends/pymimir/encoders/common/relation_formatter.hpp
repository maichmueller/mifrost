/**
 * @file relation_formatter.hpp
 * @brief Pymimir naming helpers for relations, atoms, literals, and objects.
 *
 * All encoders use this file so relation names stay the same in schema setup,
 * emitted data, and tests.
 */
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
#include <ranges>
#include <string>
#include <string_view>
#include <strong_type/strong_type.hpp>
#include <type_traits>
#include <utility>

#include "mifrost/core/encoders/common/goal_derivation.hpp"
#include "mifrost/core/utils/type_traits.hpp"

namespace mifrost {

/**
 * @brief Literal polarity prefix mode.
 */
enum class LiteralPrefix {
   positive,
   negative,
};

/**
 * @brief Strong type for goal layer index.
 */
struct GoalLevel: strong::type< std::size_t, GoalLevel, strong::regular > {
   using base = strong::type< std::size_t, GoalLevel, strong::regular >;
   using base::base;  // keep the normal constructors

   template < std::integral I >
      requires(not std::same_as< detail::raw_t< I >, bool >)
   explicit constexpr GoalLevel(I v) : base(static_cast< std::size_t >(v))
   {
   }
};

/**
 * @brief Central formatter for relation, atom, literal, action and object keys.
 *
 * This class defines naming conventions used across all encoders and tests.
 * Keep all formatter changes here so relation naming stays globally consistent.
 */
struct RelationFormatter {
   static constexpr auto kPositivePrefix = "[+]";
   static constexpr auto kNegativePrefix = "[-]";
   static constexpr auto kGoalSuffixes = std::array{"[g]", "[sg]", "[ssg]", "[sssg]"};
   static constexpr auto kGoalSatisfiedSuffix = "[sat]";
   static constexpr auto kGoalUnsatisfiedSuffix = "[unsat]";
   static constexpr auto kGoalSatisfiedAddedSuffix = "[sat+]";
   static constexpr auto kGoalSatisfiedRemovedSuffix = "[sat-]";
   static constexpr auto kDefaultNullarySymbolName = "![nullary_symbol]!";

   /// Return no goal-level suffix.
   static std::string_view goal_level_suffix(std::nullopt_t) { return ""; }

   /// Return level suffix for a concrete goal level.
   static std::string_view goal_level_suffix(const GoalLevel& level)
   {
      return kGoalSuffixes.at(level.value_of());
   }

   /// Return no satisfaction suffix.
   static std::string_view goal_derivation_suffix(std::nullopt_t) { return ""; }

   /// Return suffix for a concrete goal-satisfaction mode.
   static std::string_view goal_derivation_suffix(GoalDerivation satisfaction)
   {
      switch(satisfaction) {
         case GoalDerivation::plain: return kGoalSuffixes[0];
         case GoalDerivation::satisfied: return kGoalSatisfiedSuffix;
         case GoalDerivation::unsatisfied: return kGoalUnsatisfiedSuffix;
         case GoalDerivation::added_satisfied: return kGoalSatisfiedAddedSuffix;
         case GoalDerivation::added_unsatisfied: return kGoalSatisfiedRemovedSuffix;
      }
      return "";
   }

   /// Return no polarity prefix.
   static std::string_view polarity_prefix(std::nullopt_t) { return ""; }

   /// Return polarity prefix ("[+]" / "[-]").
   static std::string_view polarity_prefix(bool polarity)
   {
      return polarity ? kPositivePrefix : kNegativePrefix;
   }

   /// Format a predicate relation name from raw predicate name + optional suffix.
   static std::string
   format_predicate(const std::string_view name, const std::string_view suffix = "")
   {
      return fmt::format("{}{}", name, suffix);
   }

   /// Format a predicate relation name from typed predicate object.
   template < typename Tag, typename... Args >
   static std::string
   format_predicate(const mimir::formalism::Predicate< Tag > predicate, Args&&... args)
   {
      return format_predicate(predicate->get_name(), std::forward< Args >(args)...);
   }

   /// Format predicate with explicit goal-level suffix.
   static std::string format_predicate(
      const std::string_view name,
      const GoalLevel goal_level,
      const std::string_view suffix = ""
   )
   {
      return fmt::format("{}{}{}", name, suffix, goal_level_suffix(goal_level));
   }

   /**
    * @brief Fully-parameterized predicate formatter.
    *
    * Optional arguments are modeled via overload resolution on std::nullopt_t
    * to keep dispatch compile-time and avoid runtime optional branching.
    */
   template <
      typename GoalLevelArg = std::nullopt_t,
      typename SatisfactionArg = std::nullopt_t,
      typename PolarityArg = std::nullopt_t >
      requires detail::is_any_v< GoalLevelArg, GoalLevel, std::nullopt_t >
               and detail::is_any_v< SatisfactionArg, GoalDerivation, std::nullopt_t >
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
         goal_derivation_suffix(satisfaction)
      );
   }

   /// Format a ground atom.
   template < typename P >
   static std::string
   format_atom(mimir::formalism::GroundAtom< P > atom, const std::string_view suffix = "")
   {
      const auto& predicate = atom->get_predicate();
      const std::string name = fmt::format("{}{}", predicate->get_name(), suffix);
      if(predicate->get_arity() == 0) {
         return fmt::format("({})", name);
      }
      return fmt::format(
         "({} {})",
         name,
         fmt::join(
            std::views::transform(
               atom->get_objects(),
               [](const mimir::formalism::Object& obj) { return obj->get_name(); }
            ),
            " "
         )
      );
   }

   /// Format a ground literal.
   template <
      typename P,
      typename GoalLevelArg = std::nullopt_t,
      typename SatisfactionArg = std::nullopt_t,
      typename PolarityArg = std::nullopt_t >
      requires detail::is_any_v< GoalLevelArg, GoalLevel, std::nullopt_t >
               and detail::is_any_v< SatisfactionArg, GoalDerivation, std::nullopt_t >
               and detail::is_any_v< PolarityArg, bool, std::nullopt_t >
   static std::string format_literal(
      mimir::formalism::GroundLiteral< P > literal,
      GoalLevelArg goal_level = std::nullopt,
      SatisfactionArg satisfaction = std::nullopt,
      PolarityArg polarity = std::nullopt,
      const std::string_view suffix = ""
   )
   {
      // Optional arguments are resolved at compile time via overloads on std::nullopt_t.
      const bool literal_polarity = [&]() {
         if constexpr(std::is_same_v< PolarityArg, std::nullopt_t >) {
            return literal->get_polarity();
         } else {
            return polarity;
         }
      }();
      const std::string atom_str = format_atom(literal->get_atom(), suffix);
      const std::string literal_str = fmt::format(
         "{}{}", polarity_prefix(literal_polarity), atom_str
      );
      return fmt::format(
         "{}{}{}", literal_str, goal_level_suffix(goal_level), goal_derivation_suffix(satisfaction)
      );
   }

   /// Format action schema relation key.
   static std::string format_action_schema(const mimir::formalism::ActionImpl& action)
   {
      return action.get_name();
   }

   /// Format grounded action relation key.
   static std::string format_action(const mimir::formalism::GroundAction action)
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

   /// Format object key.
   static std::string format_object(const mimir::formalism::ObjectImpl& object)
   {
      return object.get_name();
   }

   /// Callable formatter entrypoint forwarding to overload set.
   template < typename... Args >
   std::string operator()(Args&&... args) const
   {
      return format(std::forward< Args >(args)...);
   }

   /// Generic format overload for raw predicate names.
   template < typename... Args >
   static std::string format(std::string_view name, Args&&... args)
   {
      return format_predicate(name, std::forward< Args >(args)...);
   }

   /// Generic format overload for typed predicates.
   template < typename Tag, typename... Args >
   static std::string format(mimir::formalism::Predicate< Tag > predicate, Args&&... args)
   {
      return format_predicate(predicate, std::forward< Args >(args)...);
   }

   /// Generic format overload for literals.
   template < typename P, typename... Args >
   static std::string format(mimir::formalism::GroundLiteral< P > literal, Args&&... args)
   {
      return format_literal(literal, std::forward< Args >(args)...);
   }

   /// Generic format overload for atoms.
   template < typename P, typename... Args >
   static std::string format(mimir::formalism::GroundAtom< P > atom, Args&&... args)
   {
      return format_atom(atom, std::forward< Args >(args)...);
   }

   static std::string format(const mimir::formalism::ObjectImpl& object)
   {
      return format_object(object);
   }

   static std::string format(const mimir::formalism::GroundAction& action)
   {
      return format_action(action);
   }

   static std::string format(const mimir::formalism::ActionImpl& action)
   {
      return format_action_schema(action);
   }
};

}  // namespace mifrost
