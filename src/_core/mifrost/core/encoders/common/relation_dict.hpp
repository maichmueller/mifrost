/**
 * @file relation_dict.hpp
 * @brief Shared map of relation names to arities.
 *
 * This file builds the common set of relation names from a domain. Different
 * encoders may add extra leading slots, but they still use the same names and
 * base arity rules.
 */
#pragma once

#include <algorithm>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/predicate.hpp>
#include <string>
#include <utility>
#include <vector>

#include "mifrost/core/encoders/common/relation_dict_types.hpp"
#include "mifrost/core/encoders/common/relation_formatter.hpp"

namespace mifrost {

/**
 * @brief Build the neutral relation table from a Pymimir domain.
 */
inline RelationDict build_pymimir_relation_dict(
   const mimir::formalism::DomainImpl& domain,
   const std::vector< mimir::formalism::Action >& actions,
   const RelationDictConfig& config,
   int predicate_arity_offset = 0,
   int action_arity_offset = 1
)
{
   RelationDict result;
   result.max_goal_level = std::max(0, config.max_goal_level);
   result.support_literals = config.support_literals;
   result.goal_derivations = config.goal_derivations;

   std::vector< std::pair< std::string, int > > regular_predicates;
   auto collect_predicates = [&]< typename Tag >(Tag) {
      for(auto pred : domain.get_predicates< Tag >()) {
         const auto& name = pred->get_name();
         const int pred_arity = std::max(
            0, static_cast< int >(pred->get_arity()) + predicate_arity_offset
         );
         result.arity[RelationFormatter::format_predicate(name)] = pred_arity;
         if(config.top_type_predicates.contains(name)) {
            continue;
         }
         regular_predicates.emplace_back(name, pred_arity);
      }
   };
   collect_predicates(mimir::formalism::StaticTag{});
   collect_predicates(mimir::formalism::FluentTag{});
   collect_predicates(mimir::formalism::DerivedTag{});

   if(includes_plain_goal_derivation(result.goal_derivations)) {
      for(const auto& [name, pred_arity] : regular_predicates) {
         for(int level = 0; level <= result.max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               result.arity[RelationFormatter::format_predicate(
                  name, goal_level, std::nullopt, polarity
               )] = pred_arity;
            }
         }
         if(result.support_literals) {
            for(bool polarity : {true, false}) {
               result.arity[RelationFormatter::format_predicate(
                  name, std::nullopt, std::nullopt, polarity
               )] = pred_arity;
            }
         }
      }
   }

   for(const auto derivation : goal_satisfaction_derivations(result.goal_derivations)) {
      for(const auto& [name, pred_arity] : regular_predicates) {
         for(int level = 0; level <= result.max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               result.arity[RelationFormatter::format_predicate(
                  name, goal_level, derivation, polarity
               )] = pred_arity;
            }
         }
         if(result.support_literals) {
            for(bool polarity : {true, false}) {
               result.arity[RelationFormatter::format_predicate(
                  name, std::nullopt, derivation, polarity
               )] = pred_arity;
            }
         }
      }
   }

   for(const auto& action : actions) {
      const int act_arity = std::max(
         0, static_cast< int >(action->get_arity()) + action_arity_offset
      );
      result.arity[RelationFormatter::format_action_schema(*action)] = act_arity;
   }

   return result;
}

}  // namespace mifrost
