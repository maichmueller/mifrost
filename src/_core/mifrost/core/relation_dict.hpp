#pragma once

#include <algorithm>
#include <map>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/predicate.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "default_relations.hpp"
#include "relation_formatter.hpp"

namespace mifrost {

/**
 * @brief Configuration for relation dictionary derivation from a domain.
 */
struct RelationDictConfig {
   int max_goal_level = 0;
   bool support_literals = false;
   std::set< std::string > top_type_predicates = {
      "object",
      "number",
      defaults::symbol_type_id,
      "_action_",
   };
   std::set< GoalSatisfaction > goal_satisfaction_derivations = {
      GoalSatisfaction::satisfied,
   };
};

/**
 * @brief Relation arity table used by encoders to enforce relation contracts.
 *
 * The dictionary expands base domain predicates with derived relation variants
 * (goal levels, polarities, satisfaction suffixes) according to config.
 */
struct RelationDict {
   /// Relation key -> arity.
   std::map< std::string, int > arity;
   /// Highest configured goal level included in derived relation variants.
   int max_goal_level = 0;
   /// Whether literal variants (without goal-level suffix) are included.
   bool support_literals = false;
   /// Satisfaction derivations that are materialized.
   std::set< GoalSatisfaction > goal_satisfaction_derivations;

   RelationDict() = default;

   explicit RelationDict(std::map< std::string, int > arity_) : arity(std::move(arity_)) {}

   RelationDict(
      std::map< std::string, int > arity_,
      int max_goal_level_,
      bool support_literals_,
      std::set< GoalSatisfaction > goal_satisfaction_derivations_
   )
       : arity(std::move(arity_)),
         max_goal_level(max_goal_level_),
         support_literals(support_literals_),
         goal_satisfaction_derivations(std::move(goal_satisfaction_derivations_))
   {
   }

   /**
    * @brief Build a relation dictionary from a domain and action list.
    *
    * @param predicate_arity_offset Applied to all predicate-based relations.
    * @param action_arity_offset Applied to all action-schema relations.
    */
   RelationDict(
      const mimir::formalism::DomainImpl& domain,
      const std::vector< mimir::formalism::Action >& actions,
      const RelationDictConfig& config,
      int predicate_arity_offset = 0,
      int action_arity_offset = 1
   )
   {
      max_goal_level = std::max(0, config.max_goal_level);
      support_literals = config.support_literals;
      goal_satisfaction_derivations = config.goal_satisfaction_derivations;
      goal_satisfaction_derivations.insert(GoalSatisfaction::none);

      std::vector< std::pair< std::string, int > > regular_predicates;
      auto collect_predicates = [&]< typename Tag >(Tag) {
         for(auto pred : domain.get_predicates< Tag >()) {
            const auto& name = pred->get_name();
            const int pred_arity = std::max(
               0, static_cast< int >(pred->get_arity()) + predicate_arity_offset
            );
            arity[RelationFormatter::format_predicate(name)] = pred_arity;
            if(config.top_type_predicates.contains(name)) {
               continue;
            }
            regular_predicates.emplace_back(name, pred_arity);
         }
      };
      collect_predicates(mimir::formalism::StaticTag{});
      collect_predicates(mimir::formalism::FluentTag{});
      collect_predicates(mimir::formalism::DerivedTag{});

      for(const auto& [name, pred_arity] : regular_predicates) {
         for(int level = 0; level <= max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               arity[RelationFormatter::format_predicate(
                  name, goal_level, std::nullopt, polarity
               )] = pred_arity;
            }
         }
         if(support_literals) {
            for(bool polarity : {true, false}) {
               arity[RelationFormatter::format_predicate(
                  name, std::nullopt, std::nullopt, polarity
               )] = pred_arity;
            }
         }
      }

      for(const auto& goal_sat : goal_satisfaction_derivations) {
         for(const auto& [name, pred_arity] : regular_predicates) {
            for(int level = 0; level <= max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(bool polarity : {true, false}) {
                  arity[RelationFormatter::format_predicate(
                     name, goal_level, goal_sat, polarity
                  )] = pred_arity;
               }
            }
            if(support_literals) {
               for(bool polarity : {true, false}) {
                  arity[RelationFormatter::format_predicate(
                     name, std::nullopt, goal_sat, polarity
                  )] = pred_arity;
               }
            }
         }
      }

      for(const auto& action : actions) {
         const int act_arity = std::max(
            0, static_cast< int >(action->get_arity()) + action_arity_offset
         );
         arity[RelationFormatter::format_action_schema(*action)] = act_arity;
      }
   }
};

}  // namespace mifrost
