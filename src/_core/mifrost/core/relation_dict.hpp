#pragma once

#include <map>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/predicate.hpp>
#include <set>
#include <string>
#include <vector>

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
      "_symbol_",
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

   /**
    * @brief Build a relation dictionary from a domain and action list.
    */
   static RelationDict build(
      const mimir::formalism::DomainImpl& domain,
      const std::vector< mimir::formalism::Action >& actions,
      const RelationDictConfig& config
   )
   {
      RelationDict out;
      out.max_goal_level = std::max(0, config.max_goal_level);
      out.support_literals = config.support_literals;
      out.goal_satisfaction_derivations = config.goal_satisfaction_derivations;
      out.goal_satisfaction_derivations.insert(GoalSatisfaction::none);

      std::vector< std::pair< std::string, int > > regular_predicates;
      auto collect_predicates = [&]([[maybe_unused]] auto tag) {
         using Tag = decltype(tag);
         for(auto pred : domain.get_predicates< Tag >()) {
            const auto& name = pred->get_name();
            const int arity = static_cast< int >(pred->get_arity());
            out.arity[RelationFormatter::format_predicate(pred)] = arity;
            if(config.top_type_predicates.contains(name)) {
               continue;
            }
            regular_predicates.emplace_back(name, arity);
         }
      };
      collect_predicates(mimir::formalism::StaticTag{});
      collect_predicates(mimir::formalism::FluentTag{});
      collect_predicates(mimir::formalism::DerivedTag{});

      for(const auto& [name, arity] : regular_predicates) {
         for(int level = 0; level <= out.max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               const std::string rel = RelationFormatter::format_predicate(
                  name, goal_level, std::nullopt, polarity
               );
               out.arity[rel] = arity;
            }
         }
         if(out.support_literals) {
            for(bool polarity : {true, false}) {
               const std::string rel = RelationFormatter::format_predicate(
                  name, std::nullopt, std::nullopt, polarity
               );
               out.arity[rel] = arity;
            }
         }
      }

      for(const auto& goal_sat : out.goal_satisfaction_derivations) {
         for(const auto& [name, arity] : regular_predicates) {
            for(int level = 0; level <= out.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(bool polarity : {true, false}) {
                  const std::string rel = RelationFormatter::format_predicate(
                     name, goal_level, goal_sat, polarity
                  );
                  out.arity[rel] = arity;
               }
            }
            if(out.support_literals) {
               for(bool polarity : {true, false}) {
                  const std::string rel = RelationFormatter::format_predicate(
                     name, std::nullopt, goal_sat, polarity
                  );
                  out.arity[rel] = arity;
               }
            }
         }
      }

      for(const auto& action : actions) {
         const int arity = static_cast< int >(action->get_arity()) + 1;
         out.arity[RelationFormatter::format_action_schema(*action)] = arity;
      }

      return out;
   }
};

}  // namespace mifrost
