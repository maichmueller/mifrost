/**
 * @file relation_dict_types.hpp
 * @brief Backend-neutral relation dictionary records.
 */
#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>

#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/goal_derivation.hpp"

namespace mifrost {

struct RelationDictConfig {
   int max_goal_level = 0;
   bool support_literals = false;
   std::set< std::string > top_type_predicates = {
      "object",
      "number",
      defaults::symbol_type_id,
      "_action_",
   };
   std::set< GoalDerivation > goal_derivations = {
      GoalDerivation::plain,
      GoalDerivation::satisfied,
   };
};

struct RelationDict {
   std::map< std::string, int > arity;
   int max_goal_level = 0;
   bool support_literals = false;
   std::set< GoalDerivation > goal_derivations;

   RelationDict() = default;

   explicit RelationDict(std::map< std::string, int > arity_) : arity(std::move(arity_)) {}

   RelationDict(
      std::map< std::string, int > arity_,
      int max_goal_level_,
      bool support_literals_,
      std::set< GoalDerivation > goal_derivations_
   )
       : arity(std::move(arity_)),
         max_goal_level(max_goal_level_),
         support_literals(support_literals_),
         goal_derivations(std::move(goal_derivations_))
   {
   }
};

}  // namespace mifrost
