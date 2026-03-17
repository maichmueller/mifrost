#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "target_source.hpp"

namespace mifrost {

constexpr std::string_view kEntityRoleIdsField = "entity_role_ids";
constexpr std::string_view kEntityRoleNamesAttr = "entity_role_names";
constexpr std::string_view kRelationLogicalAritiesAttr = "relation_logical_arities";
constexpr std::string_view kRelationEncodedAritiesAttr = "relation_encoded_arities";
constexpr std::string_view kRelationSlotRolesAttr = "relation_slot_roles";
constexpr std::string_view kRelationSlotRoleOffsetsAttr = "relation_slot_role_offsets";
constexpr std::string_view kSlotRoleNamesAttr = "slot_role_names";
constexpr std::string_view kUsePredicateVirtualNodesAttr = "use_predicate_virtual_nodes";

enum class FlatSlotRole : int64_t {
   argument_slot = 0,
   predicate_slot = 1,
   state_slot = 2,
   action_slot = 3,
   goal_target_slot = 4,
   subgoal_target_slot = 5,
   history_target_slot = 6,
   history_slot = 7,
};

enum class FlatEntityRole : int64_t {
   object = 0,
   predicate_virtual = 1,
   goal = 2,
   subgoal = 3,
   action = 4,
   history = 5,
   history_target = 6,
   state = 7,
};

inline constexpr std::array< std::string_view, 8 > kFlatSlotRoleNames = {
   "argument_slot",
   "predicate_slot",
   "state_slot",
   "action_slot",
   "goal_target_slot",
   "subgoal_target_slot",
   "history_target_slot",
   "history_slot",
};

inline constexpr std::array< std::string_view, 8 > kFlatEntityRoleNames = {
   "object",
   "predicate_virtual",
   "goal",
   "subgoal",
   "action",
   "history",
   "history_target",
   "state",
};

struct FlatTupleLayout {
   int logical_arity = 0;
   std::vector< FlatSlotRole > auxiliary_slot_roles;
   bool include_predicate_virtual_node = false;

   [[nodiscard]] int encoded_arity() const
   {
      return logical_arity + static_cast< int >(auxiliary_slot_roles.size())
             + static_cast< int >(include_predicate_virtual_node);
   }

   [[nodiscard]] std::vector< int64_t > slot_role_ids() const
   {
      std::vector< int64_t > out;
      out.reserve(static_cast< size_t >(encoded_arity()));
      for(const auto role : auxiliary_slot_roles) {
         out.push_back(static_cast< int64_t >(role));
      }
      if(include_predicate_virtual_node) {
         out.push_back(static_cast< int64_t >(FlatSlotRole::predicate_slot));
      }
      for(int idx = 0; idx < logical_arity; ++idx) {
         out.push_back(static_cast< int64_t >(FlatSlotRole::argument_slot));
      }
      return out;
   }
};

inline FlatTupleLayout make_predicate_tuple_layout(
   int logical_arity,
   std::span< const FlatSlotRole > auxiliary_slot_roles,
   bool use_predicate_virtual_nodes
)
{
   FlatTupleLayout layout;
   layout.logical_arity = logical_arity;
   layout.auxiliary_slot_roles.assign(auxiliary_slot_roles.begin(), auxiliary_slot_roles.end());
   layout.include_predicate_virtual_node = use_predicate_virtual_nodes;
   return layout;
}

inline FlatTupleLayout make_predicate_tuple_layout(
   int logical_arity,
   std::initializer_list< FlatSlotRole > auxiliary_slot_roles,
   bool use_predicate_virtual_nodes
)
{
   return make_predicate_tuple_layout(
      logical_arity,
      std::span< const FlatSlotRole >(auxiliary_slot_roles.begin(), auxiliary_slot_roles.size()),
      use_predicate_virtual_nodes
   );
}

inline FlatTupleLayout make_nonpredicate_tuple_layout(
   int logical_arity,
   std::span< const FlatSlotRole > auxiliary_slot_roles
)
{
   FlatTupleLayout layout;
   layout.logical_arity = logical_arity;
   layout.auxiliary_slot_roles.assign(auxiliary_slot_roles.begin(), auxiliary_slot_roles.end());
   layout.include_predicate_virtual_node = false;
   return layout;
}

inline FlatTupleLayout make_nonpredicate_tuple_layout(
   int logical_arity,
   std::initializer_list< FlatSlotRole > auxiliary_slot_roles
)
{
   return make_nonpredicate_tuple_layout(
      logical_arity,
      std::span< const FlatSlotRole >(auxiliary_slot_roles.begin(), auxiliary_slot_roles.size())
   );
}

inline std::vector< int64_t > build_flat_tuple_args(
   std::span< const int64_t > logical_args,
   std::span< const int64_t > auxiliary_args,
   std::optional< int64_t > predicate_virtual_index
)
{
   std::vector< int64_t > out;
   out.reserve(
      auxiliary_args.size() + logical_args.size() + (predicate_virtual_index.has_value() ? 1U : 0U)
   );
   out.insert(out.end(), auxiliary_args.begin(), auxiliary_args.end());
   if(predicate_virtual_index.has_value()) {
      out.push_back(*predicate_virtual_index);
   }
   out.insert(out.end(), logical_args.begin(), logical_args.end());
   return out;
}

inline FlatSlotRole slot_role_for_target_source(TargetSource source)
{
   switch(source) {
      case TargetSource::goals: return FlatSlotRole::goal_target_slot;
      case TargetSource::subgoals: return FlatSlotRole::subgoal_target_slot;
      case TargetSource::history: return FlatSlotRole::history_target_slot;
      case TargetSource::actions: return FlatSlotRole::action_slot;
      case TargetSource::states: return FlatSlotRole::state_slot;
   }
   return FlatSlotRole::argument_slot;
}

inline FlatEntityRole entity_role_for_target_source(TargetSource source)
{
   switch(source) {
      case TargetSource::goals: return FlatEntityRole::goal;
      case TargetSource::subgoals: return FlatEntityRole::subgoal;
      case TargetSource::history: return FlatEntityRole::history_target;
      case TargetSource::actions: return FlatEntityRole::action;
      case TargetSource::states: return FlatEntityRole::state;
   }
   return FlatEntityRole::object;
}

inline std::vector< std::string > flat_slot_role_names()
{
   std::vector< std::string > out;
   out.reserve(kFlatSlotRoleNames.size());
   for(const auto name : kFlatSlotRoleNames) {
      out.emplace_back(name);
   }
   return out;
}

inline std::vector< std::string > flat_entity_role_names()
{
   std::vector< std::string > out;
   out.reserve(kFlatEntityRoleNames.size());
   for(const auto name : kFlatEntityRoleNames) {
      out.emplace_back(name);
   }
   return out;
}

}  // namespace mifrost
