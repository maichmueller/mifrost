/**
 * @file flat_tuple_layout.hpp
 * @brief Slot-role and node-role metadata for flat tuple encoders.
 *
 * This file defines the flat tuple layout. It separates logical arity from
 * encoded arity and says where extra leading entries such as state nodes,
 * target nodes, history nodes, and predicate virtual nodes are inserted.
 */
#pragma once

#include <array>
#include <boost/container/small_vector.hpp>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "mifrost/core/encoders/common/target_source.hpp"

namespace mifrost {

/** Inline tuple storage for the common low-arity PDDL relation case. */
using FlatTupleArguments = boost::container::small_vector< int64_t, 8 >;

constexpr std::string_view kEntityRoleIdsField = "entity_role_ids";
constexpr std::string_view kEntityRoleNamesAttr = "entity_role_names";
constexpr std::string_view kRelationLogicalAritiesAttr = "relation_logical_arities";
constexpr std::string_view kRelationEncodedAritiesAttr = "relation_encoded_arities";
constexpr std::string_view kRelationSlotRolesAttr = "relation_slot_roles";
constexpr std::string_view kRelationSlotRoleOffsetsAttr = "relation_slot_role_offsets";
constexpr std::string_view kSlotRoleNamesAttr = "slot_role_names";
constexpr std::string_view kUsePredicateVirtualNodesAttr = "use_predicate_virtual_nodes";

/**
 * @brief Meaning of one encoded slot in a flat relation tuple.
 *
 * These roles are exported as relation metadata. Models should read them
 * instead of guessing slot meaning from the position alone.
 */
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

/**
 * @brief Type of one row in the flat entity table.
 *
 * Flat graphs still use one node table, so these ids tell the model which rows
 * are objects, targets, history rows, state rows, or predicate virtual nodes.
 */
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

   /// Encoded arity = logical arguments + auxiliary slots + optional predicate slot.
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

/// Build a predicate-atom tuple layout with optional predicate virtual node insertion.
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

/// Build a non-predicate tuple layout (for example action-schema or topology relations).
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

/**
 * @brief Materialize one encoded tuple while preserving logical argument order.
 *
 * Auxiliary arguments always stay in their configured prefix order, followed by
 * the optional predicate virtual node, followed by the original logical object
 * arguments in unchanged order.
 */
inline FlatTupleArguments build_flat_tuple_args(
   std::span< const int64_t > logical_args,
   std::span< const int64_t > auxiliary_args,
   std::optional< int64_t > predicate_virtual_index
)
{
   FlatTupleArguments out;
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

/// Map public target sources to their corresponding flat auxiliary slot role.
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

/// Map public target sources to the entity role used by the flat node row.
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

/// Materialize exported slot-role names in enum order.
inline std::vector< std::string > flat_slot_role_names()
{
   std::vector< std::string > out;
   out.reserve(kFlatSlotRoleNames.size());
   for(const auto name : kFlatSlotRoleNames) {
      out.emplace_back(name);
   }
   return out;
}

/// Materialize exported entity-role names in enum order.
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
