/**
 * @file flat_relation_keys.hpp
 * @brief Mimir-typed structured relation-key builders for the flat backend.
 *
 * Thin forwarders onto the backend-neutral `RelationKey` builders in
 * `mifrost/core/encoders/common/relation_key.hpp`. Schema declaration and relation emission both
 * build keys through these same functions, so the two can never independently drift on the
 * exported relation name.
 */
#pragma once

#include <mimir/formalism/action.hpp>
#include <mimir/formalism/predicate.hpp>
#include <optional>
#include <string_view>

#include "mifrost/core/encoders/common/relation_key.hpp"

namespace mifrost {

/// Build a structured predicate/goal relation key from a Mimir predicate handle.
template < typename Tag >
RelationKey predicate_relation_key(
   const mimir::formalism::Predicate< Tag > predicate,
   std::optional< bool > polarity = std::nullopt,
   std::optional< GoalLevel > goal_level = std::nullopt,
   std::optional< GoalDerivation > derivation = std::nullopt,
   std::string_view modifier = "",
   bool state_anchored = false
)
{
   return predicate_relation_key(
      std::string_view(predicate->get_name()),
      polarity,
      goal_level,
      derivation,
      modifier,
      state_anchored
   );
}

/// Build an action-schema relation key from a Mimir action.
inline RelationKey action_schema_relation_key(const mimir::formalism::ActionImpl& action)
{
   return action_relation_key(std::string_view(action.get_name()));
}

}  // namespace mifrost
