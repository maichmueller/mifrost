/**
 * @file relation_key.hpp
 * @brief Backend-neutral structured identity for one relation.
 *
 * A `RelationKey` is the semantic identity of a relation: what family it
 * belongs to, its base predicate/action name, and the encoding-time
 * decorations (polarity, goal level, goal derivation, modifiers) that
 * distinguish one exported relation from another. It never carries a
 * formatted string. Formatting into the mifrost relation-name convention
 * happens only in `format_relation_name()`, when exporting names or
 * producing diagnostics.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mifrost/core/encoders/common/goal_derivation.hpp"
#include "mifrost/core/encoders/common/goal_level.hpp"

namespace mifrost {

/// Broad category a relation belongs to.
enum class RelationFamily : int8_t {
   predicate,
   action,
   auxiliary,
};

/// Typed replacement for the untyped `source` string previously stored per relation.
enum class RelationUsage : int8_t {
   state,
   goal,
   goal_derivation,
   goal_satisfaction,
   action,
   history,
   parent,
   sibling,
   cousin,
};

/// Exported source-label string for a usage value (matches the pre-existing `source` strings).
[[nodiscard]] std::string_view relation_usage_source_label(RelationUsage usage);

/**
 * @brief Structured relation identity.
 *
 * A key with only `base_name` set (all optionals empty, no modifiers,
 * `state_anchored=false`) is an "opaque" key: `format_relation_name()`
 * degenerates to `base_name` verbatim, which is how already-formatted or
 * fully-configured names (e.g. topology relations) are represented.
 */
struct RelationKey {
   RelationFamily family = RelationFamily::predicate;
   std::string base_name;
   std::optional< bool > polarity;
   std::optional< GoalLevel > goal_level;
   std::optional< GoalDerivation > derivation;
   /// Pre-formed, already-bracketed modifier strings (e.g. "[hist]"), concatenated verbatim, in
   /// order, right after `base_name` and before `goal_level`/`derivation`. Not auto-bracketed.
   std::vector< std::string > modifiers;
   /// Renders "[state]" as the LAST element of the formatted name, after goal level/derivation.
   /// Models the pre-existing horizon "candidate-state-anchored" relation-name convention.
   bool state_anchored = false;

   [[nodiscard]] auto operator==(const RelationKey& other) const -> bool = default;
};

struct RelationKeyHash {
   using is_avalanching = void;

   [[nodiscard]] uint64_t operator()(const RelationKey& key) const noexcept;
};

/**
 * @brief Format a relation key into its exported mifrost relation name.
 *
 * Fixed ordering: polarity + base_name + modifiers... + goal_level + derivation, followed by a
 * trailing "[state]" iff `state_anchored` is set. Every optional/empty field contributes nothing,
 * so an opaque key (only `base_name` set) degenerates to `base_name` verbatim.
 */
[[nodiscard]] std::string format_relation_name(const RelationKey& key);

/// Wrap an already-fully-formatted name as an opaque key (`RelationFamily::auxiliary`).
[[nodiscard]] RelationKey opaque_relation_key(std::string formatted_name);

/// Build a structured predicate/goal relation key from a raw base name.
[[nodiscard]] RelationKey predicate_relation_key(
   std::string_view base_name,
   std::optional< bool > polarity = std::nullopt,
   std::optional< GoalLevel > goal_level = std::nullopt,
   std::optional< GoalDerivation > derivation = std::nullopt,
   std::string_view modifier = "",
   bool state_anchored = false
);

/// Build an action-schema relation key.
[[nodiscard]] RelationKey action_relation_key(std::string_view schema_name);

}  // namespace mifrost
