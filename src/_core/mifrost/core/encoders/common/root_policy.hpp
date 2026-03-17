/**
 * @file root_policy.hpp
 * @brief Rules for whether the DAG root appears in horizon outputs.
 *
 * Horizon encoders use these helpers so root handling stays the same in state
 * rows, target metadata, and emitted relations.
 */
#pragma once

namespace mifrost {

/**
 * @brief Controls how the root state of a transition DAG is shown.
 *
 * `include` keeps the root in public metadata and relations.
 * `encode_only` keeps internal root rows but hides it from target metadata.
 * `exclude` removes the root from public horizon rows.
 */
enum class RootPolicy {
   include,
   encode_only,
   exclude,
};

/// Return whether the root should appear in exported target metadata rows.
inline constexpr bool root_in_target_metadata(const RootPolicy policy)
{
   return policy == RootPolicy::include;
}

/// Return whether the root should stay in public row tables.
inline constexpr bool root_in_public_carrier(const RootPolicy policy)
{
   return policy != RootPolicy::exclude;
}

/// Return whether state-anchored relations should still be emitted for the root.
inline constexpr bool root_in_state_relations(const RootPolicy policy)
{
   return policy != RootPolicy::exclude;
}

/// Return whether the root should be handled separately from successor rows.
inline constexpr bool root_uses_split_state_relations(const RootPolicy policy)
{
   return policy == RootPolicy::exclude;
}

}  // namespace mifrost
