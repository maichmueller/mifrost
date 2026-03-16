#pragma once

namespace mifrost {

enum class RootPolicy {
   Include,
   EncodeOnly,
   Exclude,
};

inline constexpr bool root_in_target_metadata(const RootPolicy policy)
{
   return policy == RootPolicy::Include;
}

inline constexpr bool root_in_public_carrier(const RootPolicy policy)
{
   return policy != RootPolicy::Exclude;
}

inline constexpr bool root_in_state_relations(const RootPolicy policy)
{
   return policy != RootPolicy::Exclude;
}

inline constexpr bool root_uses_split_state_relations(const RootPolicy policy)
{
   return policy == RootPolicy::Exclude;
}

}  // namespace mifrost
