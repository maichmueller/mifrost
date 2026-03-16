#pragma once

namespace mifrost {

enum class RootPolicy {
   include,
   encode_only,
   exclude,
};

inline constexpr bool root_in_target_metadata(const RootPolicy policy)
{
   return policy == RootPolicy::include;
}

inline constexpr bool root_in_public_carrier(const RootPolicy policy)
{
   return policy != RootPolicy::exclude;
}

inline constexpr bool root_in_state_relations(const RootPolicy policy)
{
   return policy != RootPolicy::exclude;
}

inline constexpr bool root_uses_split_state_relations(const RootPolicy policy)
{
   return policy == RootPolicy::exclude;
}

}  // namespace mifrost
