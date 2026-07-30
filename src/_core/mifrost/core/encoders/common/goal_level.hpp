/**
 * @file goal_level.hpp
 * @brief Backend-neutral strong type for goal layer index.
 */
#pragma once

#include <concepts>
#include <cstddef>
#include <strong_type/strong_type.hpp>

#include "mifrost/core/utils/type_traits.hpp"

namespace mifrost {

/**
 * @brief Strong type for goal layer index.
 */
struct GoalLevel: strong::type< std::size_t, GoalLevel, strong::regular > {
   using base = strong::type< std::size_t, GoalLevel, strong::regular >;
   using base::base;  // keep the normal constructors

   template < std::integral I >
      requires(not std::same_as< detail::raw_t< I >, bool >)
   explicit constexpr GoalLevel(I v) : base(static_cast< std::size_t >(v))
   {
   }
};

}  // namespace mifrost
