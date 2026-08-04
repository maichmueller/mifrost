/**
 * @file ranges.hpp
 * @brief Explicit lazy range adapters for operation-bearing planning Views.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <utility>

namespace mifrost::views {

/**
 * A lazy transform range whose dereference returns a View value.
 *
 * The native range and its context are retained by value. Consequently the
 * returned range must not outlive the backend object referenced by the View
 * values it produces.
 */
template < std::ranges::input_range NativeRange, typename Transform >
class TransformRange {
   using Base = std::ranges::transform_view< std::views::all_t< NativeRange >, Transform >;

  public:
   TransformRange(NativeRange&& native, Transform transform)
       : range_(
            std::views::all(std::forward< NativeRange >(native))
            | std::views::transform(std::move(transform))
         )
   {
   }

   auto begin() { return range_.begin(); }
   auto begin() const { return range_.begin(); }
   auto end() { return range_.end(); }
   auto end() const { return range_.end(); }

  private:
   Base range_;
};

template < typename NativeRange, typename Transform >
TransformRange(NativeRange&&, Transform) -> TransformRange< NativeRange, Transform >;

}  // namespace mifrost::views
