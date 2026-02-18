#pragma once

#ifndef FWD
   #define FWD(x) std::forward< decltype(x) >(x)
#endif  // FWD

#ifndef AS_LAMBDA
   #define AS_LAMBDA(func) [](auto&&... args) { return func(FWD(args)...); }
#endif  // AS_LAMBDA
/// creates a lambda wrapper around the given function and captures surrounding state by ref
#ifndef AS_CPTR_LAMBDA
   #define AS_CPTR_LAMBDA(func) [&](auto&&... args) { return func(FWD(args)...); }
#endif  // AS_CPTR_LAMBDA

/// creates a lambda wrapper around the given function with perfect-return
/// (avoids copies on return when references are passed back (and supposed to remain references)
#ifndef AS_PRFCT_LAMBDA
   #define AS_PRFCT_LAMBDA(func) [](auto&&... args) -> decltype(auto) { return func(FWD(args)...); }
#endif  // AS_PRFCT_LAMBDA

/// creates a lambda wrapper around the given function with perfect-return and captures surrounding
/// state by ref
#ifndef AS_PRFCT_CPTR_LAMBDA
   #define AS_PRFCT_CPTR_LAMBDA(func) \
      [&](auto&&... args) -> decltype(auto) { return func(FWD(args)...); }
#endif  // AS_PRFCT_CPTR_LAMBDA

#ifndef AS_STRUCT
   #define AS_STRUCT(name, func)                                                       \
      struct name {                                                                    \
         auto operator()(auto&&... args) const noexcept { return func(FWD(args)...); } \
      }
#endif  // AS_STRUCT

#ifndef AS_PRFCT_STRUCT
   #define AS_PRFCT_STRUCT(name, func)                                                           \
      struct name {                                                                              \
         decltype(auto) operator()(auto&&... args) const noexcept { return func(FWD(args)...); } \
      }
#endif  // AS_STRUCT

#ifndef FORCE_DEBUG_ASSERT
   #ifndef NDEBUG
      #define FORCE_DEBUG_ASSERT(expression)                                             \
         if(not(expression)) {                                                           \
            throw force_library_error{"Expression " #expression " evaluated to false."}; \
         }
      #define FORCE_DEBUG_ASSERT_MSG(expression, msg)                                      \
         if(not(expression)) {                                                             \
            throw force_library_error{                                                     \
               fmt::format("Expression " #expression " evaluated to false. Info: {}", msg) \
            };                                                                             \
         }
   #else
      #define FORCE_DEBUG_ASSERT(...)
      #define FORCE_DEBUG_ASSERT_MSG(...)
   #endif
#endif

#ifndef INJECT_STRUCTURED_BINDING_GETTERS
   #define INJECT_STRUCTURED_BINDING_GETTERS(tuple_member) \
      template < size_t N >                                \
      auto& get() const&                                   \
      {                                                    \
         return std::get< N >(tuple_member);               \
      }                                                    \
      template < size_t N >                                \
      auto& get() &                                        \
      {                                                    \
         return std::get< N >(tuple_member);               \
      }                                                    \
                                                           \
      template < size_t N >                                \
      auto&& get() constand                                \
      {                                                    \
         return std::move(std::get< N >(tuple_member));    \
      }                                                    \
      template < size_t N >                                \
      auto&& get() and                                     \
      {                                                    \
         return std::move(std::get< N >(tuple_member));    \
      }                                                    \
                                                           \
      template < typename T >                              \
      auto& get() const&                                   \
      {                                                    \
         return std::get< T >(tuple_member);               \
      }                                                    \
      template < typename T >                              \
      auto& get() &                                        \
      {                                                    \
         return std::get< T >(tuple_member);               \
      }                                                    \
                                                           \
      template < typename T >                              \
      auto&& get() constand                                \
      {                                                    \
         return std::move(std::get< T >(tuple_member));    \
      }                                                    \
      template < typename T >                              \
      auto&& get() and                                     \
      {                                                    \
         return std::move(std::get< T >(tuple_member));    \
      }
#endif  // INJECT_STD_GET_FUNCTIONALITY
