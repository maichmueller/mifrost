#pragma once

#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace mifrost {

template < typename T >
T* require_instance_ptr(nanobind::handle self, std::string_view null_message)
{
   auto* ptr = nanobind::inst_ptr< T >(self);
   if(ptr == nullptr) {
      throw std::invalid_argument(std::string(null_message));
   }
   return ptr;
}

}  // namespace mifrost
