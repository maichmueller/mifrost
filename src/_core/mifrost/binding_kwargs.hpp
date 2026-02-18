#pragma once

#include <fmt/format.h>
#include <nanobind/nanobind.h>

#include <boost/describe.hpp>
#include <boost/mp11.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace mifrost {

namespace nb = nanobind;

template < typename Config >
bool try_set_config_kwarg(Config& config, std::string_view key, const nb::handle value)
{
   bool matched = false;
   using described_members = boost::describe::
      describe_members< Config, boost::describe::mod_public | boost::describe::mod_inherited >;

   boost::mp11::mp_for_each< described_members >([&](auto descriptor) {
      using desc_t = decltype(descriptor);
      if(matched or key != desc_t::name) {
         return;
      }
      using member_t = std::remove_cv_t<
         std::remove_reference_t< decltype(config.*desc_t::pointer) > >;
      if constexpr(std::is_same_v< member_t, std::string >) {
         config.*desc_t::pointer = nb::str(value).c_str();
      } else {
         config.*desc_t::pointer = nb::cast< member_t >(value);
      }
      matched = true;
   });

   return matched;
}

template < typename Config >
void apply_config_kwargs(Config& config, const nb::kwargs& kwargs, std::string_view config_name)
{
   for(const auto& [key_handle, value_handle] : kwargs) {
      const auto key = nb::str(key_handle);
      if(not try_set_config_kwarg(config, key.c_str(), value_handle)) {
         throw std::invalid_argument(
            fmt::format("Unknown {} kwarg '{}'", config_name, key.c_str())
         );
      }
   }
}

}  // namespace mifrost
