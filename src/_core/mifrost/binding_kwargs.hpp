#pragma once

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>

#include <boost/describe.hpp>
#include <boost/mp11.hpp>
#include <cctype>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "mifrost/core/relation_formatter.hpp"
#include "mifrost/core/target_source.hpp"

namespace mifrost {

namespace nb = nanobind;

namespace detail {

inline std::string ascii_lower(std::string_view value)
{
   std::string out;
   out.reserve(value.size());
   for(const unsigned char c : value) {
      out.push_back(static_cast< char >(std::tolower(c)));
   }
   return out;
}

inline std::optional< GoalDerivation > parse_goal_derivation_alias(std::string_view value)
{
   const auto normalized = ascii_lower(value);
   if(normalized.empty() or normalized == "plain" or normalized == "goal" or normalized == "base"
      or normalized == "[g]") {
      return GoalDerivation::plain;
   }
   if(normalized == "true" or normalized == "sat" or normalized == "satisfied"
      or normalized == "[sat]") {
      return GoalDerivation::satisfied;
   }
   if(normalized == "false" or normalized == "unsat" or normalized == "unsatisfied"
      or normalized == "[unsat]") {
      return GoalDerivation::unsatisfied;
   }
   if(normalized == "+" or normalized == "sat+" or normalized == "added_satisfied"
      or normalized == "[sat+]") {
      return GoalDerivation::added_satisfied;
   }
   if(normalized == "-" or normalized == "sat-" or normalized == "added_unsatisfied"
      or normalized == "[sat-]") {
      return GoalDerivation::added_unsatisfied;
   }
   return std::nullopt;
}

inline GoalDerivation cast_goal_derivation(std::string_view key, const nb::handle value)
{
   if(nb::isinstance< nb::str >(value)) {
      const std::string token = nb::str(value).c_str();
      if(const auto parsed = parse_goal_derivation_alias(token); parsed.has_value()) {
         return *parsed;
      }
      throw std::invalid_argument(
         fmt::format(
            "Invalid value '{}' for kwarg '{}'; expected GoalDerivation alias "
            "('plain', 'true', 'false', '+', '-') or GoalDerivation enum.",
            token,
            key
         )
      );
   }
   try {
      return nb::cast< GoalDerivation >(value);
   } catch(const std::exception&) {
      throw std::invalid_argument(
         fmt::format(
            "Invalid value for kwarg '{}'; expected GoalDerivation alias "
            "('plain', 'true', 'false', '+', '-') or GoalDerivation enum.",
            key
         )
      );
   }
}

inline std::optional< TargetSource > parse_target_source_alias(std::string_view value)
{
   const auto normalized = ascii_lower(value);
   if(normalized == "action" or normalized == "actions") {
      return TargetSource::actions;
   }
   if(normalized == "goal" or normalized == "goals") {
      return TargetSource::goals;
   }
   if(normalized == "subgoal" or normalized == "subgoals") {
      return TargetSource::subgoals;
   }
   if(normalized == "state" or normalized == "states") {
      return TargetSource::states;
   }
   if(normalized == "history") {
      return TargetSource::history;
   }
   return std::nullopt;
}

inline TargetSource cast_target_source(std::string_view key, const nb::handle value)
{
   if(nb::isinstance< nb::str >(value)) {
      const std::string token = nb::str(value).c_str();
      if(const auto parsed = parse_target_source_alias(token); parsed.has_value()) {
         return *parsed;
      }
      throw std::invalid_argument(
         fmt::format(
            "Invalid value '{}' for kwarg '{}'; expected TargetSource alias "
            "('action', 'goal', 'subgoal', 'state', 'history') or TargetSource enum.",
            token,
            key
         )
      );
   }
   try {
      return nb::cast< TargetSource >(value);
   } catch(const std::exception&) {
      throw std::invalid_argument(
         fmt::format(
            "Invalid value for kwarg '{}'; expected TargetSource alias "
            "('action', 'goal', 'subgoal', 'state', 'history') or TargetSource enum.",
            key
         )
      );
   }
}

template < typename T >
T cast_config_value(std::string_view key, const nb::handle value)
{
   if constexpr(std::is_same_v< T, std::set< GoalDerivation > >) {
      if(nb::isinstance< nb::iterable >(value) and not nb::isinstance< nb::str >(value)
         and not nb::isinstance< nb::bytes >(value)) {
         std::set< GoalDerivation > out;
         for(nb::handle entry : nb::borrow< nb::object >(value)) {
            out.insert(cast_goal_derivation(key, entry));
         }
         return out;
      }
      return {cast_goal_derivation(key, value)};
   } else if constexpr(std::is_same_v< T, std::set< TargetSource > >) {
      if(nb::isinstance< nb::iterable >(value) and not nb::isinstance< nb::str >(value)
         and not nb::isinstance< nb::bytes >(value)) {
         std::set< TargetSource > out;
         for(nb::handle entry : nb::borrow< nb::object >(value)) {
            out.insert(cast_target_source(key, entry));
         }
         return out;
      }
      return {cast_target_source(key, value)};
   } else if constexpr(std::is_same_v< T, TargetSource >) {
      return cast_target_source(key, value);
   } else {
      return nb::cast< T >(value);
   }
}

}  // namespace detail

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
         config.*desc_t::pointer = detail::cast_config_value< member_t >(key, value);
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
