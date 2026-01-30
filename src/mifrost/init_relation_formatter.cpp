#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>
#include <string_view>

#include "mifrost/bindings.hpp"
#include "mifrost/core/relation_formatter.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {
namespace {

template < typename FormatFn >
std::string format_predicate_optional(
   const std::string& name,
   const std::optional< int >& goal_level,
   const std::optional< GoalSatisfaction >& satisfaction,
   const std::optional< bool >& polarity,
   const std::string_view suffix,
   FormatFn&& fn
)
{
   auto call_with_level = [&](auto level_arg) {
      auto call_with_satisfaction = [&](auto satisfaction_arg) {
         auto call_with_polarity = [&](auto polarity_arg) {
            return fn(name, level_arg, satisfaction_arg, polarity_arg, suffix);
         };
         if(polarity.has_value()) {
            return call_with_polarity(*polarity);
         }
         return call_with_polarity(std::nullopt);
      };
      if(satisfaction.has_value()) {
         return call_with_satisfaction(*satisfaction);
      }
      return call_with_satisfaction(std::nullopt);
   };

   if(goal_level.has_value()) {
      return call_with_level(GoalLevel(*goal_level));
   }
   return call_with_level(std::nullopt);
}

template < typename Literal >
std::string format_literal_optional(
   Literal literal,
   const std::optional< int >& goal_level,
   const std::optional< GoalSatisfaction >& satisfaction,
   const std::optional< bool >& polarity,
   const std::string& suffix
)
{
   return format_predicate_optional(
      std::string(),
      goal_level,
      satisfaction,
      polarity,
      suffix,
      [&](
         const std::string&,
         auto level_arg,
         auto satisfaction_arg,
         auto polarity_arg,
         const std::string_view suffix_arg
      ) {
         return RelationFormatter::format_literal(
            literal, level_arg, satisfaction_arg, polarity_arg, suffix_arg
         );
      }
   );
}

}  // namespace

void init_relation_formatter(nb::module_& m)
{
   nb::enum_< GoalSatisfaction >(m, "GoalSatisfaction")
      .value("none", GoalSatisfaction::none)
      .value("satisfied", GoalSatisfaction::satisfied)
      .value("unsatisfied", GoalSatisfaction::unsatisfied)
      .value("added_satisfied", GoalSatisfaction::added_satisfied)
      .value("added_unsatisfied", GoalSatisfaction::added_unsatisfied);

   nb::class_< RelationFormatter >(m, "RelationFormatter")
      .def_static(
         "format_predicate",
         [](const std::string& name,
            const std::optional< int >& goal_level,
            const std::optional< GoalSatisfaction >& satisfaction,
            const std::optional< bool >& polarity,
            const std::string_view suffix = "") {
            return format_predicate_optional(
               name,
               goal_level,
               satisfaction,
               polarity,
               suffix,
               [](const std::string& name_arg,
                  auto level_arg,
                  auto satisfaction_arg,
                  auto polarity_arg,
                  const std::string_view suffix_arg) {
                  return RelationFormatter::format_predicate(
                     name_arg, level_arg, satisfaction_arg, polarity_arg, suffix_arg
                  );
               }
            );
         },
         "name"_a,
         "goal_level"_a = std::nullopt,
         "satisfaction"_a = std::nullopt,
         "polarity"_a = std::nullopt,
         "suffix"_a = ""
      )
      .def_static(
         "format_literal",
         [](const mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >& literal,
            const std::optional< int >& goal_level,
            const std::optional< GoalSatisfaction >& satisfaction,
            const std::optional< bool >& polarity,
            const std::string& suffix) {
            return format_literal_optional(literal, goal_level, satisfaction, polarity, suffix);
         },
         "literal"_a,
         "goal_level"_a = std::nullopt,
         "satisfaction"_a = std::nullopt,
         "polarity"_a = std::nullopt,
         "suffix"_a = ""
      )
      .def_static(
         "format_literal",
         [](const mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag >& literal,
            const std::optional< int >& goal_level,
            const std::optional< GoalSatisfaction >& satisfaction,
            const std::optional< bool >& polarity,
            const std::string& suffix) {
            return format_literal_optional(literal, goal_level, satisfaction, polarity, suffix);
         },
         "literal"_a,
         "goal_level"_a = std::nullopt,
         "satisfaction"_a = std::nullopt,
         "polarity"_a = std::nullopt,
         "suffix"_a = ""
      )
      .def_static(
         "format_literal",
         [](const mimir::formalism::GroundLiteral< mimir::formalism::StaticTag >& literal,
            const std::optional< int >& goal_level,
            const std::optional< GoalSatisfaction >& satisfaction,
            const std::optional< bool >& polarity,
            const std::string& suffix) {
            return format_literal_optional(literal, goal_level, satisfaction, polarity, suffix);
         },
         "literal"_a,
         "goal_level"_a = std::nullopt,
         "satisfaction"_a = std::nullopt,
         "polarity"_a = std::nullopt,
         "suffix"_a = ""
      )
      .def_static("format_action_schema", &RelationFormatter::format_action_schema, "action"_a)
      .def_static("format_action", &RelationFormatter::format_action, "action"_a)
      .def_static("format_object", &RelationFormatter::format_object, "object"_a);
}

}  // namespace mifrost
