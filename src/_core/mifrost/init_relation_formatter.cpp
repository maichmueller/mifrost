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

std::string format_predicate_optional(
   const std::string& name,
   const std::optional< int >& goal_level,
   const std::optional< GoalSatisfaction >& satisfaction,
   const std::optional< bool >& polarity,
   const std::string& suffix
)
{
   if(goal_level.has_value()) {
      const GoalLevel level(*goal_level);
      if(satisfaction.has_value()) {
         if(polarity.has_value()) {
            return RelationFormatter::format_predicate(
               name, level, *satisfaction, *polarity, suffix
            );
         }
         return RelationFormatter::format_predicate(
            name, level, *satisfaction, std::nullopt, suffix
         );
      }
      if(polarity.has_value()) {
         return RelationFormatter::format_predicate(name, level, std::nullopt, *polarity, suffix);
      }
      return RelationFormatter::format_predicate(name, level, std::nullopt, std::nullopt, suffix);
   }

   if(satisfaction.has_value()) {
      if(polarity.has_value()) {
         return RelationFormatter::format_predicate(
            name, std::nullopt, *satisfaction, *polarity, suffix
         );
      }
      return RelationFormatter::format_predicate(
         name, std::nullopt, *satisfaction, std::nullopt, suffix
      );
   }
   if(polarity.has_value()) {
      return RelationFormatter::format_predicate(
         name, std::nullopt, std::nullopt, *polarity, suffix
      );
   }
   return RelationFormatter::format_predicate(
      name, std::nullopt, std::nullopt, std::nullopt, suffix
   );
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
   if(goal_level.has_value()) {
      const GoalLevel level(*goal_level);
      if(satisfaction.has_value()) {
         if(polarity.has_value()) {
            return RelationFormatter::format_literal(
               literal, level, *satisfaction, *polarity, suffix
            );
         }
         return RelationFormatter::format_literal(
            literal, level, *satisfaction, std::nullopt, suffix
         );
      }
      if(polarity.has_value()) {
         return RelationFormatter::format_literal(literal, level, std::nullopt, *polarity, suffix);
      }
      return RelationFormatter::format_literal(literal, level, std::nullopt, std::nullopt, suffix);
   }

   if(satisfaction.has_value()) {
      if(polarity.has_value()) {
         return RelationFormatter::format_literal(
            literal, std::nullopt, *satisfaction, *polarity, suffix
         );
      }
      return RelationFormatter::format_literal(
         literal, std::nullopt, *satisfaction, std::nullopt, suffix
      );
   }
   if(polarity.has_value()) {
      return RelationFormatter::format_literal(
         literal, std::nullopt, std::nullopt, *polarity, suffix
      );
   }
   return RelationFormatter::format_literal(
      literal, std::nullopt, std::nullopt, std::nullopt, suffix
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
         [](const mimir::formalism::Predicate< mimir::formalism::StaticTag >& predicate,
            const std::optional< int >& goal_level,
            const std::optional< GoalSatisfaction >& satisfaction,
            const std::optional< bool >& polarity,
            const std::string& suffix = "") {
            return format_predicate_optional(
               predicate->get_name(), goal_level, satisfaction, polarity, suffix
            );
         },
         "predicate"_a,
         "goal_level"_a = std::nullopt,
         "satisfaction"_a = std::nullopt,
         "polarity"_a = std::nullopt,
         "suffix"_a = ""
      )
      .def_static(
         "format_predicate",
         [](const mimir::formalism::Predicate< mimir::formalism::FluentTag >& predicate,
            const std::optional< int >& goal_level,
            const std::optional< GoalSatisfaction >& satisfaction,
            const std::optional< bool >& polarity,
            const std::string& suffix = "") {
            return format_predicate_optional(
               predicate->get_name(), goal_level, satisfaction, polarity, suffix
            );
         },
         "predicate"_a,
         "goal_level"_a = std::nullopt,
         "satisfaction"_a = std::nullopt,
         "polarity"_a = std::nullopt,
         "suffix"_a = ""
      )
      .def_static(
         "format_predicate",
         [](const mimir::formalism::Predicate< mimir::formalism::DerivedTag >& predicate,
            const std::optional< int >& goal_level,
            const std::optional< GoalSatisfaction >& satisfaction,
            const std::optional< bool >& polarity,
            const std::string& suffix = "") {
            return format_predicate_optional(
               predicate->get_name(), goal_level, satisfaction, polarity, suffix
            );
         },
         "predicate"_a,
         "goal_level"_a = std::nullopt,
         "satisfaction"_a = std::nullopt,
         "polarity"_a = std::nullopt,
         "suffix"_a = ""
      )
      .def_static(
         "format_predicate",
         [](const std::string& name,
            const std::optional< int >& goal_level,
            const std::optional< GoalSatisfaction >& satisfaction,
            const std::optional< bool >& polarity,
            const std::string& suffix = "") {
            return format_predicate_optional(name, goal_level, satisfaction, polarity, suffix);
         },
         "name"_a,
         "goal_level"_a = std::nullopt,
         "satisfaction"_a = std::nullopt,
         "polarity"_a = std::nullopt,
         "suffix"_a = ""
      )
      .def_static(
         "format_atom",
         [](const mimir::formalism::GroundAtom< mimir::formalism::StaticTag >& atom,
            const std::string& suffix) { return RelationFormatter::format_atom(atom, suffix); },
         "atom"_a,
         "suffix"_a = ""
      )
      .def_static(
         "format_atom",
         [](const mimir::formalism::GroundAtom< mimir::formalism::FluentTag >& atom,
            const std::string& suffix) { return RelationFormatter::format_atom(atom, suffix); },
         "atom"_a,
         "suffix"_a = ""
      )
      .def_static(
         "format_atom",
         [](const mimir::formalism::GroundAtom< mimir::formalism::DerivedTag >& atom,
            const std::string& suffix) { return RelationFormatter::format_atom(atom, suffix); },
         "atom"_a,
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
