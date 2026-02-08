
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include <optional>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/default_relations.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

void init_common(nb::module_& m)
{
   m.attr("DEFAULT_SYMBOL_TYPE_ID") = defaults::symbol_type_id;
   m.attr("DEFAULT_LGAN_NN_EDGE_POS") = defaults::lgan_nn_edge_pos;
   m.attr("DEFAULT_PARENT_RELATION") = defaults::parent_relation;
   m.attr("DEFAULT_SIBLING_RELATION") = defaults::sibling_relation;
   m.attr("DEFAULT_COUSIN_RELATION") = defaults::cousin_relation;
   m.attr("DEFAULT_HISTORY_LINK_RELATION") = defaults::history_link_relation;

   nb::class_< GoalInputs >(m, "GoalInputs")
      .def(nb::init<>())
      .def(
         "__init__",
         [](GoalInputs* self, nb::iterable goals, const size_t level) {
            new(self) GoalInputs();
            self->extend(
               goals | std::views::transform(AS_LAMBDA(nb::cast< LiteralVariant >)), level
            );
         },
         "goals"_a,
         "level"_a = 0ul
      )
      .def_rw("static_goals", &GoalInputs::static_goals)
      .def_rw("fluent_goals", &GoalInputs::fluent_goals)
      .def_rw("derived_goals", &GoalInputs::derived_goals)
      .def_rw("static_goal_levels", &GoalInputs::static_goal_levels)
      .def_rw("fluent_goal_levels", &GoalInputs::fluent_goal_levels)
      .def_rw("derived_goal_levels", &GoalInputs::derived_goal_levels)
      .def(
         "extend",
         [](GoalInputs& self, nb::iterable goals, const size_t level) {
            self.extend(
               goals | std::views::transform(AS_LAMBDA(nb::cast< LiteralVariant >)), level
            );
         },
         "goals"_a,
         "level"_a = 0ul
      )
      .def("append", [](GoalInputs& self, const nb::object& goal, const size_t level) {
         if(nb::isinstance< FluentLiteral >(goal)) {
            self.append(nb::cast< FluentLiteral >(goal), level);
         } else if(nb::isinstance< DerivedLiteral >(goal)) {
            self.append(nb::cast< DerivedLiteral >(goal), level);
         } else if(nb::isinstance< StaticLiteral >(goal)) {
            self.append(nb::cast< StaticLiteral >(goal), level);
         } else {
            throw std::invalid_argument(
               "No known literal type passed. Expected one of "
               "'FluentLiteral', 'DerivedLiteral', or 'StaticLiteral'."
            );
         }
      });
}

}  // namespace mifrost
