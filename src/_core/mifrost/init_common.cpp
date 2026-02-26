
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include <optional>
#include <utility>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/default_relations.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

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

   m.def(
      "_parse_states_batch",
      [](nb::object states) { return batch_input::parse_states_batch_python(std::move(states)); },
      "states"_a
   );
   m.def(
      "_parse_goals_batch_param",
      [](nb::object goals, size_t state_count) {
         return batch_input::parse_goals_batch_param_python(std::move(goals), state_count);
      },
      "goals"_a,
      "state_count"_a
   );
   m.def(
      "_parse_actions_batch_param",
      [](nb::object actions, size_t state_count) {
         return batch_input::parse_actions_batch_param_python(std::move(actions), state_count);
      },
      "actions"_a,
      "state_count"_a
   );
   m.def(
      "_parse_subgoal_layers_batch_param",
      [](nb::object subgoal_layers, size_t state_count) {
         return batch_input::parse_subgoal_layers_batch_param_python(
            std::move(subgoal_layers), state_count
         );
      },
      "subgoal_layers"_a,
      "state_count"_a
   );
   m.def(
      "_parse_history_subgoals_batch_param",
      [](nb::object history_subgoals, size_t state_count) {
         return batch_input::parse_history_subgoals_batch_param_python(
            std::move(history_subgoals), state_count
         );
      },
      "history_subgoals"_a,
      "state_count"_a
   );
   m.def(
      "_parse_successors_batch_param",
      [](nb::object successors, size_t state_count) {
         return batch_input::parse_successors_batch_param_python(
            std::move(successors), state_count
         );
      },
      "successors"_a,
      "state_count"_a
   );
   m.def(
      "_parse_dags_batch_param",
      [](nb::object dags, size_t state_count) {
         return batch_input::parse_dags_batch_param_python(std::move(dags), state_count);
      },
      "dags"_a,
      "state_count"_a
   );
   m.def(
      "_parse_ilg_batch_inputs",
      [](nb::object states, nb::object goals, nb::object actions, nb::object subgoal_layers) {
         return batch_input::parse_ilg_batch_inputs_python(
            std::move(states), std::move(goals), std::move(actions), std::move(subgoal_layers)
         );
      },
      "states"_a,
      "goals"_a = nb::none(),
      "actions"_a = nb::none(),
      "subgoal_layers"_a = nb::none()
   );
}

}  // namespace mifrost
