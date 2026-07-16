
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include <optional>
#include <utility>

#include "mifrost/backends/pymimir/encoders/common/goal_inputs.hpp"
#include "mifrost/backends/pymimir/encoders/hetero/hgraph_stream_encoder.hpp"
#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

void init_common(nb::module_& m)
{
   m.attr("DEFAULT_SYMBOL_TYPE_ID") = defaults::symbol_type_id;
   m.attr("DEFAULT_LGAN_TN_EDGE_POS") = defaults::lgan_tn_edge_pos;
   m.attr("DEFAULT_LGAN_NN_EDGE_POS") = defaults::lgan_nn_edge_pos;
   m.attr("DEFAULT_LGAN_RR_EDGE_POS") = defaults::lgan_rr_edge_pos;
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
      [](nb::object states) {
         if(PyList_Check(states.ptr()) != 0 && PyList_GET_SIZE(states.ptr()) == 0) {
            return nb::list();
         }
         return batch_input::parse_states_batch_python(states);
      },
      "states"_a
   );
   m.def(
      "_parse_goals_batch_param",
      [](nb::object goals, size_t state_count) {
         if(state_count == 0 && PyList_Check(goals.ptr()) != 0
            && PyList_GET_SIZE(goals.ptr()) == 0) {
            return nb::make_tuple(false, nb::list());
         }
         return batch_input::parse_goals_batch_param_python(goals, state_count);
      },
      "goals"_a,
      "state_count"_a
   );
   m.def(
      "_parse_actions_batch_param",
      [](nb::object actions, size_t state_count) {
         if(state_count == 0 && PyList_Check(actions.ptr()) != 0
            && PyList_GET_SIZE(actions.ptr()) == 0) {
            return nb::make_tuple(false, nb::list());
         }
         return batch_input::parse_actions_batch_param_python(actions, state_count);
      },
      "actions"_a,
      "state_count"_a
   );
   m.def(
      "_parse_subgoal_layers_batch_param",
      [](nb::object subgoal_layers, size_t state_count) {
         if(state_count == 0 && PyList_Check(subgoal_layers.ptr()) != 0
            && PyList_GET_SIZE(subgoal_layers.ptr()) == 0) {
            return nb::make_tuple(false, nb::list());
         }
         return batch_input::parse_subgoal_layers_batch_param_python(subgoal_layers, state_count);
      },
      "subgoal_layers"_a,
      "state_count"_a
   );
   m.def(
      "_parse_history_subgoals_batch_param",
      [](nb::object history_subgoals, size_t state_count) {
         if(state_count == 0 && PyList_Check(history_subgoals.ptr()) != 0
            && PyList_GET_SIZE(history_subgoals.ptr()) == 0) {
            return nb::make_tuple(false, nb::list());
         }
         return batch_input::parse_history_subgoals_batch_param_python(
            history_subgoals, state_count
         );
      },
      "history_subgoals"_a,
      "state_count"_a
   );
   m.def(
      "_parse_successors_batch_param",
      [](nb::object successors, size_t state_count) {
         if(state_count == 0 && PyList_Check(successors.ptr()) != 0
            && PyList_GET_SIZE(successors.ptr()) == 0) {
            return nb::list();
         }
         return batch_input::parse_successors_batch_param_python(successors, state_count);
      },
      "successors"_a,
      "state_count"_a
   );
   m.def(
      "_parse_dags_batch_param",
      [](nb::object dags, size_t state_count) {
         if(state_count == 0 && PyList_Check(dags.ptr()) != 0 && PyList_GET_SIZE(dags.ptr()) == 0) {
            return nb::list();
         }
         return batch_input::parse_dags_batch_param_python(dags, state_count);
      },
      "dags"_a,
      "state_count"_a
   );
   m.def(
      "_parse_ilg_batch_inputs",
      [](nb::object states, nb::object goals, nb::object actions, nb::object subgoal_layers) {
         if(PyList_Check(states.ptr()) != 0 && PyList_GET_SIZE(states.ptr()) == 0) {
            return nb::make_tuple(nb::list(), nb::list(), nb::list(), nb::list());
         }
         return batch_input::parse_ilg_batch_inputs_python(states, goals, actions, subgoal_layers);
      },
      "states"_a,
      "goals"_a = nb::none(),
      "actions"_a = nb::none(),
      "subgoal_layers"_a = nb::none()
   );
}

}  // namespace mifrost
