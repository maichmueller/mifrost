#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "mifrost/bindings.hpp"
#include "mifrost/core/transition_dag.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

void init_transition_dag(nb::module_& m)
{
   nb::class_< TransitionDAG >(m, "TransitionDAG")
      .def(nb::init< mimir::search::State >(), "root"_a)
      .def(
         "register_transition",
         &TransitionDAG::register_transition,
         "parent"_a,
         "child"_a,
         "action"_a = std::nullopt,
         nb::rv_policy::copy
      )
      .def("index", &TransitionDAG::index, "state"_a)
      .def("depth", &TransitionDAG::depth, "idx"_a)
      .def("action", &TransitionDAG::action, "idx"_a)
      .def("state", &TransitionDAG::state, "idx"_a)
      .def("children", &TransitionDAG::children, "parent_idx"_a)
      .def("nodes", &TransitionDAG::nodes, nb::rv_policy::reference_internal)
      .def("successors", &TransitionDAG::successors)
      .def("transitions", &TransitionDAG::transitions)
      .def("root", &TransitionDAG::root)
      .def("root_index", &TransitionDAG::root_index)
      .def("contains", &TransitionDAG::contains, "state"_a);

   nb::class_< TransitionDAG::Node >(m, "TransitionNode")
      .def_ro("state", &TransitionDAG::Node::state)
      .def_ro("index", &TransitionDAG::Node::index)
      .def_ro("depth", &TransitionDAG::Node::depth)
      .def_ro("action", &TransitionDAG::Node::action);
}

}  // namespace mifrost
