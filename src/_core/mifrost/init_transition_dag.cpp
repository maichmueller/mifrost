#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "mifrost/bindings.hpp"
#include "mifrost/core/transition_dag.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MAKE_OPAQUE(std::vector< mifrost::TransitionDAG::Node >);

namespace mifrost {

void init_transition_dag(nb::module_& m)
{
   nb::bind_vector< std::vector< TransitionDAG::Node > >(m, "TransitionNodeList");

   auto dag_cls = nb::class_< TransitionDAG >(m, "TransitionDAG")
                     .def(nb::init< mimir::search::State >(), "root"_a)
                     .def(
                        "register_transition",
                        [](TransitionDAG& dag,
                           const mimir::search::State& parent,
                           const mimir::search::State& child,
                           const std::optional< mimir::formalism::GroundAction >& action) {
                           return dag.register_transition(parent, child, action);
                        },
                        "parent"_a,
                        "child"_a,
                        "action"_a = std::nullopt
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

   auto node_cls = nb::class_< TransitionDAG::Node >(m, "TransitionNode")
                      .def_ro("state", &TransitionDAG::Node::state)
                      .def_ro("index", &TransitionDAG::Node::index)
                      .def_ro("depth", &TransitionDAG::Node::depth)
                      .def_ro("action", &TransitionDAG::Node::action);

   dag_cls.attr("Node") = node_cls;
}

}  // namespace mifrost
