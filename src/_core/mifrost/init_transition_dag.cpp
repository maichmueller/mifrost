#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
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
                           const std::optional< mimir::formalism::GroundAction >& action,
                           const std::optional< int64_t >& candidate_id) {
                           return dag.register_transition(parent, child, action, candidate_id);
                        },
                        "parent"_a,
                        "child"_a,
                        "action"_a = std::nullopt,
                        "candidate_id"_a = std::nullopt
                     )
                     .def(
                        "register_transitions",
                        [](TransitionDAG& dag, const nb::iterable& transitions) {
                           std::vector< TransitionDAG::TransitionRecord > records;
                           for(const auto entry : transitions) {
                              if(not nb::isinstance< nb::tuple >(entry)) {
                                 throw nb::type_error("register_transitions expects tuple records");
                              }
                              auto record = nb::cast< nb::tuple >(entry);
                              if(nb::len(record) == 3) {
                                 auto [parent, child, action] = nb::cast< std::tuple<
                                    mimir::search::State,
                                    mimir::search::State,
                                    std::optional< mimir::formalism::GroundAction > > >(record);
                                 records.push_back(
                                    TransitionDAG::TransitionRecord{
                                       .parent = std::move(parent),
                                       .child = std::move(child),
                                       .action = std::move(action),
                                       .candidate_id = std::nullopt,
                                    }
                                 );
                                 continue;
                              }
                              if(nb::len(record) == 4) {
                                 auto [parent, child, action, candidate_id] = nb::cast< std::tuple<
                                    mimir::search::State,
                                    mimir::search::State,
                                    std::optional< mimir::formalism::GroundAction >,
                                    std::optional< int64_t > > >(record);
                                 records.push_back(
                                    TransitionDAG::TransitionRecord{
                                       .parent = std::move(parent),
                                       .child = std::move(child),
                                       .action = std::move(action),
                                       .candidate_id = candidate_id,
                                    }
                                 );
                                 continue;
                              }
                              throw nb::type_error(
                                 "register_transitions expects 3-tuple or 4-tuple records"
                              );
                           }
                           dag.register_transitions(records);
                        },
                        "transitions"_a
                     )
                     .def("index", &TransitionDAG::index, "state"_a)
                     .def("depth", &TransitionDAG::depth, "idx"_a)
                     .def("action", &TransitionDAG::action, "idx"_a)
                     .def("state", &TransitionDAG::state, "idx"_a)
                     .def(
                        "children",
                        [](const TransitionDAG& self, int parent_idx) {
                           if(const auto* children_ptr = self.children(parent_idx); children_ptr) {
                              return nb::cast< nb::list >(nb::cast(*children_ptr));
                           }
                           return nb::list();
                        },
                        "parent_idx"_a
                     )
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
                      .def_ro("action", &TransitionDAG::Node::action)
                      .def_ro("candidate_id", &TransitionDAG::Node::candidate_id);

   dag_cls.attr("Node") = node_cls;
}

}  // namespace mifrost
