#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mifrost/bindings.hpp"
#include "mifrost/core/semantic/semantic_transition_dag.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

void init_semantic_transition_dag(nb::module_& m)
{
   nb::enum_< SemanticCandidateIdCoverage >(m, "SemanticCandidateIdCoverage")
      .value("none", SemanticCandidateIdCoverage::none)
      .value("complete", SemanticCandidateIdCoverage::complete)
      .value("partial", SemanticCandidateIdCoverage::partial);

   nb::class_< SemanticTransitionDAG::Node >(m, "SemanticTransitionNode")
      .def(
         nb::init<
            SemanticFlatRelationInput,
            int64_t,
            int64_t,
            std::optional< SemanticGroundAction >,
            std::optional< int64_t >,
            std::optional< std::vector< SemanticLiteral > >,
            std::optional< std::string > >(),
         "state"_a,
         "index"_a,
         "depth"_a,
         "incoming_action"_a.none() = nb::none(),
         "candidate_id"_a.none() = nb::none(),
         "delta_literals"_a.none() = nb::none(),
         "display_name"_a.none() = nb::none()
      )
      .def_rw("state", &SemanticTransitionDAG::Node::state)
      .def_rw("index", &SemanticTransitionDAG::Node::index)
      .def_rw("depth", &SemanticTransitionDAG::Node::depth)
      .def_prop_rw(
         "incoming_action",
         [](const SemanticTransitionDAG::Node& node) { return node.incoming_action; },
         [](SemanticTransitionDAG::Node& node, const std::optional< SemanticGroundAction >& value) {
            node.incoming_action = value;
         },
         nb::arg().none()
      )
      .def_prop_rw(
         "candidate_id",
         [](const SemanticTransitionDAG::Node& node) { return node.candidate_id; },
         [](SemanticTransitionDAG::Node& node, const std::optional< int64_t >& value) {
            node.candidate_id = value;
         },
         nb::arg().none()
      )
      .def_prop_rw(
         "delta_literals",
         [](const SemanticTransitionDAG::Node& node) { return node.delta_literals; },
         [](SemanticTransitionDAG::Node& node,
            const std::optional< std::vector< SemanticLiteral > >& value) {
            node.delta_literals = value;
         },
         nb::arg().none()
      )
      .def_prop_rw(
         "display_name",
         [](const SemanticTransitionDAG::Node& node) { return node.display_name; },
         [](SemanticTransitionDAG::Node& node, const std::optional< std::string >& value) {
            node.display_name = value;
         },
         nb::arg().none()
      );

   nb::class_< SemanticTransitionDAG >(m, "SemanticTransitionDAG")
      .def(
         nb::init<
            std::vector< SemanticPredicateSpec >,
            std::vector< SemanticActionSpec >,
            std::vector< SemanticTransitionDAG::Node >,
            std::vector< SemanticTransitionDAG::Edge > >(),
         "predicates"_a,
         "actions"_a,
         "nodes"_a,
         "edges"_a
      )
      .def_prop_ro("predicates", &SemanticTransitionDAG::predicates)
      .def_prop_ro("actions", &SemanticTransitionDAG::actions)
      .def_prop_ro("nodes", &SemanticTransitionDAG::nodes)
      .def_prop_ro("edges", &SemanticTransitionDAG::edges)
      .def_prop_ro("root", [](const SemanticTransitionDAG& dag) { return dag.root(); })
      .def("__len__", &SemanticTransitionDAG::size)
      .def("children", &SemanticTransitionDAG::children, "node_index"_a)
      .def("parents", &SemanticTransitionDAG::parents, "node_index"_a)
      .def(
         "candidate_id_coverage",
         &SemanticTransitionDAG::candidate_id_coverage,
         "include_root"_a = false
      )
      .def(
         "validate_candidate_ids",
         &SemanticTransitionDAG::validate_candidate_ids,
         "include_root"_a = false
      )
      .def("candidate_ids", &SemanticTransitionDAG::candidate_ids, "include_root"_a = false);
}

}  // namespace mifrost
