#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mifrost/backends/pymimir/encoders/common/transition_dag.hpp"
#include "mifrost/bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MAKE_OPAQUE(std::vector< mifrost::TransitionDAG::Node >);

namespace mifrost {

namespace {

std::optional< std::vector< LiteralVariant > > parse_delta_literals(nb::handle value)
{
   if(value.is_none()) {
      return std::nullopt;
   }
   if(not PySequence_Check(value.ptr()) || nb::isinstance< nb::str >(value)
      || nb::isinstance< nb::bytes >(value)) {
      throw nb::type_error("delta_literals must be a sequence of ground literals or None");
   }
   std::vector< LiteralVariant > out;
   const size_t length = nb::len(value);
   out.reserve(length);
   static nb::object* to_advanced_literal = [] {
      auto types_module = nb::module_::import_("mifrost.encoders.types");
      return new nb::object(types_module.attr("to_advanced_literal"));
   }();
   for(size_t idx = 0; idx < length; ++idx) {
      nb::handle item = value[idx];
      try {
         out.push_back(nb::cast< LiteralVariant >((*to_advanced_literal)(item)));
      } catch(...) {
         PyErr_Clear();
         throw nb::type_error(
            fmt ::format(
               "delta_literals entry at index {} must be a ground literal. "
               "Got type: {}, after advanced accessor, got type: {}",
               idx,
               nb::str(item.type()).c_str(),
               nb::str(((*to_advanced_literal)(item)).type()).c_str()
            )
               .c_str()
         );
      }
   }
   return out;
}

}  // namespace

void init_transition_dag(nb::module_& m)
{
   nb::bind_vector< std::vector< TransitionDAG::Node > >(m, "TransitionNodeList");

   auto dag_cls =
      nb::class_< TransitionDAG >(m, "TransitionDAG")
         .def(nb::init< mimir::search::State >(), "root"_a)
         .def_static(
            "from_rustworkx",
            [](const nb::object& graph, bool fallback_missing_candidate_id_to_node_index) {
               nb::object rx_module = nb::none();
               bool is_rustworkx_digraph = false;
               try {
                  rx_module = nb::module_::import_("rustworkx");
                  is_rustworkx_digraph = nb::isinstance(graph, rx_module.attr("PyDiGraph"));
               } catch(const nb::python_error& exc) {
                  if(not exc.matches(PyExc_ModuleNotFoundError)) {
                     throw;
                  }
               }
               if(not is_rustworkx_digraph) {
                  throw nb::type_error(
                     ("transition_dag_from_rustworkx expects a rustworkx.PyDiGraph, got "
                      + std::string(nb::str(graph.type()).c_str()))
                        .c_str()
                  );
               }

               auto mapping_or_attr = [](const nb::object& node_data,
                                         const char* key) -> std::pair< nb::object, bool > {
                  const nb::str key_obj(key);
                  if(PyMapping_Check(node_data.ptr())) {
                     const int has_key = PyMapping_HasKey(node_data.ptr(), key_obj.ptr());
                     if(has_key == 1) {
                        if(PyObject* item = PyObject_GetItem(node_data.ptr(), key_obj.ptr())) {
                           return {nb::steal< nb::object >(item), true};
                        }
                        PyErr_Clear();
                     } else if(has_key == -1) {
                        PyErr_Clear();
                     }
                  }
                  const int has_attr = PyObject_HasAttrString(node_data.ptr(), key);
                  if(has_attr == 1) {
                     if(PyObject* attr = PyObject_GetAttrString(node_data.ptr(), key)) {
                        return {nb::steal< nb::object >(attr), true};
                     }
                     PyErr_Clear();
                  }
                  return {nb::none(), false};
               };

               auto types_module = nb::module_::import_("mifrost.encoders.types");
               auto to_advanced_state = types_module.attr("to_advanced_state");
               auto to_advanced_action = types_module.attr("to_advanced_action");

               std::vector< int64_t > node_indices;
               for(const nb::handle node_idx_handle : graph.attr("node_indices")()) {
                  node_indices.push_back(nb::cast< int64_t >(node_idx_handle));
               }
               if(node_indices.empty()) {
                  throw nb::value_error("rustworkx PyDiGraph must not be empty");
               }

               std::vector< int64_t > root_candidates;
               root_candidates.reserve(node_indices.size());
               for(const int64_t node_idx : node_indices) {
                  if(nb::cast< int64_t >(graph.attr("in_degree")(node_idx)) == 0) {
                     root_candidates.push_back(node_idx);
                  }
               }
               if(root_candidates.size() != 1) {
                  throw nb::value_error("rustworkx PyDiGraph must contain exactly one root node");
               }
               const int64_t root_idx = root_candidates.front();

               hash_map< int64_t, mimir::search::State > node_states;
               hash_map< int64_t, std::optional< int64_t > > node_candidate_ids;
               hash_map< int64_t, std::optional< std::vector< LiteralVariant > > >
                  node_delta_literals;
               node_states.reserve(node_indices.size());
               node_candidate_ids.reserve(node_indices.size());
               node_delta_literals.reserve(node_indices.size());

               for(const int64_t node_idx : node_indices) {
                  const nb::object node_data = nb::cast< nb::object >(
                     graph.attr("get_node_data")(node_idx)
                  );

                  auto [state_data, has_state] = mapping_or_attr(node_data, "state");
                  const nb::object state_input = has_state ? state_data : node_data;

                  try {
                     node_states.emplace(
                        node_idx, nb::cast< mimir::search::State >(to_advanced_state(state_input))
                     );
                  } catch(const nb::python_error&) {
                     throw nb::type_error(("rustworkx PyDiGraph node data at index "
                                           + std::to_string(node_idx) + " must be a state input")
                                             .c_str());
                  }

                  auto [raw_candidate_id, has_candidate_id] = mapping_or_attr(
                     node_data, "candidate_id"
                  );
                  if(not has_candidate_id || raw_candidate_id.is_none()) {
                     node_candidate_ids.emplace(node_idx, std::nullopt);
                  } else {
                     if(nb::isinstance< nb::bool_ >(raw_candidate_id)
                        || not nb::isinstance< nb::int_ >(raw_candidate_id)) {
                        throw nb::type_error(
                           fmt::format(
                              "rustworkx PyDiGraph node data 'candidate_id' at index {} must be "
                              "an int or None",
                              node_idx
                           )
                              .c_str()
                        );
                     }
                     node_candidate_ids.emplace(
                        node_idx, std::optional{nb::cast< int64_t >(raw_candidate_id)}
                     );
                  }

                  auto [raw_delta_literals, has_delta_literals] = mapping_or_attr(
                     node_data, "delta_literals"
                  );
                  node_delta_literals.emplace(
                     node_idx,
                     has_delta_literals ? parse_delta_literals(raw_delta_literals) : std::nullopt
                  );
               }

               std::size_t explicit_candidate_count = 0;
               std::vector< int64_t > non_root_node_indices;
               non_root_node_indices.reserve(node_indices.size());
               for(const int64_t node_idx : node_indices) {
                  if(node_idx != root_idx) {
                     non_root_node_indices.push_back(node_idx);
                     const auto find_iter = node_candidate_ids.find(node_idx);
                     if(find_iter != node_candidate_ids.end() && find_iter->second.has_value()) {
                        ++explicit_candidate_count;
                     }
                  }
               }
               if(explicit_candidate_count > 0
                  && explicit_candidate_count != non_root_node_indices.size()
                  && not fallback_missing_candidate_id_to_node_index) {
                  int64_t missing_idx = -1;
                  for(const int64_t node_idx : non_root_node_indices) {
                     if(const auto it = node_candidate_ids.find(node_idx);
                        it == node_candidate_ids.end() || not it->second.has_value()) {
                        missing_idx = node_idx;
                        break;
                     }
                  }
                  throw nb::value_error(
                     ("rustworkx PyDiGraph has partial candidate_id coverage; missing "
                      "candidate_id for node index "
                      + std::to_string(missing_idx))
                        .c_str()
                  );
               }

               hash_map< int64_t, std::optional< int64_t > > resolved_candidate_ids;
               resolved_candidate_ids.reserve(non_root_node_indices.size());
               for(const int64_t node_idx : non_root_node_indices) {
                  auto candidate_id = node_candidate_ids[node_idx];
                  if(not candidate_id.has_value() && fallback_missing_candidate_id_to_node_index) {
                     candidate_id = node_idx;
                  }
                  resolved_candidate_ids.emplace(node_idx, candidate_id);
               }

               hash_set< int64_t > seen_candidate_ids;
               seen_candidate_ids.reserve(non_root_node_indices.size());
               for(const int64_t node_idx : non_root_node_indices) {
                  const auto candidate_id = resolved_candidate_ids[node_idx];
                  if(not candidate_id.has_value()) {
                     continue;
                  }
                  if(seen_candidate_ids.contains(*candidate_id)) {
                     throw nb::value_error(("rustworkx PyDiGraph has duplicate candidate_id "
                                            + std::to_string(*candidate_id) + " for non-root nodes")
                                              .c_str());
                  }
                  seen_candidate_ids.insert(*candidate_id);
               }

               TransitionDAG dag(node_states.at(root_idx));
               std::vector< TransitionDAG::TransitionRecord > records;

               std::vector< std::tuple< int64_t, int64_t, nb::object > > pending_edges;
               for(const nb::handle edge_handle : graph.attr("weighted_edge_list")()) {
                  const auto edge = nb::cast< nb::tuple >(edge_handle);
                  pending_edges.emplace_back(
                     nb::cast< int64_t >(edge[0]),
                     nb::cast< int64_t >(edge[1]),
                     nb::cast< nb::object >(edge[2])
                  );
               }

               hash_set< int64_t > available_nodes{root_idx};
               while(not pending_edges.empty()) {
                  std::vector< std::tuple< int64_t, int64_t, nb::object > > next_pending;
                  next_pending.reserve(pending_edges.size());
                  bool progressed = false;

                  for(const auto& [src_idx, dst_idx, edge_data] : pending_edges) {
                     if(not available_nodes.contains(src_idx)) {
                        next_pending.emplace_back(src_idx, dst_idx, edge_data);
                        continue;
                     }

                     std::optional< mimir::formalism::GroundAction > action = std::nullopt;
                     if(not edge_data.is_none()) {
                        try {
                           action = nb::cast< mimir::formalism::GroundAction >(
                              to_advanced_action(edge_data)
                           );
                        } catch(const nb::python_error&) {
                           throw nb::type_error(
                              ("rustworkx PyDiGraph edge data at (" + std::to_string(src_idx) + ", "
                               + std::to_string(dst_idx) + ") must be an action input or None")
                                 .c_str()
                           );
                        }
                     }

                     const auto candidate_id_it = resolved_candidate_ids.find(dst_idx);
                     const std::optional< int64_t > candidate_id = candidate_id_it
                                                                         == resolved_candidate_ids
                                                                               .end()
                                                                      ? std::nullopt
                                                                      : candidate_id_it->second;

                     const auto delta_literals_it = node_delta_literals.find(dst_idx);
                     const std::optional< std::vector< LiteralVariant > >
                        delta_literals = delta_literals_it == node_delta_literals.end()
                                            ? std::nullopt
                                            : delta_literals_it->second;

                     records.emplace_back(
                        TransitionDAG::TransitionRecord{
                           .parent = node_states.at(src_idx),
                           .child = node_states.at(dst_idx),
                           .action = action,
                           .candidate_id = candidate_id,
                           .delta_literals = delta_literals,
                        }
                     );
                     available_nodes.insert(dst_idx);
                     progressed = true;
                  }

                  if(not progressed) {
                     throw nb::value_error(
                        "rustworkx PyDiGraph could not be imported into TransitionDAG; "
                        "ensure it is a single rooted DAG/tree"
                     );
                  }
                  pending_edges = std::move(next_pending);
               }
               dag.register_transitions(records);

               return dag;
            },
            "graph"_a,
            "fallback_missing_candidate_id_to_node_index"_a = false
         )
         .def(
            "register_transition",
            [](TransitionDAG& dag,
               const mimir::search::State& parent,
               const mimir::search::State& child,
               const std::optional< mimir::formalism::GroundAction >& action,
               const std::optional< int64_t >& candidate_id,
               nb::handle delta_literals) {
               return dag.register_transition(
                  parent, child, action, candidate_id, parse_delta_literals(delta_literals)
               );
            },
            "parent"_a,
            "child"_a,
            "action"_a = std::nullopt,
            "candidate_id"_a = std::nullopt,
            // `nb::none()`, not `std::nullopt`: the two preceding arguments are
            // `std::optional`, whose caster accepts None on its own, but this one
            // is an `nb::handle`. A `std::nullopt` default is stored as Python
            // None and then cast back to `nb::handle` on every call that omits
            // it, which nanobind 3 refuses unless the argument accepts None --
            // and the refusal sinks the whole overload, so even
            // `register_transition(parent, child, action, candidate_id=...)`
            // failed. `nb::none()` sets that flag itself, which is why every
            // other Python-object argument in these bindings already spells its
            // default this way.
            "delta_literals"_a = nb::none()
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
                           .delta_literals = std::nullopt,
                        }
                     );
                     continue;
                  }
                  if(nb::len(record) == 5) {
                     auto parent = nb::cast< mimir::search::State >(record[0]);
                     auto child = nb::cast< mimir::search::State >(record[1]);
                     auto action = nb::cast< std::optional< mimir::formalism::GroundAction > >(
                        record[2]
                     );
                     auto candidate_id = nb::cast< std::optional< int64_t > >(record[3]);
                     records.push_back(
                        TransitionDAG::TransitionRecord{
                           .parent = std::move(parent),
                           .child = std::move(child),
                           .action = std::move(action),
                           .candidate_id = candidate_id,
                           .delta_literals = parse_delta_literals(record[4]),
                        }
                     );
                     continue;
                  }
                  throw nb::type_error(
                     "register_transitions expects 3-tuple, 4-tuple, or 5-tuple records"
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
                      .def_ro("candidate_id", &TransitionDAG::Node::candidate_id)
                      // Explicit return type: nanobind 3 makes `nb::none` a class
                      // rather than a function returning `nb::object`, so the two
                      // branches below no longer deduce a common lambda return
                      // type ("inconsistent types nanobind::none and
                      // nanobind::object").
                      .def_prop_ro(
                         "delta_literals", [](const TransitionDAG::Node& node) -> nb::object {
                            if(not node.delta_literals.has_value()) {
                               return nb::none();
                            }
                            nb::list out;
                            for(const auto& literal_variant : *node.delta_literals) {
                               std::visit(
                                  [&](const auto& literal) { out.append(nb::cast(literal)); },
                                  literal_variant
                               );
                            }
                            return nb::cast(out);
                         }
                      );

   dag_cls.attr("Node") = node_cls;
}

}  // namespace mifrost
