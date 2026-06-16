#include <absl/container/btree_map.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/axiom_evaluators/interface.hpp>
#include <mimir/search/formatter.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <optional>
#include <ranges>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>

#include "mifrost/batch_builder_python.hpp"
#include "mifrost/batch_encoding_attributes.hpp"
#include "mifrost/batch_encoding_collection.hpp"
#include "mifrost/batch_encoding_conversion.hpp"
#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_graph_field_mutation.hpp"
#include "mifrost/batch_encoding_graph_field_serialization.hpp"
#include "mifrost/batch_encoding_python_collation.hpp"
#include "mifrost/batch_encoding_state.hpp"
#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/dlpack_utils.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/goal_inputs.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/encoders/common/transition_dag.hpp"
#include "mifrost/core/encoders/hetero/hgraph_stream_encoder.hpp"
#include "mifrost/core/encoders/hetero/horizon_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/successor_hgraph_encoder.hpp"
#include "mifrost/core/map_view.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/core/nb_instance.hpp"
#include "mifrost/core/schema_key_separators.hpp"
#include "mifrost/pyg_views.hpp"
#include "mifrost/schema_python.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

void init_batch_encoding(nb::module_& m)
{
   register_mapview_maybe< absl::btree_map< std::string, bool > >(m);
   register_mapview_maybe< hash_map< std::string, int > >(m);

   register_batch_builder(m);

   nb::class_< HeteroBatchEncodingView >(m, "HeteroBatchEncodingView")
      .def_prop_ro("num_graphs", &HeteroBatchEncodingView::num_graphs)
      .def_prop_ro("num_nodes", &HeteroBatchEncodingView::num_nodes)
      .def_prop_ro("num_edges", &HeteroBatchEncodingView::num_edges)
      .def_prop_ro("graph_kind", &HeteroBatchEncodingView::graph_kind)
      .def_prop_ro("node_types", &HeteroBatchEncodingView::node_types)
      .def_prop_ro("edge_types", &HeteroBatchEncodingView::edge_types)
      .def_prop_ro("object_names", &HeteroBatchEncodingView::object_names)
      .def_prop_ro(
         "base", &HeteroBatchEncodingView::base, nb::sig("def base(self) -> BatchEncoding")
      )
      .def_prop_ro(
         "x_dict",
         &HeteroBatchEncodingView::x_dict,
         nb::sig("def x_dict(self) -> collections.abc.Mapping[str, torch.Tensor]")
      )
      .def_prop_ro(
         "edge_index_dict",
         &HeteroBatchEncodingView::edge_index_dict,
         nb::sig(
            "def edge_index_dict(self) -> collections.abc.Mapping[tuple[str, str, str], "
            "torch.Tensor]"
         )
      )
      .def_prop_ro(
         "batch_dict",
         &HeteroBatchEncodingView::batch_dict,
         nb::sig("def batch_dict(self) -> collections.abc.Mapping[str, torch.Tensor]")
      )
      .def_prop_ro(
         "ptr_dict",
         &HeteroBatchEncodingView::ptr_dict,
         nb::sig("def ptr_dict(self) -> collections.abc.Mapping[str, torch.Tensor]")
      )
      .def_prop_ro(
         "edge_attr_dict",
         &HeteroBatchEncodingView::edge_attr_dict,
         nb::sig(
            "def edge_attr_dict(self) -> collections.abc.Mapping[tuple[str, str, str], "
            "torch.Tensor]"
         )
      )
      .def(
         "to",
         [](HeteroBatchEncodingView& view, nb::handle device) -> HeteroBatchEncodingView& {
            view.set_device(device);
            return view;
         },
         "device"_a,
         nb::rv_policy::reference_internal
      )
      .def("__getattr__", [](HeteroBatchEncodingView& view, const std::string& key) -> nb::object {
         return nb::borrow< nb::object >(view.base()).attr(key.c_str());
      });

   nb::class_< HomoBatchEncodingView >(m, "HomoBatchEncodingView")
      .def_prop_ro("num_graphs", &HomoBatchEncodingView::num_graphs)
      .def_prop_ro("num_nodes", &HomoBatchEncodingView::num_nodes)
      .def_prop_ro("num_edges", &HomoBatchEncodingView::num_edges)
      .def_prop_ro("graph_kind", &HomoBatchEncodingView::graph_kind)
      .def_prop_ro("node_types", &HomoBatchEncodingView::node_types)
      .def_prop_ro("edge_types", &HomoBatchEncodingView::edge_types)
      .def_prop_ro("object_names", &HomoBatchEncodingView::object_names)
      .def_prop_ro("base", &HomoBatchEncodingView::base, nb::sig("def base(self) -> BatchEncoding"))
      .def_prop_ro("x", &HomoBatchEncodingView::x, nb::sig("def x(self) -> torch.Tensor | None"))
      .def_prop_ro(
         "edge_index",
         &HomoBatchEncodingView::edge_index,
         nb::sig("def edge_index(self) -> torch.Tensor | None")
      )
      .def_prop_ro(
         "batch", &HomoBatchEncodingView::batch, nb::sig("def batch(self) -> torch.Tensor | None")
      )
      .def_prop_ro(
         "ptr", &HomoBatchEncodingView::ptr, nb::sig("def ptr(self) -> torch.Tensor | None")
      )
      .def_prop_ro(
         "edge_attr",
         &HomoBatchEncodingView::edge_attr,
         nb::sig("def edge_attr(self) -> torch.Tensor | None")
      )
      .def(
         "to",
         [](HomoBatchEncodingView& view, nb::handle device) -> HomoBatchEncodingView& {
            view.set_device(device);
            return view;
         },
         "device"_a,
         nb::rv_policy::reference_internal
      )
      .def("__getattr__", [](HomoBatchEncodingView& view, const std::string& key) -> nb::object {
         return nb::borrow< nb::object >(view.base()).attr(key.c_str());
      });

   auto batch_encoding_cls =
      nb::class_< BatchBuilder::BatchEncoding >(m, "BatchEncoding", nb::dynamic_attr())
         .def(nb::init<>())
         .def_ro("num_graphs", &BatchBuilder::BatchEncoding::num_graphs)
         .def_prop_ro("num_nodes", &batch_encoding_num_nodes)
         .def_prop_ro("num_edges", &batch_encoding_num_edges)
         .def_prop_ro("node_types", &batch_encoding_node_types)
         .def_prop_ro("edge_types", &batch_encoding_edge_types)
         .def_ro("graph_kind", &BatchBuilder::BatchEncoding::graph_kind)
         .def_ro("schema", &BatchBuilder::BatchEncoding::schema)
         .def_prop_ro(
            "schema_flags",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.schema_flags called with invalid instance"
               );
               return make_map_view(encoding->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def_ro("node_feature_dims", &BatchBuilder::BatchEncoding::node_feature_dims)
         .def_prop_ro(
            "graph_attrs",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.graph_attrs called with invalid instance"
               );
               materialize_batch_encoding_lazy_graph_attrs(*encoding);
               return encoding->graph_attrs;
            }
         )
         .def(
            "schema_flags_view",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.schema_flags_view called with invalid instance"
               );
               return make_map_view(encoding->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def(
            "node_feature_dims_view",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.node_feature_dims_view called with invalid instance"
               );
               return make_map_view(encoding->node_feature_dims, self);
            },
            nb::rv_policy::move
         )
         .def(
            "as_dict",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.as_dict called with invalid instance"
               );
               return batch_encoding_as_dict(*encoding, self);
            }
         )
         .def(
            "to",
            [](nb::handle self, nb::handle device) {
               return batch_encoding_to_device(self, device);
            },
            "device"_a
         )
         .def(
            "set_field",
            [](nb::handle self, const std::string& key, nb::handle value) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.set_field called with invalid instance"
               );
               set_batch_encoding_graph_field(*encoding, key, value);
               clear_owner_tensor_cache(self);
            },
            "key"_a,
            "value"_a
         )
         .def(
            "set_fields",
            [](nb::handle self, const nb::dict& values) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.set_fields called with invalid instance"
               );
               set_batch_encoding_graph_fields(*encoding, values);
               clear_owner_tensor_cache(self);
            },
            "values"_a
         )
         .def("collate_spec", [](nb::handle self) { return batch_encoding_collate_spec(self); })
         .def(
            "has_field",
            [](nb::handle self, const std::string& key) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.has_field called with invalid instance"
               );
               return batch_encoding_has_graph_field(*encoding, key);
            },
            "key"_a
         )
         .def(
            "get_field",
            [](nb::handle self, const std::string& key) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.get_field called with invalid instance"
               );
               return batch_encoding_get_graph_field(*encoding, key, self);
            },
            "key"_a
         )
         .def(
            "__getattr__",
            [](nb::handle self, const std::string& key) -> nb::object {
               return batch_encoding_getattr(self, key);
            }
         )
         .def(
            "__setattr__",
            [](nb::handle self, const std::string& key, nb::handle value) {
               batch_encoding_setattr(self, key, value);
            }
         )
         .def(
            "__repr__",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.__repr__ called with invalid instance"
               );
               return batch_encoding_repr(self, *encoding);
            }
         )
         .def(
            "__str__",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.__str__ called with invalid instance"
               );
               return batch_encoding_str(self, *encoding);
            }
         )
         .def("keys", [](nb::handle self) { return batch_encoding_keys(self); })
         .def("items", [](nb::handle self) { return batch_encoding_items(self); })
         .def(
            "as_pyg",
            [](nb::handle self, std::optional< bool > as_batch, bool include_python_attrs) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.as_pyg called with invalid instance"
               );
               nb::object out = batch_encoding_as_pyg(*encoding, as_batch);
               if(include_python_attrs) {
                  copy_python_attrs_to_object(self, out, as_batch, *encoding);
               }
               return out;
            },
            nb::sig(
               "def as_pyg(self, as_batch: bool | None = None, include_python_attrs: bool = "
               "True) -> mifrost.encoders.types.PygDataLike"
            ),
            "as_batch"_a = nb::none(),
            "include_python_attrs"_a = true
         )
         .def(
            "as_hetero",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.as_hetero called with invalid instance"
               );
               if(encoding->graph_kind != "hetero") {
                  throw std::invalid_argument(
                     "BatchEncoding graph_kind mismatch: expected 'hetero'"
                  );
               }
               return HeteroBatchEncodingView(nb::borrow< nb::object >(self));
            }
         )
         .def(
            "as_homo",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.as_homo called with invalid instance"
               );
               if(encoding->graph_kind != "homo" && encoding->graph_kind != "flat") {
                  throw std::invalid_argument(
                     "BatchEncoding graph_kind mismatch: expected 'homo' or 'flat'"
                  );
               }
               if(encoding->schema.node_types.size() > 1
                  or encoding->schema.edge_types.size() > 1) {
                  throw std::invalid_argument(
                     "BatchEncoding.as_homo() expects schema with at most one node type and one "
                     "edge type"
                  );
               }
               return HomoBatchEncodingView(nb::borrow< nb::object >(self));
            }
         )
         .def("schema_fingerprint", &schema_fingerprint)
         .def(
            "save",
            [](nb::handle self, const std::string& path, bool include_metadata) {
               batch_encoding_save(self, path, include_metadata);
            },
            "path"_a,
            "include_metadata"_a = false
         )
         .def_static("load", [](const std::string& path) { return batch_encoding_load(path); })
         .def(
            "dumps",
            [](nb::handle self, bool include_metadata) {
               return batch_encoding_dumps(self, include_metadata);
            },
            "include_metadata"_a = true
         )
         .def_static(
            "loads",
            [](nb::bytes payload) { return batch_encoding_loads(std::move(payload)); },
            "payload"_a
         )
         .def("__getstate__", [](nb::handle self) { return batch_encoding_getstate(self); })
         .def("__reduce__", [](nb::handle self) { return batch_encoding_reduce(self); })
         .def("__reduce_ex__", [](nb::handle self, int) { return batch_encoding_reduce(self); })
         .def("__setstate__", [](nb::handle self, const nb::dict& state) {
            batch_encoding_setstate(self, state);
         });

   batch_encoding_cls.attr("__mifrost_map_view_methods__") = nb::make_tuple(
      "schema_flags_view", "node_feature_dims_view"
   );

   m.def(
      "_set_batch_encoding_collate_spec",
      [](nb::handle self, const nb::dict& specs) {
         register_batch_encoding_collate_spec(self, specs);
      },
      "encoding"_a,
      "specs"_a
   );

   m.def(
      "batch_encodings",
      [](nb::sequence encodings, nb::object collate_spec_obj, bool fast_path) -> nb::object {
         return batch_encodings_from_sequence(encodings, std::move(collate_spec_obj), fast_path);
      },
      nb::sig(
         "def batch_encodings(encodings, collate_spec=None, fast_path=False) -> BatchEncoding"
      ),
      "encodings"_a,
      "collate_spec"_a = nb::none(),
      "fast_path"_a = false
   );
}

}  // namespace mifrost
