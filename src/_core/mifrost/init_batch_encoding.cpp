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

   auto batch_builder_cls =
      nb::class_< BatchBuilder >(m, "BatchBuilder")
         .def(nb::init<>())
         .def(
            "add_node_features",
            [](BatchBuilder& builder,
               const std::string& node_type,
               const std::string& attr_name,
               nb::ndarray< nb::numpy, float > data) {
               if(data.ndim() != 1 and data.ndim() != 2) {
                  throw std::invalid_argument("add_node_features expects a 1D/2D array");
               }
               const int feature_dim = data.ndim() == 2 ? static_cast< int >(data.shape(1)) : 1;
               const auto count = static_cast< size_t >(data.size());
               builder.add_node_features(
                  node_type, attr_name, std::span< const float >(data.data(), count), feature_dim
               );
            }
         )
         .def(
            "add_edges",
            [](BatchBuilder& builder,
               const std::string& src_type,
               const std::string& rel_type,
               const std::string& dst_type,
               nb::ndarray< nb::numpy, int64_t > src,
               nb::ndarray< nb::numpy, int64_t > dst) {
               if(src.ndim() != 1 or dst.ndim() != 1) {
                  throw std::invalid_argument("add_edges expects 1D arrays for src/dst indices");
               }
               if(src.size() != dst.size()) {
                  throw std::invalid_argument("add_edges expects src/dst arrays of equal length");
               }
               builder.add_edges(
                  src_type,
                  rel_type,
                  dst_type,
                  std::span< const int64_t >(src.data(), src.size()),
                  std::span< const int64_t >(dst.data(), dst.size())
               );
            }
         )
         .def(
            "add_edge_features",
            [](BatchBuilder& builder,
               const std::string& src_type,
               const std::string& rel_type,
               const std::string& dst_type,
               const std::string& attr_name,
               nb::ndarray< nb::numpy, float > data) {
               if(data.ndim() != 1 and data.ndim() != 2) {
                  throw std::invalid_argument("add_edge_features expects a 1D/2D array");
               }
               const int feature_dim = data.ndim() == 2 ? static_cast< int >(data.shape(1)) : 1;
               const auto count = static_cast< size_t >(data.size());
               builder.add_edge_features(
                  src_type,
                  rel_type,
                  dst_type,
                  attr_name,
                  std::span< const float >(data.data(), count),
                  feature_dim
               );
            }
         )
         .def("add_nodes", &BatchBuilder::add_nodes)
         .def("add_edge", &BatchBuilder::add_edge)
         .def("set_node_names", &BatchBuilder::set_node_names)
         .def("set_object_names", &BatchBuilder::set_object_names)
         .def("build", &BatchBuilder::build)
         .def("build_pyg", [](BatchBuilder& builder) { return batch_builder_build_pyg(builder); })
         .def("append_batch_encoding", &BatchBuilder::append_batch_encoding)
         .def(
            "load_from_batch_encoding",
            nb::overload_cast< const BatchBuilder::BatchEncoding& >(
               &BatchBuilder::load_from_batch_encoding
            )
         )
         .def("next_graph", &BatchBuilder::next_graph)
         .def("set_graph_kind", &BatchBuilder::set_graph_kind, "kind"_a)
         .def("set_schema_flag", &BatchBuilder::set_schema_flag, "key"_a, "value"_a)
         .def(
            "schema_flags_view",
            [](nb::handle self) {
               auto* builder = require_instance_ptr< BatchBuilder >(
                  self, "BatchBuilder.schema_flags_view called with invalid instance"
               );
               return make_map_view(builder->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def(
            "node_feature_dims_view",
            [](nb::handle self) {
               auto* builder = require_instance_ptr< BatchBuilder >(
                  self, "BatchBuilder.node_feature_dims_view called with invalid instance"
               );
               return make_map_view(builder->node_feature_dims, self);
            },
            nb::rv_policy::move
         )
         .def("field_keys", &BatchBuilder::field_keys)
         .def(
            "field_specs",
            [](const BatchBuilder& builder) {
               nb::dict out;
               for(const auto& [key, spec] : builder.field_specs()) {
                  out[key.c_str()] = graph_field_spec_to_dict(spec);
               }
               return out;
            }
         )
         .def(
            "register_field",
            [](BatchBuilder& builder, const std::string& key, const nb::dict& spec) {
               builder.register_field(key, graph_field_spec_from_dict(spec));
            },
            "key"_a,
            "spec"_a
         )
         .def(
            "set_field",
            [](BatchBuilder& builder, const std::string& key, nb::handle value) {
               set_batch_builder_graph_field(builder, key, value);
            },
            "key"_a,
            "value"_a
         )
         .def(
            "set_fields",
            [](BatchBuilder& builder, const nb::dict& values) {
               set_batch_builder_graph_fields(builder, values);
            },
            "values"_a
         );

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
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.to called with invalid instance"
               );
               (void) encoding;
               if(device.is_none()) {
                  return nb::borrow< nb::object >(self);
               }
               nb::object normalized = py::torch_device_ctor()(device);
               const bool same_device = owner_target_device_matches(self, normalized);
               set_owner_target_device(self, normalized);
               nb::dict attrs = batch_encoding_python_attrs(self);
               for(auto [key_obj, value_obj] : attrs) {
                  const std::string key = py::to_std_string(key_obj);
                  if(is_forbidden_dynamic_attr_key(*encoding, key)) {
                     continue;
                  }
                  attrs[key_obj] = move_object_to_device(
                     nb::borrow< nb::object >(value_obj), normalized
                  );
               }
               if(not same_device) {
                  clear_owner_tensor_cache(self);
                  materialize_owner_tensor_cache(self, *encoding);
               }
               return nb::borrow< nb::object >(self);
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
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.__getattr__ called with invalid instance"
               );
               if(batch_encoding_has_graph_field(*encoding, key)) {
                  return batch_encoding_get_graph_field(*encoding, key, self);
               }
               if(auto value = batch_encoding_graph_attr_if_present(*encoding, key);
                  value.has_value()) {
                  return std::move(*value);
               }
               const std::string message = "'BatchEncoding' object has no attribute '" + key + "'";
               PyErr_SetString(PyExc_AttributeError, message.c_str());
               throw nb::python_error();
            }
         )
         .def(
            "__setattr__",
            [](nb::handle self, const std::string& key, nb::handle value) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.__setattr__ called with invalid instance"
               );
               if(batch_encoding_has_graph_field(*encoding, key)) {
                  if(is_native_graph_field_ptr_key(*encoding, key)) {
                     throw std::invalid_argument(
                        "Direct assignment to ragged ptr key '" + key
                        + "' is not supported; assign the base field as (values, ptr)"
                     );
                  }
                  set_batch_encoding_graph_field(*encoding, key, value);
                  clear_owner_tensor_cache(self);
                  return;
               }
               if(is_forbidden_dynamic_attr_key(*encoding, key)) {
                  throw std::invalid_argument(
                     "Dynamic attribute key '" + key + "' collides with reserved/native key"
                  );
               }
               py::set_python_attribute(self, key, value);
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
         .def(
            "keys",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.keys called with invalid instance"
               );
               auto key_set = batch_encoding_native_graph_field_keys(*encoding);
               nb::dict attrs = batch_encoding_python_attrs(self);
               for(auto [key_obj, value_obj] : attrs) {
                  (void) value_obj;
                  const std::string key = py::to_std_string(key_obj);
                  if(is_forbidden_dynamic_attr_key(*encoding, key) or key_set.contains(key)) {
                     continue;
                  }
                  key_set.insert(key);
               }
               nb::list out;
               for(const auto& key : key_set) {
                  out.append(key);
               }
               return out;
            }
         )
         .def(
            "items",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.items called with invalid instance"
               );
               auto key_set = batch_encoding_native_graph_field_keys(*encoding);
               nb::dict attrs = batch_encoding_python_attrs(self);
               for(auto [key_obj, value_obj] : attrs) {
                  (void) value_obj;
                  const std::string key = py::to_std_string(key_obj);
                  if(is_forbidden_dynamic_attr_key(*encoding, key) or key_set.contains(key)) {
                     continue;
                  }
                  key_set.insert(key);
               }

               nb::list out;
               for(const auto& key : key_set) {
                  nb::object value;
                  if(batch_encoding_has_graph_field(*encoding, key)) {
                     value = batch_encoding_get_graph_field(*encoding, key, self);
                  } else {
                     value = nb::borrow< nb::object >(attrs[key.c_str()]);
                  }
                  out.append(nb::make_tuple(key, std::move(value)));
               }
               return out;
            }
         )
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
               nb::object file = py::builtins_open()(path, "wb");
               nb::dict state = batch_encoding_state_from_instance(self, include_metadata);
               auto payload = py::pickle_dumps()(state, 5);
               file.attr("write")(payload);
               file.attr("close")();
            },
            "path"_a,
            "include_metadata"_a = false
         )
         .def_static(
            "load",
            [](const std::string& path) {
               nb::object file = py::builtins_open()(path, "rb");
               nb::bytes payload = nb::cast< nb::bytes >(file.attr("read")());
               nb::dict state = nb::cast< nb::dict >(py::pickle_loads()(payload));
               file.attr("close")();
               return batch_encoding_object_from_state(state);
            }
         )
         .def(
            "dumps",
            [](nb::handle self, bool include_metadata) {
               nb::dict state = batch_encoding_state_from_instance(self, include_metadata);
               return nb::cast< nb::bytes >(py::pickle_dumps()(state, 5));
            },
            "include_metadata"_a = true
         )
         .def_static(
            "loads",
            [](nb::bytes payload) {
               nb::dict state = nb::cast< nb::dict >(py::pickle_loads()(payload));
               return batch_encoding_object_from_state(state);
            },
            "payload"_a
         )
         .def(
            "__getstate__",
            [](nb::handle self) { return batch_encoding_state_from_instance(self, true); }
         )
         .def(
            "__reduce__",
            [](nb::handle self) {
               nb::bytes payload = nb::cast< nb::bytes >(self.attr("dumps")(true));
               return nb::make_tuple(
                  py::mifrost_batch_encoding_loader(), nb::make_tuple(std::move(payload))
               );
            }
         )
         .def(
            "__reduce_ex__",
            [](nb::handle self, int) {
               nb::bytes payload = nb::cast< nb::bytes >(self.attr("dumps")(true));
               return nb::make_tuple(
                  py::mifrost_batch_encoding_loader(), nb::make_tuple(std::move(payload))
               );
            }
         )
         .def("__setstate__", [](nb::handle self, const nb::dict& state) {
            auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
               self, "BatchEncoding.__setstate__ called with invalid instance"
            );
            *encoding = batch_encoding_from_state_dict(state);
            batch_encoding_clear_python_attrs(self);
            batch_encoding_apply_python_attrs_from_state(self, state);
            clear_owner_tensor_cache(self);
         });

   batch_builder_cls.attr("__mifrost_map_view_methods__") = nb::make_tuple(
      "schema_flags_view", "node_feature_dims_view"
   );
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
         auto enc_cast = [](const nb::handle& source) -> BatchEncoding* {
            return require_instance_ptr< BatchBuilder::BatchEncoding >(
               source, "batch_encodings expects BatchEncoding inputs"
            );
         };

         if(nb::len(encodings) == 0) {
            return nb::cast(BatchBuilder::BatchEncoding{});
         }
         const BatchEncoding* zeroth_entry = enc_cast(encodings[0]);

         std::vector< const BatchEncoding* > entries;
         entries.reserve(nb::len(encodings));
         entries.push_back(zeroth_entry);
         for(size_t i = 1; i < static_cast< size_t >(nb::len(encodings)); ++i) {
            entries.push_back(enc_cast(encodings[i]));
         }

         bool use_fast_path = false;
         if(fast_path and not entries.empty()) {
            const auto expected_fp = schema_fingerprint(*entries.front());
            use_fast_path = true;
            for(size_t i = 1; i < entries.size(); ++i) {
               if(schema_fingerprint(*entries[i]) != expected_fp) {
                  use_fast_path = false;
                  break;
               }
            }
         }

         BatchBuilder builder;
         builder.set_graph_kind(zeroth_entry->graph_kind);
         for(size_t i = 0; i < entries.size(); ++i) {
            const BatchEncoding* encoding = entries[i];
            if(encoding->num_graphs != 1) {
               throw std::invalid_argument("batch_encodings expects inputs with num_graphs == 1");
            }
            if(not use_fast_path or i == 0) {
               validate_batch_encoding_graph_fields(*encoding, "batch_encodings input validation");
            }
            builder.append_batch_encoding(*encoding);
         }

         BatchEncoding out = builder.build();
         auto [collate_spec, source_attrs] = std::invoke([&] {
            try {
               return build_python_collation_inputs(encodings, std::move(collate_spec_obj));
            } catch(const std::exception& ex) {
               throw std::invalid_argument(
                  "batch_encodings collate_spec preparation failed: " + std::string(ex.what())
               );
            }
         });

         const auto reserved_native_keys = batch_encoding_native_tensor_keys(out);
         auto filtered_specs = filter_python_collate_spec_for_native_collisions(
            collate_spec, reserved_native_keys
         );
         const auto default_keys = collect_default_python_collation_keys(
            source_attrs, filtered_specs
         );
         for(const auto& key : default_keys) {
            if(reserved_native_keys.contains(key)) {
               throw std::invalid_argument(
                  "Default collation key '" + key + "' collides with a native field key"
               );
            }
         }
         auto out_py = nb::cast(out);
         if(filtered_specs.empty() and default_keys.empty()) {
            return out_py;
         }

         try {
            nb::dict out_attrs = apply_python_collation(
               filtered_specs,
               source_attrs,
               std::views::iota(size_t{0}, nb::len(encodings))
                  | std::views::transform([&](size_t i) { return enc_cast(encodings[i]); })
            );
            nb::dict default_attrs = apply_default_python_collation(default_keys, source_attrs);
            for(auto [k, v] : default_attrs) {
               out_attrs[k] = nb::borrow< nb::object >(v);
            }
            for(auto [k, v] : out_attrs) {
               py::set_python_attribute(out_py, nb::str(k), v);
            }
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "batch_encodings python collation failed: " + std::string(ex.what())
            );
         }

         if(not filtered_specs.empty()) {
            try {
               register_batch_encoding_collate_spec(
                  out_py, python_collate_spec_to_dict(filtered_specs)
               );
            } catch(const std::exception& ex) {
               throw std::invalid_argument(
                  "batch_encodings collate_spec registration failed: " + std::string(ex.what())
               );
            }
         }
         return out_py;
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
