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
#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_graph_field_mutation.hpp"
#include "mifrost/batch_encoding_graph_field_serialization.hpp"
#include "mifrost/batch_encoding_python_collation.hpp"
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

constexpr std::string_view kPythonTensorCacheAttr = "__mifrost_tensor_cache__";

std::string make_type_attr_key(std::string_view type_key, std::string_view attr)
{
   std::string key;
   key.reserve(type_key.size() + attr.size() + 1);
   key.append(type_key);
   key.push_back(schema_key::kTypeAttrSeparator);
   key.append(attr);
   return key;
}

template < typename T >
nb::object vector_to_1d_tensor_view(std::vector< T >& vec, nb::handle owner)
{
   return dlpack_utils::vector_to_dlpack_view_1d(vec, owner);
}

template < typename T >
nb::object
vector_to_2d_tensor_view(std::vector< T >& vec, size_t rows, size_t cols, nb::handle owner)
{
   return dlpack_utils::vector_to_dlpack_view_2d(vec, rows, cols, owner);
}

template < typename T >
nb::object vector_to_1d_tensor_owned(std::vector< T >&& vec)
{
   return dlpack_utils::vector_to_dlpack_owned_1d(std::move(vec));
}

template < typename T >
nb::object vector_to_2d_tensor_owned(std::vector< T >&& vec, size_t rows, size_t cols)
{
   return dlpack_utils::vector_to_dlpack_owned_2d(std::move(vec), rows, cols);
}

void copy_store_attrs_without_batch(nb::object& dst_store, nb::object& src_store)
{
   for(auto key_obj : src_store.attr("keys")()) {
      const std::string key = py::to_std_string(key_obj);
      if(key == "ptr" or key == "batch") {
         continue;
      }
      dst_store.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
   }
}

void copy_global_attrs_for_single(nb::object& dst, nb::object& src)
{
   nb::object global_store = src.attr("_global_store");
   for(auto key_obj : global_store.attr("keys")()) {
      const std::string key = py::to_std_string(key_obj);
      if(key == "_num_graphs") {
         continue;
      }
      nb::object value = global_store.attr("__getitem__")(key_obj);
      if(key == "object_names") {
         value = py::flatten_single_graph_metadata_list(value);
      }
      dst.attr(key.c_str()) = value;
   }
}

void copy_global_attrs_for_batch(nb::object& dst, nb::object& src)
{
   nb::object global_store = src.attr("_global_store");
   for(auto key_obj : global_store.attr("keys")()) {
      const std::string key = py::to_std_string(key_obj);
      nb::object value = global_store.attr("__getitem__")(key_obj);
      dst.attr(key.c_str()) = value;
   }
}

void make_homo_edge_index_undirected_in_place(nb::object& out)
{
   // Keep homo Data semantics aligned with Python fallback conversion:
   // duplicate non-self-loop edges to expose an undirected edge_index.
   if(not nb::cast< bool >(out.attr("__contains__")("edge_index"))) {
      return;
   }

   nb::object edge_index = out.attr("__getitem__")("edge_index");
   if(nb::cast< int64_t >(edge_index.attr("numel")()) <= 0) {
      return;
   }

   nb::object src = edge_index.attr("__getitem__")(0);
   nb::object dst = edge_index.attr("__getitem__")(1);
   nb::object mask = src.attr("__ne__")(dst);
   nb::object rev = py::torch_stack_fn()(
      nb::make_tuple(dst.attr("__getitem__")(mask), src.attr("__getitem__")(mask)),
      nb::arg("dim") = 0
   );
   nb::object cat = py::torch_module().attr("cat");
   out.attr("__setitem__")("edge_index", cat(nb::make_tuple(edge_index, rev), nb::arg("dim") = 1));

   if(nb::cast< bool >(out.attr("__contains__")("edge_attr"))) {
      nb::object edge_attr = out.attr("__getitem__")("edge_attr");
      out.attr("__setitem__")(
         "edge_attr",
         cat(nb::make_tuple(edge_attr, edge_attr.attr("__getitem__")(mask)), nb::arg("dim") = 0)
      );
   }
}

nb::object batch_to_single_hetero_data(nb::object& pyg_batch)
{
   nb::object out = py::torch_geometric_heterodata_ctor()();

   for(auto node_type_obj : pyg_batch.attr("node_types")) {
      std::string node_type = py::to_std_string(node_type_obj);
      nb::object src_store = pyg_batch.attr("__getitem__")(node_type);
      nb::object dst_store = out.attr("__getitem__")(node_type);
      copy_store_attrs_without_batch(dst_store, src_store);
      if(nb::cast< bool >(src_store.attr("__contains__")("node_names"))) {
         nb::object flat_names = py::flatten_single_graph_metadata_list(
            src_store.attr("__getitem__")("node_names")
         );
         dst_store.attr("node_names") = flat_names;
         dst_store.attr("num_nodes") = nb::len(flat_names);
      }
   }

   for(auto edge_type_obj : pyg_batch.attr("edge_types")) {
      nb::object src_store = pyg_batch.attr("__getitem__")(edge_type_obj);
      nb::object dst_store = out.attr("__getitem__")(edge_type_obj);
      for(auto key_obj : src_store.attr("keys")()) {
         dst_store.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
   }

   copy_global_attrs_for_single(out, pyg_batch);
   return out;
}

nb::object batch_to_batch_homo_data(nb::object& pyg_batch)
{
   nb::object out = py::torch_geometric_batch_ctor()(
      nb::arg("_base_cls") = py::torch_geometric_data_ctor()
   );

   nb::list node_types = nb::cast< nb::list >(pyg_batch.attr("node_types"));
   if(nb::len(node_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=True) for homo expects a single node type"
      );
   }
   if(nb::len(node_types) == 1) {
      std::string node_type = py::to_std_string(node_types[0]);
      nb::object src_store = pyg_batch.attr("__getitem__")(node_type);
      for(auto key_obj : src_store.attr("keys")()) {
         out.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
   }

   nb::list edge_types = nb::cast< nb::list >(pyg_batch.attr("edge_types"));
   if(nb::len(edge_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=True) for homo expects a single edge type"
      );
   }
   if(nb::len(edge_types) == 1) {
      nb::object src_store = pyg_batch.attr("__getitem__")(edge_types[0]);
      for(auto key_obj : src_store.attr("keys")()) {
         out.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
   }

   copy_global_attrs_for_batch(out, pyg_batch);
   return out;
}

nb::object batch_to_single_homo_data(nb::object& pyg_batch)
{
   nb::object out = py::torch_geometric_data_ctor()();

   nb::list node_types = nb::cast< nb::list >(pyg_batch.attr("node_types"));
   if(nb::len(node_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=False) for homo expects a single node type"
      );
   }

   if(nb::len(node_types) == 1) {
      std::string node_type = py::to_std_string(node_types[0]);
      nb::object src_store = pyg_batch.attr("__getitem__")(node_type);
      for(auto key_obj : src_store.attr("keys")()) {
         const std::string key = py::to_std_string(key_obj);
         if(key == "ptr" or key == "batch") {
            continue;
         }
         out.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
      if(nb::cast< bool >(src_store.attr("__contains__")("node_names"))) {
         nb::object flat_names = py::flatten_single_graph_metadata_list(
            src_store.attr("__getitem__")("node_names")
         );
         out.attr("node_names") = flat_names;
      }
   }

   nb::list edge_types = nb::cast< nb::list >(pyg_batch.attr("edge_types"));
   if(nb::len(edge_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=False) for homo expects a single edge type"
      );
   }
   if(nb::len(edge_types) == 1) {
      nb::object src_store = pyg_batch.attr("__getitem__")(edge_types[0]);
      for(auto key_obj : src_store.attr("keys")()) {
         out.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
   }

   make_homo_edge_index_undirected_in_place(out);
   copy_global_attrs_for_single(out, pyg_batch);
   return out;
}

std::vector< std::string > materialize_target_name_states(
   std::span< const mimir::search::State > states
)
{
   std::vector< std::string > names;
   names.reserve(states.size());
   for(const auto& state : states) {
      std::ostringstream stream;
      stream << state;
      names.push_back(stream.str());
   }
   return names;
}

void materialize_batch_encoding_lazy_graph_attrs(BatchBuilder::BatchEncoding& encoding)
{
   if(not encoding.lazy_target_name_strings.empty()) {
      const auto graph_attr_it = encoding.graph_attrs.find(std::string(kTargetNamesAttr));
      if(graph_attr_it == encoding.graph_attrs.end()) {
         encoding.graph_attrs.emplace(
            std::string(kTargetNamesAttr), std::move(encoding.lazy_target_name_strings)
         );
      } else {
         auto* existing = std::get_if< std::vector< std::string > >(&graph_attr_it->second);
         if(existing == nullptr) {
            throw std::invalid_argument(
               "BatchEncoding target_names graph attr must be a string vector"
            );
         }
         existing->insert(
            existing->end(),
            std::make_move_iterator(encoding.lazy_target_name_strings.begin()),
            std::make_move_iterator(encoding.lazy_target_name_strings.end())
         );
         encoding.lazy_target_name_strings.clear();
      }
   }
   if(not encoding.lazy_target_name_states.empty()) {
      auto names = materialize_target_name_states(std::span(encoding.lazy_target_name_states));
      const auto graph_attr_it = encoding.graph_attrs.find(std::string(kTargetNamesAttr));
      if(graph_attr_it == encoding.graph_attrs.end()) {
         encoding.graph_attrs.emplace(std::string(kTargetNamesAttr), std::move(names));
      } else {
         auto* existing = std::get_if< std::vector< std::string > >(&graph_attr_it->second);
         if(existing == nullptr) {
            throw std::invalid_argument(
               "BatchEncoding target_names graph attr must be a string vector"
            );
         }
         existing->insert(
            existing->end(),
            std::make_move_iterator(names.begin()),
            std::make_move_iterator(names.end())
         );
      }
      encoding.lazy_target_name_states.clear();
   }
}

nb::dict batch_encoding_to_state_dict(BatchBuilder::BatchEncoding& encoding, bool include_metadata)
{
   auto map_to_dict =
      []< typename value_t >(const absl::btree_map< std::string, value_t >& values) {
         nb::dict out;
         for(const auto& [key, value] : values) {
            out[key.c_str()] = value;
         }
         return out;
      };

   nb::dict state;
   materialize_batch_encoding_lazy_graph_attrs(encoding);
   state["format_version"] = 1;
   state["graph_kind"] = encoding.graph_kind;
   state["num_graphs"] = encoding.num_graphs;
   state["schema_flags"] = map_to_dict(encoding.schema_flags);
   state["node_feature_dims"] = encoding.node_feature_dims;
   state["graph_attrs"] = encoding.graph_attrs;
   state["graph_fields"] = graph_field_map_to_dict(encoding.graph_fields);
   state["ptrs"] = encoding.ptrs;
   state["node_counts"] = map_to_dict(encoding.node_counts);
   state["schema"] = schema_to_dict(encoding.schema);
   if(include_metadata) {
      state["node_names"] = encoding.node_names;
      state["object_names"] = encoding.object_names;
   } else {
      state["node_names"] = hash_map< std::string, std::vector< std::string > >{};
      state["object_names"] = std::vector< std::string >{};
   }

   nb::dict columns;
   for(const auto& [key, column] : encoding.columns) {
      nb::dict c;
      c["dim"] = column.dim;
      std::visit(
         [&]< typename T >(const std::vector< T >& data) {
            if constexpr(std::is_same_v< T, float >) {
               c["dtype"] = "f32";
            } else {
               c["dtype"] = "i64";
            }
            c["length"] = static_cast< int64_t >(data.size());
            const auto* ptr = reinterpret_cast< const char* >(data.data());
            c["raw"] = nb::bytes(ptr, data.size() * sizeof(T));
         },
         column.data
      );
      columns[key.c_str()] = std::move(c);
   }
   state["columns"] = std::move(columns);
   return state;
}

template < typename value_type >
auto map_from_dict(const nb::dict& source)
{
   absl::btree_map< std::string, value_type > out;
   for(auto [key_obj, value_obj] : source) {
      out.emplace(py::to_std_string(key_obj), nb::cast< value_type >(value_obj));
   }
   return out;
};

BatchBuilder::BatchEncoding batch_encoding_from_state_dict(const nb::dict& state)
{
   const int version = nb::cast< int >(state["format_version"]);
   if(version != 1) {
      throw std::invalid_argument("Unsupported BatchEncoding format version");
   }

   BatchBuilder::BatchEncoding encoding;
   try {
      encoding.graph_kind = py::to_std_string(state["graph_kind"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['graph_kind']: " + std::string(ex.what()));
   }
   try {
      encoding.num_graphs = nb::cast< int64_t >(state["num_graphs"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['num_graphs']: " + std::string(ex.what()));
   }
   try {
      nb::dict schema_flags = nb::cast< nb::dict >(state["schema_flags"]);
      encoding.schema_flags = map_from_dict< bool >(schema_flags);
   } catch(const std::exception& ex) {
      throw std::invalid_argument(
         "Failed to parse state['schema_flags']: " + std::string(ex.what())
      );
   }
   {
      nb::dict node_feature_dims;
      try {
         node_feature_dims = nb::cast< nb::dict >(state["node_feature_dims"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_feature_dims']: " + std::string(ex.what())
         );
      }
      encoding.node_feature_dims.clear();
      encoding.node_feature_dims.reserve(node_feature_dims.size());
      for(auto [key_obj, value_obj] : node_feature_dims) {
         try {
            encoding.node_feature_dims.emplace(
               py::to_std_string(key_obj), nb::cast< int >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['node_feature_dims'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   {
      nb::dict graph_attrs;
      try {
         graph_attrs = nb::cast< nb::dict >(state["graph_attrs"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['graph_attrs']: " + std::string(ex.what())
         );
      }
      encoding.graph_attrs.clear();
      encoding.graph_attrs.reserve(graph_attrs.size());
      for(auto [key_obj, value_obj] : graph_attrs) {
         try {
            encoding.graph_attrs.emplace(
               py::to_std_string(key_obj), nb::cast< BatchBuilder::GraphAttrValue >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['graph_attrs'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   if(state.contains("graph_fields")) {
      try {
         encoding.graph_fields = graph_field_map_from_dict(
            nb::cast< nb::dict >(state["graph_fields"])
         );
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['graph_fields']: " + std::string(ex.what())
         );
      }
   }
   {
      nb::dict ptrs;
      try {
         ptrs = nb::cast< nb::dict >(state["ptrs"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument("Failed to parse state['ptrs']: " + std::string(ex.what()));
      }
      encoding.ptrs.clear();
      encoding.ptrs.reserve(ptrs.size());
      for(auto [key_obj, value_obj] : ptrs) {
         try {
            encoding.ptrs.emplace(
               py::to_std_string(key_obj), nb::cast< std::vector< int64_t > >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['ptrs'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   {
      try {
         nb::dict node_counts = nb::cast< nb::dict >(state["node_counts"]);
         encoding.node_counts = map_from_dict< int64_t >(node_counts);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_counts']: " + std::string(ex.what())
         );
      }
   }
   {
      try {
         nb::dict schema = nb::cast< nb::dict >(state["schema"]);
         encoding.schema = schema_from_dict(schema);
      } catch(const std::exception& ex) {
         throw std::invalid_argument("Failed to parse state['schema']: " + std::string(ex.what()));
      }
   }
   {
      nb::dict node_names;
      try {
         node_names = nb::cast< nb::dict >(state["node_names"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_names']: " + std::string(ex.what())
         );
      }
      encoding.node_names.clear();
      encoding.node_names.reserve(node_names.size());
      for(auto [key_obj, value_obj] : node_names) {
         try {
            encoding.node_names.emplace(
               py::to_std_string(key_obj), nb::cast< std::vector< std::string > >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['node_names'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   try {
      encoding.object_names = nb::cast< std::vector< std::string > >(state["object_names"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument(
         "Failed to parse state['object_names']: " + std::string(ex.what())
      );
   }

   nb::dict columns;
   try {
      columns = nb::cast< nb::dict >(state["columns"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['columns']: " + std::string(ex.what()));
   }
   for(auto [key_obj, col_obj] : columns) {
      auto col = nb::cast< nb::dict >(col_obj);
      const auto key = py::to_std_string(key_obj);
      const auto dim = nb::cast< int >(col["dim"]);
      const auto dtype = py::to_std_string(col["dtype"]);
      const auto length = nb::cast< int64_t >(col["length"]);
      const auto raw_bytes = nb::cast< nb::bytes >(col["raw"]);
      const std::string_view raw(raw_bytes.c_str(), raw_bytes.size());

      if(dtype == "f32") {
         if(raw.size() != length * sizeof(float)) {
            throw std::invalid_argument("Malformed f32 column payload");
         }
         std::vector< float > data(length);
         if(length > 0) {
            std::memcpy(data.data(), raw.data(), raw.size());
         }
         encoding.columns[key] = BatchBuilder::Column{std::move(data), dim};
      } else if(dtype == "i64") {
         if(raw.size() != length * sizeof(int64_t)) {
            throw std::invalid_argument("Malformed i64 column payload");
         }
         std::vector< int64_t > data(length);
         if(length > 0) {
            std::memcpy(data.data(), raw.data(), raw.size());
         }
         encoding.columns[key] = BatchBuilder::Column{std::move(data), dim};
      } else {
         throw std::invalid_argument("Unsupported BatchEncoding column dtype");
      }
   }

   return encoding;
}

nb::dict batch_encoding_state_from_instance(nb::handle self, bool include_metadata)
{
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      self, "BatchEncoding state extraction called with invalid instance"
   );
   validate_batch_encoding_graph_fields(*encoding, "BatchEncoding state extraction");
   nb::dict state = batch_encoding_to_state_dict(*encoding, include_metadata);
   nb::dict py_attrs = batch_encoding_python_attrs_copy(self);
   if(py_attrs.contains(kPythonTensorCacheAttr.data())) {
      py_attrs.attr("pop")(kPythonTensorCacheAttr.data());
   }
   if(nb::len(py_attrs) > 0) {
      state["python_attrs"] = std::move(py_attrs);
   }
   return state;
}

nb::object batch_encoding_object_from_state(const nb::dict& state)
{
   nb::object obj = py::mifrost_core_batch_encoding_cls()();
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      obj, "Failed to instantiate BatchEncoding during state load"
   );
   *encoding = batch_encoding_from_state_dict(state);
   auto attrs = nb::cast< nb::dict >(obj.attr("__dict__"));
   attrs.clear();
   batch_encoding_apply_python_attrs_from_state(obj, state, attrs);
   clear_owner_tensor_cache(obj);
   return obj;
}

int64_t batch_encoding_num_nodes(const BatchBuilder::BatchEncoding& encoding)
{
   int64_t total = 0;
   for(const auto& count : std::views::values(encoding.node_counts)) {
      total += count;
   }
   return total;
}

int64_t batch_encoding_num_edges(const BatchBuilder::BatchEncoding& encoding)
{
   int64_t total = 0;
   for(const auto& [key, col] : encoding.columns) {
      const auto edge_index_pos = key.find(schema_key::kEdgeIndexKeyPrefix);
      if(edge_index_pos == std::string::npos) {
         continue;
      }
      const auto component_pos = edge_index_pos + schema_key::kEdgeIndexKeyPrefix.size();
      if(component_pos >= key.size() or key[component_pos] != schema_key::kEdgeIndexSrcComponent) {
         continue;
      }
      std::visit([&](const auto& data) { total += static_cast< int64_t >(data.size()); }, col.data);
   }
   return total;
}

std::vector< std::string > batch_encoding_node_types(const BatchBuilder::BatchEncoding& encoding)
{
   return encoding.schema.node_types;
}

nb::list batch_encoding_edge_types(const BatchBuilder::BatchEncoding& encoding)
{
   nb::list out;
   for(const auto& edge_type : encoding.schema.edge_types) {
      out.append(nb::make_tuple(edge_type.src, edge_type.rel, edge_type.dst));
   }
   return out;
}

nb::dict batch_encoding_as_dict(BatchBuilder::BatchEncoding& encoding, nb::handle owner)
{
   materialize_batch_encoding_lazy_graph_attrs(encoding);
   nb::dict tensors;

   for(auto& [key, col] : encoding.columns) {
      const bool is_edge_index = key.find(schema_key::kEdgeIndexKeyPrefix) != std::string::npos;
      std::visit(
         [&]< typename T >(std::vector< T >& data) {
            if(is_edge_index) {
               tensors[key.c_str()] = vector_to_1d_tensor_view(data, owner);
               return;
            }
            const size_t rows = col.dim > 0 ? data.size() / static_cast< size_t >(col.dim) : 0;
            tensors[key.c_str()] = vector_to_2d_tensor_view(
               data, rows, static_cast< size_t >(col.dim), owner
            );
         },
         col.data
      );
   }

   bool exported_ptr = false;
   for(auto& [node_type, ptr] : encoding.ptrs) {
      if(ptr.size() < 2) {
         continue;
      }
      exported_ptr = true;
      tensors[make_type_attr_key(node_type, schema_key::kPtrAttr)
                 .c_str()] = vector_to_1d_tensor_view(ptr, owner);
      tensors[make_type_attr_key(node_type, schema_key::kBatchAttr)
                 .c_str()] = vector_to_1d_tensor_owned(ptr_to_batch(ptr));
   }
   if(not exported_ptr) {
      for(const auto& [node_type, count] : encoding.node_counts) {
         if(count <= 0) {
            continue;
         }
         std::vector< int64_t > ptr{0, count};
         tensors[make_type_attr_key(node_type, schema_key::kPtrAttr)
                    .c_str()] = vector_to_1d_tensor_owned(std::move(ptr));
         tensors[make_type_attr_key(node_type, schema_key::kBatchAttr)
                    .c_str()] = vector_to_1d_tensor_owned(std::vector< int64_t >(count, 0));
      }
   }

   for(auto& [attr, field] : encoding.graph_fields) {
      const std::string key = make_type_attr_key("__graph__", attr);
      std::visit(
         [&]< typename T >(std::vector< T >& data) {
            if(field.spec.dim == 1) {
               tensors[key.c_str()] = vector_to_1d_tensor_view(data, owner);
               return;
            }
            const bool cat_dim_one = (field.spec.mode == GraphFieldMode::CAT
                                      or field.spec.mode == GraphFieldMode::RAGGED_CAT)
                                     and graph_field_cat_dim_is_one(field.spec.cat_dim);
            const size_t rows = cat_dim_one ? static_cast< size_t >(field.spec.dim)
                                            : data.size() / static_cast< size_t >(field.spec.dim);
            const size_t cols = cat_dim_one ? data.size() / static_cast< size_t >(field.spec.dim)
                                            : static_cast< size_t >(field.spec.dim);
            tensors[key.c_str()] = vector_to_2d_tensor_view(data, rows, cols, owner);
         },
         field.values
      );
      if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
         tensors[make_type_attr_key(key, schema_key::kPtrAttr).c_str()] = vector_to_1d_tensor_view(
            field.ptr, owner
         );
      }
   }

   nb::dict out;
   out["tensors"] = std::move(tensors);
   out["schema"] = schema_to_dict(encoding.schema);

   nb::dict node_names_dict;
   for(const auto& [node_type, names] : encoding.node_names) {
      node_names_dict[node_type.c_str()] = nb::cast(names);
   }
   out["node_names"] = std::move(node_names_dict);

   nb::dict dims_dict;
   for(const auto& [node_type, dim] : encoding.node_feature_dims) {
      dims_dict[node_type.c_str()] = dim;
   }
   out["node_feature_dims"] = std::move(dims_dict);
   out["object_names"] = nb::cast(encoding.object_names);
   out["num_graphs"] = encoding.num_graphs;
   out["schema_fingerprint"] = schema_fingerprint(encoding);

   if(not encoding.graph_attrs.empty()) {
      nb::dict graph_attrs_dict;
      for(const auto& [key, value] : encoding.graph_attrs) {
         std::visit([&](const auto& v) { graph_attrs_dict[key.c_str()] = nb::cast(v); }, value);
      }
      out["graph_attrs"] = std::move(graph_attrs_dict);
   }

   return out;
}

std::optional< nb::object >
batch_encoding_graph_attr_if_present(BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   if(key == kTargetNamesAttr) {
      materialize_batch_encoding_lazy_graph_attrs(encoding);
   }
   auto it = encoding.graph_attrs.find(std::string(key));
   if(it == encoding.graph_attrs.end()) {
      return std::nullopt;
   }
   return std::visit([](const auto& value) -> nb::object { return nb::cast(value); }, it->second);
}

nb::object
batch_encoding_as_pyg(BatchBuilder::BatchEncoding& encoding, std::optional< bool > as_batch)
{
   materialize_batch_encoding_lazy_graph_attrs(encoding);
   validate_batch_encoding_graph_fields(encoding, "BatchEncoding.as_pyg");
   const bool want_batch = as_batch.value_or(encoding.num_graphs != 1);
   const bool is_flat =
      encoding.graph_kind == "flat"
      || (encoding.schema.flags.contains("flat_relations") && encoding.schema.flags.at("flat_relations"));
   const bool is_homo_like = encoding.graph_kind == "homo" || is_flat;
   BatchBuilder builder;
   builder.set_graph_kind(encoding.graph_kind);
   builder.load_from_batch_encoding(encoding);
   nb::object pyg_batch = batch_builder_build_pyg(builder);

   if(not want_batch and encoding.num_graphs != 1) {
      throw std::invalid_argument("BatchEncoding.as_pyg(as_batch=False) requires num_graphs == 1");
   }

   if(not want_batch) {
      if(is_homo_like) {
         nb::object out = batch_to_single_homo_data(pyg_batch);
         if(is_flat) {
            return py::mifrost_flat_relation_data_from_pyg_fn()(
               out, nb::arg("schema_fingerprint") = schema_fingerprint(encoding)
            );
         }
         return out;
      }
      return batch_to_single_hetero_data(pyg_batch);
   }

   if(is_homo_like) {
      nb::object out = batch_to_batch_homo_data(pyg_batch);
      if(is_flat) {
         return py::mifrost_flat_relation_data_from_pyg_fn()(
            out, nb::arg("schema_fingerprint") = schema_fingerprint(encoding)
         );
      }
      return out;
   }
   return pyg_batch;
}

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
