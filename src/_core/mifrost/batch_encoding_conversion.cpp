#include "mifrost/batch_encoding_conversion.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "mifrost/batch_builder_python.hpp"
#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_schema.hpp"
#include "mifrost/batch_encoding_state.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/dlpack_utils.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/graph_fields.hpp"
#include "mifrost/core/schema_key_separators.hpp"
#include "mifrost/schema_python.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

namespace {

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

nb::object batch_to_single_homo_data(nb::object& pyg_batch, bool undirect_edge_index)
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

   if(undirect_edge_index) {
      make_homo_edge_index_undirected_in_place(out);
   }
   copy_global_attrs_for_single(out, pyg_batch);
   return out;
}

}  // namespace

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
   // Families that materialize their own directed reverse edges (schema flag
   // "include_reverse_edges") must not be undirected-mirrored: mirroring
   // would duplicate every edge and mislabel the copies' channel rows.
   const bool homo_undirected = is_homo_like
                                && ! encoding.schema.flags.contains("include_reverse_edges");
   BatchBuilder builder;
   builder.set_graph_kind(encoding.graph_kind);
   builder.load_from_batch_encoding(encoding);
   nb::object pyg_batch = batch_builder_build_pyg(builder);

   if(not want_batch and encoding.num_graphs != 1) {
      throw std::invalid_argument("BatchEncoding.as_pyg(as_batch=False) requires num_graphs == 1");
   }

   if(not want_batch) {
      if(is_homo_like) {
         nb::object out = batch_to_single_homo_data(pyg_batch, homo_undirected);
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

}  // namespace mifrost
