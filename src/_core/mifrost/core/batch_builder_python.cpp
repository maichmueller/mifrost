#include "mifrost/batch_builder_python.hpp"

#if defined(MIFROST_ENABLE_PYTHON_API)

   #include <absl/container/btree_map.h>
   #include <fmt/format.h>
   #include <nanobind/ndarray.h>
   #include <nanobind/stl/map.h>
   #include <nanobind/stl/string.h>
   #include <nanobind/stl/vector.h>

   #include <algorithm>
   #include <array>
   #include <span>
   #include <stdexcept>
   #include <string_view>

   #include "mifrost/batch_encoding_graph_field_mutation.hpp"
   #include "mifrost/batch_encoding_graph_field_serialization.hpp"
   #include "mifrost/common.hpp"
   #include "mifrost/core/dlpack_utils.hpp"
   #include "mifrost/core/encoders/common/target_metadata.hpp"
   #include "mifrost/core/map_view.hpp"
   #include "mifrost/core/nb_instance.hpp"
   #include "mifrost/core/schema_key_separators.hpp"

namespace mifrost {
using namespace nb::literals;

namespace {

bool is_reserved_pyg_graph_attr_key(std::string_view key)
{
   static constexpr std::array< std::string_view, 14 > kReserved{
      "x",
      "edge_index",
      "edge_attr",
      "batch",
      "ptr",
      "x_dict",
      "edge_index_dict",
      "edge_attr_dict",
      "batch_dict",
      "ptr_dict",
      "_num_graphs",
      "object_names",
      "node_names",
      "num_nodes",
   };
   return key.starts_with("__mifrost_") or std::ranges::find(kReserved, key) != kReserved.end();
}

bool pyg_global_store_contains_key(nb::object& batch, const std::string& key)
{
   nb::object global_store = batch.attr("_global_store");
   return nb::cast< bool >(global_store.attr("__contains__")(key.c_str()));
}

std::string make_type_attr_key(std::string_view type_key, std::string_view attr)
{
   std::string key;
   key.reserve(type_key.size() + attr.size() + 1);
   key.append(type_key);
   key.push_back(schema_key::kTypeAttrSeparator);
   key.append(attr);
   return key;
}

bool key_has_edge_separator(std::string_view key)
{
   return key.find(schema_key::kEdgeTypeSeparator) != std::string_view::npos;
}

std::string_view::size_type find_type_attr_separator(std::string_view key)
{
   return key.find(schema_key::kTypeAttrSeparator);
}

bool key_has_edge_index_prefix(std::string_view key)
{
   return key.find(schema_key::kEdgeIndexKeyPrefix) != std::string_view::npos;
}

bool key_has_ptr_suffix(std::string_view key)
{
   if(key == schema_key::kPtrAttr) {
      return true;
   }
   if(key.size() <= schema_key::kPtrAttr.size()) {
      return false;
   }
   const auto suffix_pos = key.size() - schema_key::kPtrAttr.size();
   return key[suffix_pos - 1] == schema_key::kTypeAttrSeparator
          and key.substr(suffix_pos) == schema_key::kPtrAttr;
}

std::vector< std::string > materialize_target_name_batches(
   const std::vector< std::shared_ptr< const DeferredStringBatch > >& batches
)
{
   std::vector< std::string > names;
   for(const auto& batch : batches) {
      auto batch_names = batch->materialize();
      names.insert(
         names.end(),
         std::make_move_iterator(batch_names.begin()),
         std::make_move_iterator(batch_names.end())
      );
   }
   return names;
}

void append_target_name_strings(BatchBuilder& builder, const std::vector< std::string >& names)
{
   if(names.empty()) {
      return;
   }
   auto graph_attr_it = builder.graph_attrs.find(std::string(kTargetNamesAttr));
   if(graph_attr_it == builder.graph_attrs.end()) {
      builder.graph_attrs.emplace(std::string(kTargetNamesAttr), names);
      return;
   }
   auto* existing = std::get_if< std::vector< std::string > >(&graph_attr_it->second);
   if(existing == nullptr) {
      throw std::invalid_argument("Graph attr 'target_names' must be a string vector");
   }
   existing->insert(existing->end(), names.begin(), names.end());
}

void materialize_builder_lazy_target_names(BatchBuilder& builder)
{
   if(not builder.lazy_target_name_strings.empty()) {
      append_target_name_strings(builder, builder.lazy_target_name_strings);
      builder.lazy_target_name_strings.clear();
   }
   if(builder.lazy_target_name_batches.empty()) {
      return;
   }
   append_target_name_strings(
      builder, materialize_target_name_batches(builder.lazy_target_name_batches)
   );
   builder.lazy_target_name_batches.clear();
}

void set_graph_attrs_on_pyg_batch(
   nb::object& batch,
   const hash_map< std::string, BatchBuilder::GraphAttrValue >& graph_attrs,
   const std::unique_ptr< hash_map< std::string, GraphField > >& graph_fields
)
{
   if(graph_attrs.empty()) {
      return;
   }

   auto collides_with_native_graph_field = [&](const std::string& key) {
      if(not graph_fields) {
         return false;
      }
      if(graph_fields->contains(key)) {
         return true;
      }
      if(key.size() > 4 and key.ends_with("_ptr")) {
         std::string base_key = key.substr(0, key.size() - 4);
         if(graph_fields->contains(base_key)) {
            return true;
         }
      }
      return false;
   };

   for(const auto& [key, value] : graph_attrs) {
      if(is_reserved_pyg_graph_attr_key(key) or pyg_global_store_contains_key(batch, key)
         or collides_with_native_graph_field(key)) {
         throw std::invalid_argument(
            "Graph attr key '" + key + "' collides with a reserved/existing PyG key"
         );
      }
      std::visit(
         [&](const auto& typed_value) {
            batch.attr("__setattr__")(key.c_str(), nb::cast(typed_value));
         },
         value
      );
   }
}

template < typename T >
nb::object vector_to_1d_dlpack(std::vector< T >&& vec)
{
   return dlpack_utils::vector_to_dlpack_owned_1d(std::move(vec));
}

template < typename T >
nb::object vector_to_2d_dlpack(std::vector< T >&& vec, size_t rows, size_t cols)
{
   return dlpack_utils::vector_to_dlpack_owned_2d(std::move(vec), rows, cols);
}

}  // namespace

nb::dict batch_builder_build_dict(BatchBuilder& builder)
{
   auto& columns = builder.columns;
   auto& ptrs = builder.ptrs;

   nb::dict out;

   for(auto& [key, col] : columns) {
      const bool is_edge_index = key_has_edge_index_prefix(key);
      std::visit(
         [&]< typename T >(std::vector< T >& vec) {
            size_t size = vec.size();
            if(is_edge_index) {
               out[key.c_str()] = vector_to_1d_dlpack(std::move(vec));
               return;
            }

            int dim = col.dim;
            size_t num_rows = dim > 0 ? size / dim : 0;
            out[key.c_str()] = vector_to_2d_dlpack(
               std::move(vec), num_rows, static_cast< size_t >(dim)
            );
         },
         col.data
      );
   }

   for(auto& [ntype, p_vec] : ptrs) {
      auto tensor = vector_to_1d_dlpack(std::move(p_vec));
      std::string key = make_type_attr_key(ntype, schema_key::kPtrAttr);
      out[key.c_str()] = tensor;
   }

   return out;
}

nb::object batch_builder_build_pyg(BatchBuilder& builder)
{
   auto& columns = builder.columns;
   auto& ptrs = builder.ptrs;
   auto& current_node_counts = builder.current_node_counts;
   auto& node_feature_dims = builder.node_feature_dims;
   auto& node_names = builder.node_names;
   auto& object_names = builder.object_names;
   auto& graph_kind = builder.graph_kind;
   auto& graph_attrs = builder.graph_attrs;
   auto& graph_fields = builder.graph_fields;

   absl::btree_map< std::string, int64_t > node_counts;
   for(const auto& [key, col] : columns) {
      if(key_has_edge_separator(key)) {
         continue;
      }
      const auto slash = find_type_attr_separator(key);
      if(slash == std::string::npos) {
         continue;
      }
      const std::string node_type = key.substr(0, slash);
      std::visit(
         [&](const auto& items) {
            const size_t size = items.size();
            const int dim = col.dim;
            const int64_t rows = dim > 0 ? static_cast< int64_t >(size / dim) : 0;
            auto& count = node_counts[node_type];
            if(rows > count) {
               count = rows;
            }
         },
         col.data
      );
   }
   for(const auto& [node_type, ptr] : ptrs) {
      if(not ptr.empty()) {
         const int64_t count = ptr.back();
         auto& existing = node_counts[node_type];
         if(count > existing) {
            existing = count;
         }
      }
   }
   for(const auto& [node_type, count] : current_node_counts) {
      auto& existing = node_counts[node_type];
      if(count > existing) {
         existing = count;
      }
   }
   for(const auto& [node_type, names] : node_names) {
      auto& existing = node_counts[node_type];
      const int64_t count = static_cast< int64_t >(names.size());
      if(count > existing) {
         existing = count;
      }
   }
   for(const auto& [node_type, dim] : node_feature_dims) {
      (void) dim;
      if(not node_counts.contains(node_type)) {
         node_counts[node_type] = 0;
      }
   }

   absl::btree_map< std::string, std::vector< int64_t > > ptr_vectors;
   absl::btree_map< std::string, std::vector< int64_t > > batch_vectors;
   int64_t graph_count = 0;
   for(const auto& [node_type, ptr] : ptrs) {
      if(ptr.size() < 2) {
         continue;
      }
      ptr_vectors[node_type] = ptr;
      graph_count = std::max< int64_t >(graph_count, ptr.size() - 1);
   }
   // A node type may first appear after the first graph.  Match the native
   // builder's PyG contract by repeating its initial per-type offset for the
   // graphs before it existed, so every ptr has one boundary per graph.
   for(auto& [node_type, ptr] : ptr_vectors) {
      (void) node_type;
      const auto missing = static_cast< size_t >(graph_count + 1) - ptr.size();
      if(missing > 0) {
         ptr.insert(ptr.begin(), static_cast< long >(missing), ptr.front());
      }
   }
   for(const auto& [node_type, ptr] : ptr_vectors) {
      std::vector< int64_t > batch;
      batch.reserve(ptr.back());
      for(size_t idx = 0; idx + 1 < ptr.size(); ++idx) {
         const int64_t count = ptr[idx + 1] - ptr[idx];
         batch.insert(batch.end(), count, static_cast< int64_t >(idx));
      }
      batch_vectors[node_type] = std::move(batch);
   }
   if(ptr_vectors.empty()) {
      for(const auto& [node_type, count] : node_counts) {
         if(count <= 0) {
            continue;
         }
         ptr_vectors[node_type] = {0, count};
         batch_vectors[node_type] = std::vector< int64_t >(count, 0);
      }
      if(not node_counts.empty()) {
         graph_count = 1;
      }
   }

   nb::dict payload = batch_builder_build_dict(builder);

   nb::object batch = py::torch_geometric_batch_ctor()(
      nb::arg("_base_cls") = py::torch_geometric_heterodata_ctor()
   );

   using EdgeKey = std::tuple< std::string, std::string, std::string >;
   struct EdgeComponents {
      nb::object src;
      nb::object dst;
   };
   absl::btree_map< EdgeKey, EdgeComponents > edge_components;

   for(auto [key_handle, value_handle] : payload) {
      const std::string key = nb::str(key_handle).c_str();
      if(key_has_ptr_suffix(key)) {
         continue;
      }

      const auto edge_pos = key.rfind(schema_key::kEdgeIndexKeyPrefix);
      if(edge_pos != std::string::npos) {
         const std::string base = key.substr(0, edge_pos);
         const std::string suffix = key.substr(edge_pos + schema_key::kEdgeIndexKeyPrefix.size());
         const auto first = base.find(schema_key::kEdgeTypeSeparator);
         const auto second = base.find(schema_key::kEdgeTypeSeparator, first + 1);
         if(first == std::string::npos or second == std::string::npos) {
            throw std::invalid_argument(fmt::format("Malformed edge key '{}'", key));
         }
         const std::string src = base.substr(0, first);
         const std::string rel = base.substr(first + 1, second - first - 1);
         const std::string dst = base.substr(second + 1);

         EdgeKey edge_key{src, rel, dst};
         auto& components = edge_components[edge_key];
         nb::object tensor = py::to_torch_tensor(nb::borrow< nb::object >(value_handle));
         if(suffix.size() == 1 and suffix.front() == schema_key::kEdgeIndexSrcComponent) {
            components.src = tensor;
         } else if(suffix.size() == 1 and suffix.front() == schema_key::kEdgeIndexDstComponent) {
            components.dst = tensor;
         } else {
            throw std::invalid_argument(fmt::format("Unexpected edge index suffix '{}'", key));
         }
         continue;
      }

      const auto slash = find_type_attr_separator(key);
      if(slash == std::string::npos) {
         continue;
      }
      const std::string type_key = key.substr(0, slash);
      const std::string attr = key.substr(slash + 1);

      const auto first = type_key.find(schema_key::kEdgeTypeSeparator);
      const auto second = first == std::string::npos
                             ? std::string::npos
                             : type_key.find(schema_key::kEdgeTypeSeparator, first + 1);
      const auto third = second == std::string::npos
                            ? std::string::npos
                            : type_key.find(schema_key::kEdgeTypeSeparator, second + 1);
      const bool is_edge_type = first != std::string::npos and second != std::string::npos
                                and third == std::string::npos;
      nb::object store;
      if(is_edge_type) {
         const std::string src = type_key.substr(0, first);
         const std::string rel = type_key.substr(first + 1, second - first - 1);
         const std::string dst = type_key.substr(second + 1);
         store = batch.attr("__getitem__")(nb::make_tuple(src, rel, dst));
      } else {
         store = batch.attr("__getitem__")(type_key);
      }
      nb::object tensor = py::to_torch_tensor(nb::borrow< nb::object >(value_handle));
      store.attr("__setitem__")(attr, tensor);
   }

   for(const auto& [edge_key, components] : edge_components) {
      if(not components.src.is_valid() or not components.dst.is_valid()) {
         throw std::invalid_argument("Incomplete edge_index components for edge type");
      }
      nb::object edge_index = py::torch_stack_fn()(
         nb::make_tuple(components.src, components.dst), nb::arg("dim") = 0
      );
      nb::object store = batch.attr("__getitem__")(
         nb::make_tuple(std::get< 0 >(edge_key), std::get< 1 >(edge_key), std::get< 2 >(edge_key))
      );
      store.attr("__setitem__")("edge_index", edge_index);
   }

   for(const auto& [node_type, ptr] : ptr_vectors) {
      nb::object ptr_tensor = py::to_torch_tensor(
         dlpack_utils::vector_to_dlpack_owned_copy_1d(ptr)
      );

      auto batch_it = batch_vectors.find(node_type);
      const std::vector< int64_t > batch_values = batch_it != batch_vectors.end()
                                                     ? batch_it->second
                                                     : std::vector< int64_t >{};
      nb::object batch_tensor = py::to_torch_tensor(
         dlpack_utils::vector_to_dlpack_owned_copy_1d(batch_values)
      );

      nb::object store = batch.attr("__getitem__")(node_type);
      store.attr("__setitem__")("ptr", ptr_tensor);
      store.attr("__setitem__")("batch", batch_tensor);
   }

   for(const auto& [node_type, count] : node_counts) {
      nb::object store = batch.attr("__getitem__")(node_type);
      bool has_x = nb::cast< bool >(store.attr("__contains__")("x"));
      if(not has_x) {
         int dim = 0;
         auto dim_it = node_feature_dims.find(node_type);
         if(dim_it != node_feature_dims.end()) {
            dim = dim_it->second;
         }
         nb::object zeros = py::torch_zeros_fn()(
            nb::make_tuple(count, dim), nb::arg("dtype") = py::torch_float32_dtype()
         );
         store.attr("__setitem__")("x", zeros);
      }
   }

   for(const auto& [node_type, names] : node_names) {
      nb::object store = batch.attr("__getitem__")(node_type);
      if(graph_count > 0) {
         std::vector< std::vector< std::string > > per_graph;
         auto ptr_it = ptr_vectors.find(node_type);
         if(ptr_it != ptr_vectors.end() and ptr_it->second.size() >= 2) {
            const auto& ptr = ptr_it->second;
            per_graph.reserve(ptr.size() - 1);
            for(size_t i = 0; i + 1 < ptr.size(); ++i) {
               const auto start = static_cast< size_t >(std::max< int64_t >(0, ptr[i]));
               const auto end = static_cast< size_t >(
                  std::min< int64_t >(ptr[i + 1], static_cast< int64_t >(names.size()))
               );
               if(start <= end and end <= names.size()) {
                  per_graph.emplace_back(names.begin() + start, names.begin() + end);
               } else {
                  per_graph.emplace_back();
               }
            }
         } else {
            per_graph.emplace_back(names);
         }
         store.attr("node_names") = nb::cast(per_graph);
      } else {
         store.attr("node_names") = nb::cast(names);
      }
   }
   if(not object_names.empty()) {
      if(graph_count > 0) {
         std::vector< std::vector< std::string > > per_graph;
         bool assigned = false;
         for(const auto& [node_type, names] : node_names) {
            if(names != object_names) {
               continue;
            }
            auto ptr_it = ptr_vectors.find(node_type);
            if(ptr_it == ptr_vectors.end() or ptr_it->second.size() < 2) {
               break;
            }
            const auto& ptr = ptr_it->second;
            per_graph.reserve(ptr.size() - 1);
            for(size_t i = 0; i + 1 < ptr.size(); ++i) {
               const auto start = static_cast< size_t >(std::max< int64_t >(0, ptr[i]));
               const auto end = static_cast< size_t >(
                  std::min< int64_t >(ptr[i + 1], static_cast< int64_t >(object_names.size()))
               );
               if(start <= end and end <= object_names.size()) {
                  per_graph.emplace_back(object_names.begin() + start, object_names.begin() + end);
               } else {
                  per_graph.emplace_back();
               }
            }
            assigned = true;
            break;
         }
         if(not assigned) {
            per_graph.emplace_back(object_names);
         }
         batch.attr("object_names") = nb::cast(per_graph);
      } else {
         batch.attr("object_names") = nb::cast(object_names);
      }
   }

   materialize_builder_lazy_target_names(builder);
   set_graph_attrs_on_pyg_batch(batch, graph_attrs, graph_fields);

   if(graph_fields) {
      for(auto& [attr, field] : *graph_fields) {
         nb::object value_tensor;
         std::visit(
            [&](auto& values) {
               using T = std::decay_t< decltype(values) >::value_type;
               const size_t size = values.size();
               if(field.spec.dim == 1) {
                  value_tensor = py::to_torch_tensor(vector_to_1d_dlpack< T >(std::move(values)));
               } else {
                  const bool cat_dim_one = (field.spec.mode == GraphFieldMode::CAT
                                            or field.spec.mode == GraphFieldMode::RAGGED_CAT)
                                           and graph_field_cat_dim_is_one(field.spec.cat_dim);
                  const size_t rows = cat_dim_one ? static_cast< size_t >(field.spec.dim)
                                                  : size / static_cast< size_t >(field.spec.dim);
                  const size_t cols = cat_dim_one ? size / static_cast< size_t >(field.spec.dim)
                                                  : static_cast< size_t >(field.spec.dim);
                  value_tensor = py::to_torch_tensor(
                     vector_to_2d_dlpack< T >(std::move(values), rows, cols)
                  );
               }
            },
            field.values
         );
         batch.attr("__setattr__")(attr.c_str(), value_tensor);

         if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
            std::string ptr_attr = attr + "_ptr";
            batch.attr("__setattr__")(
               ptr_attr.c_str(), py::to_torch_tensor(vector_to_1d_dlpack(std::move(field.ptr)))
            );
         }
      }
   }

   if(graph_count > 0) {
      batch.attr("_num_graphs") = graph_count;
   }

   builder.reset();
   return batch;
}

void register_batch_builder(nb::module_& m)
{
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
               if(feature_dim <= 0) {
                  throw std::invalid_argument(
                     "add_node_features feature dimension must be positive, got "
                     + std::to_string(feature_dim)
                  );
               }
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
               if(feature_dim <= 0) {
                  throw std::invalid_argument(
                     "add_edge_features feature dimension must be positive, got "
                     + std::to_string(feature_dim)
                  );
               }
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
         .def("ensure_edge_type", &BatchBuilder::ensure_edge_type)
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
            "set_graph_attr",
            [](BatchBuilder& builder, const std::string& key, nb::handle value) {
               if(nb::isinstance< nb::str >(value)) {
                  builder.set_graph_attr(key, std::string(nb::str(value).c_str()));
                  return;
               }
               if(nb::isinstance< nb::bool_ >(value)) {
                  builder.set_graph_attr(key, static_cast< int64_t >(nb::cast< bool >(value)));
                  return;
               }
               if(nb::isinstance< nb::int_ >(value)) {
                  builder.set_graph_attr(key, nb::cast< int64_t >(value));
                  return;
               }
               {
                  std::vector< std::string > strings;
                  if(nb::try_cast< std::vector< std::string > >(value, strings)) {
                     builder.set_graph_attr(key, std::move(strings));
                     return;
                  }
               }
               {
                  std::vector< int64_t > integers;
                  if(nb::try_cast< std::vector< int64_t > >(value, integers)) {
                     builder.set_graph_attr(key, std::move(integers));
                     return;
                  }
               }
               throw std::invalid_argument(
                  "BatchBuilder.set_graph_attr expects str, int, bool, list[str], or list[int]"
               );
            },
            "key"_a,
            "value"_a
         )
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

   batch_builder_cls.attr("__mifrost_map_view_methods__") = nb::make_tuple(
      "schema_flags_view", "node_feature_dims_view"
   );
}

}  // namespace mifrost

#endif
