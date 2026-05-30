#include "batch_builder.hpp"

#if defined(MIFROST_ENABLE_PYTHON_API)

   #include <absl/container/btree_map.h>
   #include <fmt/format.h>
   #include <nanobind/stl/map.h>
   #include <nanobind/stl/string.h>
   #include <nanobind/stl/vector.h>

   #include <algorithm>
   #include <array>
   #include <mimir/search/formatter.hpp>
   #include <sstream>
   #include <stdexcept>
   #include <string_view>

   #include "mifrost/common.hpp"
   #include "mifrost/core/dlpack_utils.hpp"
   #include "mifrost/core/encoders/common/target_metadata.hpp"
   #include "mifrost/core/schema_key_separators.hpp"

namespace mifrost {

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

std::vector< std::string > format_target_name_states(std::span< const mimir::search::State > states)
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
   if(builder.lazy_target_name_states.empty()) {
      return;
   }
   append_target_name_strings(
      builder, format_target_name_states(std::span(builder.lazy_target_name_states))
   );
   builder.lazy_target_name_states.clear();
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

nb::dict BatchBuilder::build_dict()
{
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

nb::object BatchBuilder::build_pyg()
{
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

   nb::dict payload = build_dict();

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

   materialize_builder_lazy_target_names(*this);
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

   reset();
   return batch;
}

}  // namespace mifrost

#endif