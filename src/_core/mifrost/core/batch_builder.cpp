#include "batch_builder.hpp"

#include <fmt/format.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>

namespace mifrost {

void BatchBuilder::add_node_features(
   const std::string& node_type,
   const std::string& attr_name,
   std::span< const float > data,
   int feature_dim
)
{
   set_node_feature_dim(node_type, feature_dim);

   std::string key = node_type + "/" + attr_name;
   auto& col = get_column< float >(key, feature_dim);
   col.insert(col.end(), data.begin(), data.end());

   int64_t num_nodes = data.size() / feature_dim;
   if(current_node_counts[node_type] < num_nodes) {
      current_node_counts[node_type] = num_nodes;
   }
}

void BatchBuilder::set_node_feature_dim(const std::string& node_type, int dim)
{
   auto it = node_feature_dims.find(node_type);
   if(it == node_feature_dims.end()) {
      node_feature_dims[node_type] = dim;
      return;
   }
   if(it->second != dim) {
      throw std::invalid_argument("Node feature dim mismatch for node_type '" + node_type + "'");
   }
}

void BatchBuilder::add_nodes(const std::string& node_type, int64_t count)
{
   if(count < 0) {
      throw std::invalid_argument("Node count must be non-negative");
   }
   if(current_node_counts[node_type] < count) {
      current_node_counts[node_type] = count;
   }
}

void BatchBuilder::ensure_edge_type(
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type
)
{
   const std::string edge_key_base = src_type + "|" + rel_type + "|" + dst_type;
   const std::string src_key = edge_key_base + "/edge_index_0";
   const std::string dst_key = edge_key_base + "/edge_index_1";
   get_column< int64_t >(src_key, 1);
   get_column< int64_t >(dst_key, 1);
}

void BatchBuilder::set_node_names(const std::string& node_type, std::vector< std::string > names)
{
   const auto graph_count = static_cast< int64_t >(names.size());
   auto& existing = node_names[node_type];
   if(existing.empty()) {
      existing = std::move(names);
   } else {
      existing.reserve(existing.size() + names.size());
      existing.insert(existing.end(), names.begin(), names.end());
   }
   if(current_node_counts[node_type] < graph_count) {
      current_node_counts[node_type] = graph_count;
   }
}

void BatchBuilder::set_object_names(std::vector< std::string > names)
{
   if(object_names.empty()) {
      object_names = std::move(names);
      return;
   }
   object_names.reserve(object_names.size() + names.size());
   object_names.insert(object_names.end(), names.begin(), names.end());
}

void BatchBuilder::add_edges(
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type,
   std::span< const int64_t > src_indices,
   std::span< const int64_t > dst_indices
)
{
   if(src_indices.size() != dst_indices.size()) {
      throw std::invalid_argument("src and dst indices must have same length");
   }

   // Key Convention: Store source/dest columns separately for simplified concats
   // later? Or store as single flattened vector? Let's stick to storing separate
   // src and dst index columns for now as they are easier to build. Construct
   // keys: "src_type|rel_type|dst_type/edge_index_0"
   std::string edge_key_base = src_type + "|" + rel_type + "|" + dst_type;

   std::string src_key = edge_key_base + "/edge_index_0";
   std::string dst_key = edge_key_base + "/edge_index_1";

   // Ensure both columns exist before taking references to avoid rehash issues.
   get_column< int64_t >(src_key, 1);
   get_column< int64_t >(dst_key, 1);

   auto& col_src = std::get< LongCol >(columns.at(src_key).data);
   auto& col_dst = std::get< LongCol >(columns.at(dst_key).data);

   int64_t src_offset = node_offsets[src_type];
   int64_t dst_offset = node_offsets[dst_type];

   col_src.reserve(col_src.size() + src_indices.size());
   col_dst.reserve(col_dst.size() + dst_indices.size());

   // Apply offsets and push
   for(auto idx : src_indices)
      col_src.emplace_back(idx + src_offset);
   for(auto idx : dst_indices)
      col_dst.emplace_back(idx + dst_offset);
}

void BatchBuilder::next_graph()
{
   for(auto& [ntype, count] : current_node_counts) {
      node_offsets[ntype] += count;

      auto& p = ptrs[ntype];
      if(p.empty())
         p.emplace_back(0);
      p.emplace_back(node_offsets[ntype]);

      count = 0;
   }
   // Batch indices tracking could go here if we want homogeneous batch vector
   current_graph_idx++;
}

// --- DLPack Owner Capsule ---

template < typename T >
void vector_deleter(void* p) noexcept
{
   delete static_cast< std::vector< T >* >(p);
}

// --- Build / Export ---

nb::dict BatchBuilder::build_dict()
{
   nb::dict out;

   for(auto& [key, col] : columns) {
      bool is_edge_index = key.find("/edge_index_") != std::string::npos;
      std::visit(
         [&](auto&& items) {
            using VectorType = std::decay_t< decltype(items) >;
            using ScalarType = typename VectorType::value_type;

            // Move the vector onto the heap so the capsule can own it
            auto* heap_vec = new VectorType(std::move(items));

            size_t size = heap_vec->size();

            nb::capsule owner(heap_vec, [](void* p) noexcept {
               delete static_cast< VectorType* >(p);
            });

            if(is_edge_index) {
               size_t shape[1] = {size};
               auto tensor = nb::ndarray< nb::numpy, ScalarType, nb::shape< -1 > >(
                  heap_vec->data(), 1, shape, owner
               );
               out[key.c_str()] = tensor;
               return;
            }

            int dim = col.dim;
            size_t num_rows = dim > 0 ? size / dim : 0;
            size_t shape[2] = {num_rows, (size_t) dim};
            auto tensor = nb::ndarray< nb::numpy, ScalarType, nb::shape< -1, -1 > >(
               heap_vec->data(), 2, shape, owner
            );

            out[key.c_str()] = tensor;
         },
         col.data
      );
   }

   // Also export Ptr columns (converting them to tensor columns first
   // essentially)
   for(auto& [ntype, p_vec] : ptrs) {
      auto* heap_vec = new std::vector< int64_t >(std::move(p_vec));
      size_t shape[1] = {heap_vec->size()};

      nb::capsule owner(heap_vec, [](void* p) noexcept {
         delete static_cast< std::vector< int64_t >* >(p);
      });

      auto tensor = nb::ndarray< nb::numpy, int64_t, nb::shape< -1 > >(
         heap_vec->data(), 1, shape, owner
      );

      std::string key = ntype + "/ptr";
      out[key.c_str()] = tensor;
   }

   return out;
}

nb::object BatchBuilder::build()
{
   std::map< std::string, int64_t > node_counts;
   for(const auto& [key, col] : columns) {
      if(key.find('|') != std::string::npos) {
         continue;
      }
      const auto slash = key.find('/');
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
      if(! ptr.empty()) {
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
      if(! node_counts.contains(node_type)) {
         node_counts[node_type] = 0;
      }
   }

   std::map< std::string, std::vector< int64_t > > ptr_vectors;
   std::map< std::string, std::vector< int64_t > > batch_vectors;
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
      if(! node_counts.empty()) {
         graph_count = 1;
      }
   }

   nb::dict payload = build_dict();

   nb::object torch = nb::module_::import_("torch");
   nb::object tg_data = nb::module_::import_("torch_geometric.data");
   nb::object batch = tg_data.attr("Batch")(nb::arg("_base_cls") = tg_data.attr("HeteroData"));

   using EdgeKey = std::tuple< std::string, std::string, std::string >;
   struct EdgeParts {
      nb::object src;
      nb::object dst;
   };
   std::map< EdgeKey, EdgeParts > edge_parts;

   auto to_tensor = [&](const nb::object& array) { return torch.attr("as_tensor")(array); };

   for(auto [key_handle, value_handle] : payload) {
      const std::string key = nb::str(key_handle).c_str();
      if(key.size() >= 4 && key.compare(key.size() - 4, 4, "/ptr") == 0) {
         continue;
      }

      const auto edge_pos = key.rfind("/edge_index_");
      if(edge_pos != std::string::npos) {
         const std::string base = key.substr(0, edge_pos);
         const std::string suffix = key.substr(edge_pos + 12);
         const auto first = base.find('|');
         const auto second = base.find('|', first + 1);
         if(first == std::string::npos || second == std::string::npos) {
            throw std::invalid_argument(fmt::format("Malformed edge key '{}'", key));
         }
         const std::string src = base.substr(0, first);
         const std::string rel = base.substr(first + 1, second - first - 1);
         const std::string dst = base.substr(second + 1);

         EdgeKey edge_key{src, rel, dst};
         auto& parts = edge_parts[edge_key];
         nb::object tensor = to_tensor(nb::borrow< nb::object >(value_handle));
         if(suffix == "0") {
            parts.src = tensor;
         } else if(suffix == "1") {
            parts.dst = tensor;
         } else {
            throw std::invalid_argument(fmt::format("Unexpected edge index suffix '{}'", key));
         }
         continue;
      }

      const auto slash = key.find('/');
      if(slash == std::string::npos) {
         continue;
      }
      const std::string node_type = key.substr(0, slash);
      const std::string attr = key.substr(slash + 1);
      nb::object store = batch.attr("__getitem__")(node_type);
      nb::object tensor = to_tensor(nb::borrow< nb::object >(value_handle));
      store.attr("__setitem__")(attr, tensor);
   }

   for(const auto& [edge_key, parts] : edge_parts) {
      if(! parts.src.is_valid() || ! parts.dst.is_valid()) {
         throw std::invalid_argument("Incomplete edge_index parts for edge type");
      }
      nb::object edge_index = torch.attr("stack")(
         nb::make_tuple(parts.src, parts.dst), nb::arg("dim") = 0
      );
      nb::object store = batch.attr("__getitem__")(
         nb::make_tuple(std::get< 0 >(edge_key), std::get< 1 >(edge_key), std::get< 2 >(edge_key))
      );
      store.attr("__setitem__")("edge_index", edge_index);
   }

   for(const auto& [node_type, ptr] : ptr_vectors) {
      auto* heap_ptr = new std::vector< int64_t >(ptr);
      size_t shape[1] = {heap_ptr->size()};
      nb::capsule owner(heap_ptr, [](void* p) noexcept {
         delete static_cast< std::vector< int64_t >* >(p);
      });
      auto ptr_array = nb::ndarray< nb::numpy, int64_t, nb::shape< -1 > >(
         heap_ptr->data(), 1, shape, owner
      );
      nb::object ptr_tensor = to_tensor(ptr_array.cast());

      auto batch_it = batch_vectors.find(node_type);
      auto* heap_batch = new std::vector< int64_t >(
         batch_it != batch_vectors.end() ? batch_it->second : std::vector< int64_t >{}
      );
      size_t batch_shape[1] = {heap_batch->size()};
      nb::capsule batch_owner(heap_batch, [](void* p) noexcept {
         delete static_cast< std::vector< int64_t >* >(p);
      });
      auto batch_array = nb::ndarray< nb::numpy, int64_t, nb::shape< -1 > >(
         heap_batch->data(), 1, batch_shape, batch_owner
      );
      nb::object batch_tensor = to_tensor(batch_array.cast());

      nb::object store = batch.attr("__getitem__")(node_type);
      store.attr("__setitem__")("ptr", ptr_tensor);
      store.attr("__setitem__")("batch", batch_tensor);
   }

   for(const auto& [node_type, count] : node_counts) {
      nb::object store = batch.attr("__getitem__")(node_type);
      bool has_x = nb::cast< bool >(store.attr("__contains__")("x"));
      if(! has_x) {
         int dim = 0;
         auto dim_it = node_feature_dims.find(node_type);
         if(dim_it != node_feature_dims.end()) {
            dim = dim_it->second;
         }
         nb::object zeros = torch.attr("zeros")(
            nb::make_tuple(count, dim), nb::arg("dtype") = torch.attr("float32")
         );
         store.attr("__setitem__")("x", zeros);
      }
   }

   for(const auto& [node_type, names] : node_names) {
      nb::object store = batch.attr("__getitem__")(node_type);
      store.attr("node_names") = nb::cast(names);
   }
   if(! object_names.empty()) {
      batch.attr("object_names") = nb::cast(object_names);
   }

   if(graph_count > 0) {
      batch.attr("_num_graphs") = graph_count;
   }

   return batch;
}

nb::dict BatchBuilder::build_parts()
{
   std::map< std::string, int64_t > node_counts;
   for(const auto& [key, col] : columns) {
      if(key.find('|') != std::string::npos) {
         continue;
      }
      const auto slash = key.find('/');
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
      if(! ptr.empty()) {
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
      if(! node_counts.contains(node_type)) {
         node_counts[node_type] = 0;
      }
   }

   std::map< std::string, std::vector< int64_t > > ptr_vectors;
   std::map< std::string, std::vector< int64_t > > batch_vectors;
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
      if(! node_counts.empty()) {
         graph_count = 1;
      }
   }

   nb::dict payload = build_dict();

   for(const auto& [node_type, ptr] : ptr_vectors) {
      auto* heap_ptr = new std::vector< int64_t >(ptr);
      size_t shape[1] = {heap_ptr->size()};
      nb::capsule owner(heap_ptr, [](void* p) noexcept {
         delete static_cast< std::vector< int64_t >* >(p);
      });
      auto ptr_array = nb::ndarray< nb::numpy, int64_t, nb::shape< -1 > >(
         heap_ptr->data(), 1, shape, owner
      );
      payload[(node_type + "/ptr").c_str()] = ptr_array;

      auto batch_it = batch_vectors.find(node_type);
      auto* heap_batch = new std::vector< int64_t >(
         batch_it != batch_vectors.end() ? batch_it->second : std::vector< int64_t >{}
      );
      size_t batch_shape[1] = {heap_batch->size()};
      nb::capsule batch_owner(heap_batch, [](void* p) noexcept {
         delete static_cast< std::vector< int64_t >* >(p);
      });
      auto batch_array = nb::ndarray< nb::numpy, int64_t, nb::shape< -1 > >(
         heap_batch->data(), 1, batch_shape, batch_owner
      );
      payload[(node_type + "/batch").c_str()] = batch_array;
   }

   nb::dict out;
   out["tensors"] = payload;
   nb::dict names_dict;
   for(const auto& [node_type, names] : node_names) {
      names_dict[node_type.c_str()] = nb::cast(names);
   }
   out["node_names"] = names_dict;
   nb::dict dims_dict;
   for(const auto& [node_type, dim] : node_feature_dims) {
      dims_dict[node_type.c_str()] = dim;
   }
   out["node_feature_dims"] = dims_dict;
   out["object_names"] = nb::cast(object_names);
   out["num_graphs"] = graph_count;
   return out;
}

}  // namespace mifrost
