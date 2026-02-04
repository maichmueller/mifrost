#include "batch_builder.hpp"

#include <fmt/format.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

#include "schema.hpp"
#include "utils/macro.hpp"

namespace mifrost {

BatchBuilder::BatchBuilder()
{
   constexpr size_t kSmallReserve = 32;
   constexpr size_t kColumnReserve = 64;
   current_node_counts.reserve(kSmallReserve);
   node_offsets.reserve(kSmallReserve);
   node_feature_dims.reserve(kSmallReserve);
   node_names.reserve(kSmallReserve);
   ptrs.reserve(kSmallReserve);
   columns.reserve(kColumnReserve);
}

void BatchBuilder::add_node_features(
   const std::string& node_type,
   const std::string& attr_name,
   std::span< const float > data,
   int feature_dim
)
{
   set_node_feature_dim(node_type, feature_dim);

   std::string key;
   key.reserve(node_type.size() + 1 + attr_name.size());
   key.append(node_type);
   key.push_back('/');
   key.append(attr_name);
   auto& col = get_column< float >(key, feature_dim);
   col.insert(col.end(), data.begin(), data.end());

   int64_t num_nodes = data.size() / feature_dim;
   auto [it, inserted] = current_node_counts.try_emplace(node_type, num_nodes);
   if(not inserted and it->second < num_nodes) {
      it->second = num_nodes;
   }
}

void BatchBuilder::set_node_feature_dim(const std::string& node_type, int dim)
{
   auto [it, inserted] = node_feature_dims.try_emplace(node_type, dim);
   if(not inserted and it->second != dim) {
      throw std::invalid_argument(
         fmt::format("Node feature dim mismatch for node_type '{}'", node_type)
      );
   }
}

void BatchBuilder::add_nodes(const std::string& node_type, int64_t count)
{
   if(count < 0) {
      throw std::invalid_argument("Node count must be non-negative");
   }
   auto [it, inserted] = current_node_counts.try_emplace(node_type, count);
   if(not inserted and it->second < count) {
      it->second = count;
   }
}

void BatchBuilder::ensure_edge_type(
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type
)
{
   std::string edge_key_base;
   constexpr std::string_view sep = "|";
   edge_key_base.reserve(src_type.size() + rel_type.size() + dst_type.size() + 2 * sep.size() + 1);
   edge_key_base.append(src_type);
   edge_key_base.append(sep);
   edge_key_base.append(rel_type);
   edge_key_base.append(sep);
   edge_key_base.append(dst_type);

   std::string src_key;
   std::string dst_key;
   constexpr std::string_view suffix_0 = "/edge_index_0";
   constexpr std::string_view suffix_1 = "/edge_index_1";
   // src
   src_key.reserve(edge_key_base.size() + suffix_0.size() + 1);
   src_key.append(edge_key_base);
   src_key.append(suffix_0);
   // dst
   dst_key.reserve(edge_key_base.size() + suffix_1.size() + 1);
   dst_key.append(edge_key_base);
   dst_key.append(suffix_1);
   get_column< int64_t >(src_key, 1);
   get_column< int64_t >(dst_key, 1);
}

void BatchBuilder::set_node_names(const std::string& node_type, std::vector< std::string > names)
{
   const auto graph_count = static_cast< int64_t >(names.size());
   auto [it, inserted] = node_names.try_emplace(node_type, std::vector< std::string >{});
   auto& existing = it->second;
   if(existing.empty()) {
      existing = std::move(names);
   } else {
      existing.reserve(existing.size() + names.size());
      existing.insert(existing.end(), names.begin(), names.end());
   }
   auto [count_it, count_inserted] = current_node_counts.try_emplace(node_type, graph_count);
   if(not count_inserted and count_it->second < graph_count) {
      count_it->second = graph_count;
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

void BatchBuilder::set_graph_kind(std::string kind)
{
   graph_kind = std::move(kind);
}

void BatchBuilder::set_schema_flag(const std::string& key, bool value)
{
   schema_flags[key] = value;
}

void BatchBuilder::set_graph_attr(const std::string& key, std::vector< int64_t > values)
{
   graph_attrs[key] = std::move(values);
}

void BatchBuilder::set_graph_attr(const std::string& key, std::vector< std::string > values)
{
   graph_attrs[key] = std::move(values);
}

void BatchBuilder::set_graph_attr(const std::string& key, int64_t value)
{
   graph_attrs[key] = value;
}

void BatchBuilder::set_graph_attr(const std::string& key, std::string value)
{
   graph_attrs[key] = std::move(value);
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
   // later? Or store as single flattened vector?
   // We stick to storing separate src and dst index columns for now as they are easier to build.
   // Construct keys: "src_type|rel_type|dst_type/edge_index_0"
   std::string edge_key_base;
   edge_key_base.reserve(src_type.size() + rel_type.size() + dst_type.size() + 2);
   edge_key_base.append(src_type);
   edge_key_base.push_back('|');
   edge_key_base.append(rel_type);
   edge_key_base.push_back('|');
   edge_key_base.append(dst_type);

   std::string src_key;
   src_key.reserve(edge_key_base.size() + 13);
   src_key.append(edge_key_base);
   src_key.append("/edge_index_0");
   std::string dst_key;
   dst_key.reserve(edge_key_base.size() + 13);
   dst_key.append(edge_key_base);
   dst_key.append("/edge_index_1");

   auto& col_src = get_column< int64_t >(src_key, 1);
   auto& col_dst = get_column< int64_t >(dst_key, 1);

   int64_t src_offset = node_offsets.try_emplace(src_type, 0).first->second;
   int64_t dst_offset = node_offsets.try_emplace(dst_type, 0).first->second;

   col_src.reserve(col_src.size() + src_indices.size());
   col_dst.reserve(col_dst.size() + dst_indices.size());

   // Apply offsets and push
   for(auto idx : src_indices)
      col_src.emplace_back(idx + src_offset);
   for(auto idx : dst_indices)
      col_dst.emplace_back(idx + dst_offset);
}

void BatchBuilder::add_edge_features(
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type,
   const std::string& attr_name,
   std::span< const float > data,
   int feature_dim
)
{
   std::string key;
   key.reserve(src_type.size() + rel_type.size() + dst_type.size() + attr_name.size() + 4);
   key.append(src_type);
   key.push_back('|');
   key.append(rel_type);
   key.push_back('|');
   key.append(dst_type);
   key.push_back('/');
   key.append(attr_name);
   auto& col = get_column< float >(key, feature_dim);
   col.insert(col.end(), data.begin(), data.end());
}

void BatchBuilder::next_graph()
{
   for(auto& [ntype, count] : current_node_counts) {
      auto& offset = node_offsets.try_emplace(ntype, 0).first->second;
      offset += count;

      auto& p = ptrs.try_emplace(ntype, std::vector< int64_t >{}).first->second;
      if(p.empty()) {
         p.emplace_back(0);
      }
      p.emplace_back(offset);

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

template < typename T >
std::vector< T >* heap_vector(std::vector< T >&& vec, bool compact)
{
   auto* heap_vec = new std::vector< T >(std::move(vec));
   if(compact) {
      if(heap_vec->capacity() != heap_vec->size()) {
         std::vector< T > tight(heap_vec->begin(), heap_vec->end());
         *heap_vec = std::move(tight);
      }
   } else {
      heap_vec->shrink_to_fit();
   }
   return heap_vec;
}

template < typename T >
auto vector_to_1d_ndarray(std::vector< T >&& vec, bool compact = false)
{
   auto* heap_vec = heap_vector(std::move(vec), compact);
   size_t shape[1] = {heap_vec->size()};
   nb::capsule owner(heap_vec, vector_deleter< T >);
   return nb::ndarray< nb::numpy, T, nb::shape< -1 > >(heap_vec->data(), 1, shape, owner);
}

template < typename T >
auto vector_to_2d_ndarray(std::vector< T >&& vec, size_t rows, size_t cols, bool compact = false)
{
   auto* heap_vec = heap_vector(std::move(vec), compact);
   size_t shape[2] = {rows, cols};
   nb::capsule owner(heap_vec, vector_deleter< T >);
   return nb::ndarray< nb::numpy, T, nb::shape< -1, -1 > >(heap_vec->data(), 2, shape, owner);
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

            std::vector< ScalarType > vec = FWD(items);
            size_t size = vec.size();
            if(is_edge_index) {
               out[key.c_str()] = vector_to_1d_ndarray(std::move(vec));
               return;
            }

            int dim = col.dim;
            size_t num_rows = dim > 0 ? size / dim : 0;
            out[key.c_str()] = vector_to_2d_ndarray(
               std::move(vec), num_rows, static_cast< size_t >(dim)
            );
         },
         col.data
      );
   }

   // Also export Ptr columns (converting them to tensor columns first
   // essentially)
   for(auto& [ntype, p_vec] : ptrs) {
      auto tensor = vector_to_1d_ndarray(std::move(p_vec), true);
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
      if(not node_counts.empty()) {
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
      if(key.size() >= 4 and key.compare(key.size() - 4, 4, "/ptr") == 0) {
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
      if(not parts.src.is_valid() || not parts.dst.is_valid()) {
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
      if(not has_x) {
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
            if(ptr_it == ptr_vectors.end() || ptr_it->second.size() < 2) {
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
      if(not node_counts.empty()) {
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

   std::vector< NodeTensorSpec > node_specs;
   struct EdgeTensorKeySpec {
      EdgeType edge_type;
      std::string attr;
      std::string part;
      std::string key;
   };
   std::vector< EdgeTensorKeySpec > edge_specs;
   std::set< EdgeType > edge_types_set;

   for(const auto& [key, col] : columns) {
      (void) col;  // silence unused variable warning
      const auto slash = key.find('/');
      if(slash == std::string::npos) {
         continue;
      }
      const bool is_edge = key.find('|') != std::string::npos;
      if(not is_edge) {
         node_specs.push_back(
            NodeTensorSpec{
               key.substr(0, slash),
               key.substr(slash + 1),
               key,
            }
         );
         continue;
      }
      const std::string base = key.substr(0, slash);
      const std::string attr = key.substr(slash + 1);
      const auto first = base.find('|');
      if(first == std::string::npos) {
         continue;
      }
      const auto second = base.find('|', first + 1);
      if(second == std::string::npos) {
         continue;
      }
      EdgeType edge_key{
         base.substr(0, first),
         base.substr(first + 1, second - first - 1),
         base.substr(second + 1),
      };
      edge_types_set.insert(edge_key);

      std::string part;
      std::string attr_name = attr;
      constexpr std::string_view kEdgeIndexPrefix = "edge_index_";
      if(attr.rfind(kEdgeIndexPrefix, 0) == 0) {
         attr_name = "edge_index";
         part = attr.substr(kEdgeIndexPrefix.size());
      }
      edge_specs.push_back(
         EdgeTensorKeySpec{
            edge_key,
            attr_name,
            part,
            key,
         }
      );
   }

   for(const auto& [node_type, ptr] : ptr_vectors) {
      (void) ptr;
      node_specs.push_back(NodeTensorSpec{node_type, "ptr", node_type + "/ptr"});
      node_specs.push_back(NodeTensorSpec{node_type, "batch", node_type + "/batch"});
   }

   std::sort(node_specs.begin(), node_specs.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.key < rhs.key;
   });
   std::sort(edge_specs.begin(), edge_specs.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.key < rhs.key;
   });

   std::vector< EdgeType > edge_types;
   edge_types.reserve(edge_types_set.size());
   for(const auto& entry : edge_types_set) {
      edge_types.push_back(entry);
   }
   std::map< EdgeType, int > edge_type_ids;
   for(size_t idx = 0; idx < edge_types.size(); ++idx) {
      edge_type_ids[edge_types[idx]] = static_cast< int >(idx);
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

   if(not graph_attrs.empty()) {
      nb::dict graph_attrs_dict;
      for(const auto& [key, value] : graph_attrs) {
         std::visit([&](const auto& v) { graph_attrs_dict[key.c_str()] = nb::cast(v); }, value);
      }
      out["graph_attrs"] = graph_attrs_dict;
   }

   std::vector< std::string > node_types;
   node_types.reserve(node_counts.size());
   for(const auto& [node_type, count] : node_counts) {
      (void) count;
      node_types.push_back(node_type);
   }

   std::vector< EdgeTensorSpec > edge_tensor_specs;
   edge_tensor_specs.reserve(edge_specs.size());
   for(const auto& spec : edge_specs) {
      const auto it = edge_type_ids.find(spec.edge_type);
      if(it == edge_type_ids.end()) {
         throw std::invalid_argument("Edge tensor spec references unknown edge type");
      }
      EdgeTensorSpec out_spec;
      out_spec.edge_type = it->second;
      out_spec.attr = spec.attr;
      out_spec.key = spec.key;
      out_spec.part = spec.part;
      edge_tensor_specs.push_back(std::move(out_spec));
   }

   Schema schema;
   schema.version = 1;
   schema.graph_kind = graph_kind;
   schema.node_types = std::move(node_types);
   schema.edge_types = std::move(edge_types);
   schema.node_tensors = std::move(node_specs);
   schema.edge_tensors = std::move(edge_tensor_specs);
   schema.flags = schema_flags;
   schema.validate();

   out["schema"] = schema.to_dict();
   return out;
}

}  // namespace mifrost
