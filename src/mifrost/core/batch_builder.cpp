#include "batch_builder.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace mifrost {

BatchBuilder::BatchBuilder() {}

void BatchBuilder::add_node_features(const std::string &node_type,
                                     const std::string &attr_name,
                                     std::span<const float> data,
                                     int feature_dim) {
  std::string key = node_type + "/" + attr_name;
  auto &col = get_column<float>(key, feature_dim);
  col.insert(col.end(), data.begin(), data.end());

  int64_t num_nodes = data.size() / feature_dim;
  if (current_node_counts[node_type] < num_nodes) {
    current_node_counts[node_type] = num_nodes;
  }
}

void BatchBuilder::add_edges(const std::string &src_type,
                             const std::string &rel_type,
                             const std::string &dst_type,
                             std::span<const int64_t> src_indices,
                             std::span<const int64_t> dst_indices) {
  if (src_indices.size() != dst_indices.size()) {
    throw std::invalid_argument("src and dst indices must have same length");
  }

  // Key Convention: Store source/dest columns separately for simplified concats
  // later? Or store as single flattened vector? Let's stick to storing separate
  // src and dst index columns for now as they are easier to build. Construct
  // keys: "src_type|rel_type|dst_type/edge_index_0"
  std::string edge_key_base = src_type + "|" + rel_type + "|" + dst_type;

  auto &col_src = get_column<int64_t>(edge_key_base + "/edge_index_0", 1);
  auto &col_dst = get_column<int64_t>(edge_key_base + "/edge_index_1", 1);

  int64_t src_offset = node_offsets[src_type];
  int64_t dst_offset = node_offsets[dst_type];

  col_src.reserve(col_src.size() + src_indices.size());
  col_dst.reserve(col_dst.size() + dst_indices.size());

  // Apply offsets and push
  for (auto idx : src_indices)
    col_src.push_back(idx + src_offset);
  for (auto idx : dst_indices)
    col_dst.push_back(idx + dst_offset);
}

void BatchBuilder::next_graph() {
  for (auto &[ntype, count] : current_node_counts) {
    node_offsets[ntype] += count;

    auto &p = ptrs[ntype];
    if (p.empty())
      p.push_back(0);
    p.push_back(node_offsets[ntype]);

    count = 0;
  }
  // Batch indices tracking could go here if we want homogeneous batch vector
  current_graph_idx++;
}

// --- DLPack Owner Capsule ---

template <typename T> void vector_deleter(void *p) noexcept {
  delete static_cast<std::vector<T> *>(p);
}

// --- Build / Export ---

nb::dict BatchBuilder::build() {
  nb::dict out;

  for (auto &[key, col] : columns) {
    std::visit(
        [&](auto &&items) {
          using VectorType = std::decay_t<decltype(items)>;
          using ScalarType = typename VectorType::value_type;

          // Move the vector onto the heap so the capsule can own it
          auto *heap_vec = new VectorType(std::move(items));

          size_t size = heap_vec->size();
          int dim = col.dim;
          size_t num_rows = dim > 0 ? size / dim : 0;

          size_t shape[2] = {num_rows, (size_t)dim};

          // Create nanobind array that takes ownership of the heap_vec
          // We use a capsule to manage the lifetime of heap_vec
          nb::capsule owner(heap_vec, [](void *p) noexcept {
            delete static_cast<VectorType *>(p);
          });

          auto tensor =
              nb::ndarray<nb::numpy, ScalarType, nb::shape<-1, -1>>(
                  heap_vec->data(), 2, shape, owner);

          out[key.c_str()] = tensor;
        },
        col.data);
  }

  // Also export Ptr columns (converting them to tensor columns first
  // essentially)
  for (auto &[ntype, p_vec] : ptrs) {
    auto *heap_vec = new std::vector<int64_t>(std::move(p_vec));
    size_t shape[1] = {heap_vec->size()};

    nb::capsule owner(heap_vec, [](void *p) noexcept {
      delete static_cast<std::vector<int64_t> *>(p);
    });

    auto tensor = nb::ndarray<nb::numpy, int64_t, nb::shape<-1>>(
        heap_vec->data(), 1, shape, owner);

    std::string key = ntype + "/ptr";
    out[key.c_str()] = tensor;
  }

  return out;
}

} // namespace mifrost
