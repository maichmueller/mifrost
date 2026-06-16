#pragma once

#include "mifrost/init_batch_encoding.hpp"

using namespace nanobind::literals;

namespace mifrost {

/**
 * @brief Cached hetero facade over a native BatchEncoding.
 *
 * Exposes dictionary-style tensor views (`x_dict`, `edge_index_dict`, etc.)
 * without rebuilding PyG data objects on every access.
 */
class HeteroBatchEncodingView {
  public:
   /**
    * @brief Bind the facade to a BatchEncoding Python owner object.
    */
   explicit HeteroBatchEncodingView(nb::object owner);

   /// @brief Number of graphs represented by the underlying batch.
   [[nodiscard]] int64_t num_graphs() const { return encoding_->num_graphs; }
   /// @brief Total number of nodes across the batch.
   [[nodiscard]] int64_t num_nodes() const { return batch_encoding_num_nodes(*encoding_); }
   /// @brief Total number of edges across the batch.
   [[nodiscard]] int64_t num_edges() const { return batch_encoding_num_edges(*encoding_); }
   /// @brief Graph kind string (`"hetero"` / `"homo"`).
   [[nodiscard]] std::string graph_kind() const { return encoding_->graph_kind; }

   /// @brief Ordered node type names in schema order.
   [[nodiscard]] std::vector< std::string > node_types() const
   {
      return encoding_->schema.node_types;
   }

   /// @brief Ordered edge type tuples in schema order.
   [[nodiscard]] nb::list edge_types() const { return batch_encoding_edge_types(*encoding_); }
   /// @brief Object names attached to the batch, if present.
   [[nodiscard]] std::vector< std::string > object_names() const { return encoding_->object_names; }
   /// @brief Original ``BatchEncoding`` backing this view.
   [[nodiscard]] nb::object base() const { return owner_; }

   /// @brief Mapping `node_type -> x` tensor.
   nb::object x_dict();
   /// @brief Mapping `edge_type -> edge_index` tensor.
   nb::object edge_index_dict();
   /// @brief Mapping `node_type -> batch` tensor.
   nb::object batch_dict();
   /// @brief Mapping `node_type -> ptr` tensor.
   nb::object ptr_dict();
   /// @brief Mapping `edge_type -> edge_attr` tensor where available.
   nb::object edge_attr_dict();
   /**
    * @brief Move facade tensors to `device` and refresh caches eagerly.
    *
    * No-op when `device` is `None`.
    */
   void set_device(nb::handle device);

  private:
   /// @brief Drop all cached tensors/mappings.
   void clear_caches();
   /// @brief Rebuild cache entries eagerly for stable post-`to()` access.
   void prewarm_caches();

   /// @brief Check whether a native tensor key exists.
   [[nodiscard]] bool has_tensor(const std::string& key) const
   {
      return batch_encoding_has_native_tensor(*encoding_, key);
   }

   /// @brief Resolve and cache a tensor by native key.
   [[nodiscard]] nb::object tensor(const std::string& key);

   nb::object owner_;
   BatchBuilder::BatchEncoding* encoding_ = nullptr;
   nb::dict tensor_cache_;
   nb::object x_dict_cache_;
   nb::object edge_index_dict_cache_;
   nb::object batch_dict_cache_;
   nb::object ptr_dict_cache_;
   nb::object edge_attr_dict_cache_;
};

/**
 * @brief Cached homo facade over a native BatchEncoding.
 *
 * Exposes single-tensor properties (`x`, `edge_index`, `batch`, `ptr`,
 * `edge_attr`) for homogeneous graph workflows.
 */
class HomoBatchEncodingView {
  public:
   /**
    * @brief Bind the facade to a BatchEncoding Python owner object.
    */
   explicit HomoBatchEncodingView(nb::object owner);
   /// @brief Number of graphs represented by the underlying batch.
   [[nodiscard]] int64_t num_graphs() const { return encoding_->num_graphs; }
   /// @brief Total number of nodes across the batch.
   [[nodiscard]] int64_t num_nodes() const { return batch_encoding_num_nodes(*encoding_); }
   /// @brief Total number of edges across the batch.
   [[nodiscard]] int64_t num_edges() const { return batch_encoding_num_edges(*encoding_); }
   /// @brief Graph kind string (`"hetero"` / `"homo"`).
   [[nodiscard]] std::string graph_kind() const { return encoding_->graph_kind; }
   /// @brief Ordered node type names in schema order.
   [[nodiscard]] std::vector< std::string > node_types() const
   {
      return encoding_->schema.node_types;
   }
   /// @brief Ordered edge type tuples in schema order.
   [[nodiscard]] nb::list edge_types() const { return batch_encoding_edge_types(*encoding_); }
   /// @brief Object names attached to the batch, if present.
   [[nodiscard]] std::vector< std::string > object_names() const { return encoding_->object_names; }
   /// @brief Original ``BatchEncoding`` backing this view.
   [[nodiscard]] nb::object base() const { return owner_; }

   /// @brief Node feature tensor (`x`) or `None` if unavailable.
   nb::object x();

   /// @brief Edge index tensor `[2, E]` or `None` if unavailable.
   nb::object edge_index();

   /// @brief Batch assignment tensor or `None` if unavailable.
   nb::object batch();

   /// @brief Pointer tensor delimiting graph segments or `None` if unavailable.
   nb::object ptr();

   /// @brief Edge attribute tensor or `None` if unavailable.
   nb::object edge_attr();

   /**
    * @brief Move facade tensors to `device` and refresh caches eagerly.
    *
    * No-op when `device` is `None`.
    */
   void set_device(nb::handle device);

  private:
   /// @brief Drop all cached tensors and readiness flags.
   void clear_caches();

   /// @brief Rebuild cache entries eagerly for stable post-`to()` access.
   void prewarm_caches();

   /// @brief Check whether a native tensor key exists.
   bool has_tensor(const std::string& key)
   {
      return batch_encoding_has_native_tensor(*encoding_, key);
   }

   /// @brief Resolve and cache a tensor by native key.
   [[nodiscard]] nb::object tensor(const std::string& key);

   nb::object owner_;
   BatchBuilder::BatchEncoding* encoding_ = nullptr;
   nb::dict tensor_cache_;
   bool x_ready_ = false;
   bool edge_index_ready_ = false;
   bool batch_ready_ = false;
   bool ptr_ready_ = false;
   bool edge_attr_ready_ = false;
   nb::object x_cache_;
   nb::object edge_index_cache_;
   nb::object batch_cache_;
   nb::object ptr_cache_;
   nb::object edge_attr_cache_;
};

void register_batch_encoding_views(nb::module_& m);

}  // namespace mifrost
