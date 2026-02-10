#pragma once

#include <absl/container/btree_map.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "common_types.hpp"
#include "schema.hpp"

namespace mifrost {

namespace nb = nanobind;

/**
 * @brief Container for columnar graph data that grows dynamically.
 *
 * Mirrors the logic of Python's PygBatchBuilder but optimized for C++:
 * - Stores data in std::vector (contiguous memory)
 * - Supports zero-copy transfer to Python via DLPack
 * - Handles automatic offsetting of edge indices
 */
class BatchBuilder {
  public:
   /// Floating-point tensor column storage.
   using FloatCol = std::vector< float >;
   /// Integer tensor column storage.
   using LongCol = std::vector< int64_t >;
   /// Variant over supported tensor scalar storage backends.
   using ColumnData = std::variant< FloatCol, LongCol >;

   /// One logical tensor column.
   struct Column {
      /// Contiguous scalar storage.
      ColumnData data;
      /// Feature dimension for this column.
      int dim = 1;
   };

   /// Native (non-Python) representation of normalized batch encoding.
   struct BatchEncoding;

   // --- Graph Structure Tracking ---

   /// Per-node-type cumulative node counts in the current (open) graph.
   hash_map< std::string, int64_t > current_node_counts;

   /// Per-node-type offset of the current graph start in the batch.
   hash_map< std::string, int64_t > node_offsets;

   /// Node feature dimensions per node type (used to synthesize empty x tensors).
   hash_map< std::string, int > node_feature_dims;

   /// Optional node names per node type (metadata path).
   hash_map< std::string, std::vector< std::string > > node_names;

   /// Optional per-graph object names (metadata path).
   std::vector< std::string > object_names;

   /// Schema graph kind ("hetero" or "homo").
   std::string graph_kind;

   /// Schema feature flags exposed to Python.
   absl::btree_map< std::string, bool > schema_flags;

   /// PyG-style ptr tracking (per node type).
   hash_map< std::string, std::vector< int64_t > > ptrs;
   /// Optional homogeneous batch index cache.
   std::vector< int64_t > batch_indices;

   /// Supported graph-level attribute value types.
   using GraphAttrValue = std::
      variant< int64_t, std::string, std::vector< int64_t >, std::vector< std::string > >;
   /// Graph-level attributes forwarded to Python metadata.
   hash_map< std::string, GraphAttrValue > graph_attrs;

   /// Flat tensor column storage keyed by schema keys.
   hash_map< std::string, Column > columns;

   /// Number of committed graphs.
   int64_t current_graph_idx = 0;

  public:
   /// Create an empty builder.
   BatchBuilder();
   /// Reset builder state to a fresh, empty instance.
   void reset();

   // --- Data Ingestion ---

   /**
    * @brief Add node features for a specific node type in the current graph.
    * @param node_type e.g. "atom"
    * @param attr_name e.g. "x"
    * @param data Raw data span (num_nodes * feature_dim)
    * @param feature_dim Dimension of feature vector (e.g. 1 for simple attributes, N for
    * embeddings)
    */
   void add_node_features(
      const std::string& node_type,
      const std::string& attr_name,
      std::span< const float > data,
      int feature_dim
   );

   /**
    * @brief Add edges for a specific edge type.
    * @param src_type Source node type
    * @param rel_type Relation name
    * @param dst_type Destination node type
    * @param src_indices Source indices (local to current graph)
    * @param dst_indices Destination indices (local to current graph)
    *
    * Automatically applies offsets based on node_offsets.
    */
   void add_edges(
      const std::string& src_type,
      const std::string& rel_type,
      const std::string& dst_type,
      std::span< const int64_t > src_indices,
      std::span< const int64_t > dst_indices
   );
   /**
    * @brief Add a single edge for a specific edge type.
    *
    * Applies the same node-type offsets as `add_edges`.
    */
   void add_edge(
      const std::string& src_type,
      const std::string& rel_type,
      const std::string& dst_type,
      int64_t src_index,
      int64_t dst_index
   );
   void add_edge_features(
      const std::string& src_type,
      const std::string& rel_type,
      const std::string& dst_type,
      const std::string& attr_name,
      std::span< const float > data,
      int feature_dim
   );

   /**
    * @brief Commit the current graph to the batch.
    *
    * Updates offsets = offsets + current_node_counts
    * Resets current_node_counts = 0
    * Appends offsets to ptrs
    * Increments current_graph_idx
    */
   void next_graph();

   /**
    * @brief Finalize and return normalized batch encoding as native C++ data.
    *
    * This consumes the builder state and resets the builder. Use for stream
    * caching or C++ assembly.
    */
   BatchEncoding build();
   /**
    * @brief Finalize and return a PyG Batch/HeteroData object.
    *
    * This call consumes internal tensor buffers (moved into Python-owned ndarrays/tensors).
    * The builder is reset after export.
    */
   nb::object build_pyg();

   /**
    * @brief Append one graph worth of batch encoding into the current batch.
    *
    * Expects batch encoding with num_graphs == 1.
    */
   void append_batch_encoding(const BatchEncoding& batch_encoding);
   /// Load builder state from an existing batch encoding (copying payload).
   void load_from_batch_encoding(const BatchEncoding& batch_encoding);
   /// Load builder state from an existing batch encoding (moving payload).
   void load_from_batch_encoding(BatchEncoding&& batch_encoding);

   /// Set feature dim for a node type (used for implicit empty x tensors).
   void set_node_feature_dim(const std::string& node_type, int dim);
   /// Add a node count delta for a node type.
   void add_nodes(const std::string& node_type, int64_t count);
   /// Register an edge type even if no concrete edges exist yet.
   void ensure_edge_type(
      const std::string& src_type,
      const std::string& rel_type,
      const std::string& dst_type
   );
   /// Set node names metadata for one node type.
   void set_node_names(const std::string& node_type, std::vector< std::string > names);
   /// Set object names metadata.
   void set_object_names(std::vector< std::string > names);
   /// Set schema graph kind ("hetero"/"homo").
   void set_graph_kind(std::string kind);
   /// Set schema flag.
   void set_schema_flag(const std::string& key, bool value);
   /// Set integer vector graph attribute.
   void set_graph_attr(const std::string& key, std::vector< int64_t > values);
   /// Set string vector graph attribute.
   void set_graph_attr(const std::string& key, std::vector< std::string > values);
   /// Set integer graph attribute.
   void set_graph_attr(const std::string& key, int64_t value);
   /// Set string graph attribute.
   void set_graph_attr(const std::string& key, std::string value);

  private:
   /// Get or create a typed column with the requested feature dimension.
   template < typename T >
   std::vector< T >& get_column(const std::string& key, int dim);

   /**
    * @brief Build tensor dictionary by moving out internal vector storage.
    *
    * Destructive export: column and ptr vectors are moved into Python-owned
    * ndarray capsules for zero-copy transfer.
    */
   nb::dict build_dict();
};

template < typename T >
std::vector< T >& BatchBuilder::get_column(const std::string& key, int dim)
{
   auto it = columns.find(key);
   if(it == columns.end()) {
      if constexpr(std::is_same_v< T, float >) {
         it = columns.try_emplace(key, Column{FloatCol{}, dim}).first;
      } else if constexpr(std::is_same_v< T, int64_t >) {
         it = columns.try_emplace(key, Column{LongCol{}, dim}).first;
      } else {
         static_assert(
            std::is_same_v< T, float > || std::is_same_v< T, int64_t >, "Unsupported column type"
         );
      }
   } else if(it->second.dim != dim) {
      throw std::invalid_argument("Column dim mismatch for key: " + key);
   }

   if constexpr(std::is_same_v< T, float >) {
      return std::get< FloatCol >(it->second.data);
   } else if constexpr(std::is_same_v< T, int64_t >) {
      return std::get< LongCol >(it->second.data);
   } else {
      throw std::logic_error("Unsupported column type");
   }
}

/**
 * @brief Native (non-Python) representation of normalized batch encoding.
 */
struct BatchBuilder::BatchEncoding {
   hash_map< std::string, Column > columns;
   hash_map< std::string, std::vector< std::string > > node_names;
   std::vector< std::string > object_names;
   hash_map< std::string, int > node_feature_dims;
   hash_map< std::string, GraphAttrValue > graph_attrs;
   hash_map< std::string, std::vector< int64_t > > ptrs;
   absl::btree_map< std::string, bool > schema_flags;
   std::string graph_kind;
   int64_t num_graphs = 0;
   absl::btree_map< std::string, int64_t > node_counts;
   Schema schema;
};

}  // namespace mifrost
