#pragma once

#include <ankerl/unordered_dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

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
   // Supported column types
   using FloatCol = std::vector< float >;
   using LongCol = std::vector< int64_t >;
   // Extend with other types if needed (e.g. half float, int32)
   using ColumnData = std::variant< FloatCol, LongCol >;

   struct Column {
      ColumnData data;
      int dim = 1;  // Feature dimension (strides[1])
   };

   // --- Graph Structure Tracking ---

   // Per-node-type cumulative counts in the current batch
   // Key: node_type (e.g. "atom", "_symbol_")
   ankerl::unordered_dense::map< std::string, int64_t > current_node_counts;

   // Global offsets for the *start* of the current graph
   // (Used to shift edge indices)
   ankerl::unordered_dense::map< std::string, int64_t > node_offsets;

   // Feature dimensions per node type (used when x is implicit).
   ankerl::unordered_dense::map< std::string, int > node_feature_dims;

   // Optional node names per node type for PyG output.
   ankerl::unordered_dense::map< std::string, std::vector< std::string > > node_names;

   // Optional graph-level object names.
   std::vector< std::string > object_names;

   // Schema graph kind (e.g. "hetero" or "homo").
   std::string graph_kind;

   // Schema metadata (flags + extensions).
   std::map< std::string, bool > schema_flags;
   nb::dict schema_extensions;

   // Graph pointer (ptr) tracking.
   // For homogeneous: simple vector. For hetero: ptr per node type (PyG convention).
   ankerl::unordered_dense::map< std::string, std::vector< int64_t > > ptrs;
   std::vector< int64_t > batch_indices;  // For homogeneous case if needed

   // --- Storage ---
   // Key format: "node_type/attr_name" or "edge_type/attr_name"
   // For edges, key could be "src|rel|dst/edge_index"
   ankerl::unordered_dense::map< std::string, Column > columns;

   int64_t current_graph_idx = 0;

  public:
   BatchBuilder();

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
    * @brief Finalize and return a PyG Batch/HeteroData object.
    */
   nb::object build();
   /**
    * @brief Finalize and return graph parts for Python assembly.
    */
   nb::dict build_parts();

   void set_node_feature_dim(const std::string& node_type, int dim);
   void add_nodes(const std::string& node_type, int64_t count);
   void ensure_edge_type(
      const std::string& src_type,
      const std::string& rel_type,
      const std::string& dst_type
   );
   void set_node_names(const std::string& node_type, std::vector< std::string > names);
   void set_object_names(std::vector< std::string > names);
   void set_graph_kind(std::string kind);
   void set_schema_flag(const std::string& key, bool value);
   void set_schema_extension(const std::string& key, nb::object value);

  private:
   // Helper to get or create a column
   template < typename T >
   std::vector< T >& get_column(const std::string& key, int dim);

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

}  // namespace mifrost
