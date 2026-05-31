#pragma once

#include <absl/container/btree_map.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "graph_fields.hpp"

namespace mifrost {

/**
 * @brief Canonical hetero edge type descriptor.
 */
struct EdgeType {
   std::string src;
   std::string rel;
   std::string dst;

   auto operator<=>(const EdgeType&) const noexcept = default;
};

inline auto as_tuple(const EdgeType& e) noexcept
{
   return std::tie(e.src, e.rel, e.dst);
}

/**
 * @brief Mapping entry for one node tensor in the flat encoding tensor dict.
 */
struct NodeTensorSpec {
   std::string node_type;
   std::string attr;
   std::string key;
};

inline auto as_tuple(const NodeTensorSpec& spec) noexcept
{
   return std::tie(spec.node_type, spec.attr, spec.key);
}

/**
 * @brief Mapping entry for one edge tensor in the flat encoding tensor dict.
 */
struct EdgeTensorSpec {
   int edge_type = -1;
   std::string attr;
   std::string key;
   std::string part;

   auto operator<=>(const EdgeTensorSpec&) const noexcept = default;
};

inline auto as_tuple(const EdgeTensorSpec& spec) noexcept
{
   return std::tie(spec.edge_type, spec.attr, spec.key, spec.part);
}

/**
 * @brief Mapping entry for one graph-level tensor in the flat encoding tensor dict.
 */
struct GraphTensorSpec {
   std::string attr;
   std::string key;
   std::string ptr_key;
   GraphFieldMode mode = GraphFieldMode::STACK;
   GraphFieldDType dtype = GraphFieldDType::F32;
   int dim = 1;
   int cat_dim = 0;
   GraphFieldInc inc{};

   auto operator<=>(const GraphTensorSpec&) const noexcept = default;
};

inline auto as_tuple(const GraphTensorSpec& spec) noexcept
{
   return std::tie(
      spec.attr, spec.key, spec.ptr_key, spec.mode, spec.dtype, spec.dim, spec.cat_dim, spec.inc
   );
}

/**
 * @brief Versioned schema for normalized encoder dictionary payloads.
 *
 * This schema defines how flat tensors map to semantic node and edge stores.
 * Python assembly uses this contract to reconstruct ``Data``/``HeteroData``.
 */
struct Schema {
   int version = 1;
   std::string graph_kind;
   std::vector< std::string > node_types;
   std::vector< EdgeType > edge_types;
   std::vector< NodeTensorSpec > node_tensors;
   std::vector< EdgeTensorSpec > edge_tensors;
   std::vector< GraphTensorSpec > graph_tensors;
   absl::btree_map< std::string, bool > flags;

   Schema() = default;
   Schema(Schema&&) = default;
   Schema(const Schema&) = default;
   Schema& operator=(Schema&&) = default;
   Schema& operator=(const Schema&) = default;
   virtual ~Schema() = default;

   /**
    * @brief Validate schema invariants.
    *
    * Override in derived schemas to apply additional encoder-specific checks.
    */
   virtual void validate() const;

  protected:
   /// Base validation shared by all schema variants.
   void validate_base() const;
   void validate_history() const;
};

}  // namespace mifrost
