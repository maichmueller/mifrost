#pragma once

#include <nanobind/nanobind.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace mifrost {

namespace nb = nanobind;

/**
 * @brief Canonical hetero edge type descriptor.
 */
struct EdgeType {
   std::string src;
   std::string rel;
   std::string dst;

   bool operator<(const EdgeType& other) const
   {
      return std::tie(src, rel, dst) < std::tie(other.src, other.rel, other.dst);
   }
};

/**
 * @brief Mapping entry for one node tensor in the flat parts tensor dict.
 */
struct NodeTensorSpec {
   std::string node_type;
   std::string attr;
   std::string key;
};

/**
 * @brief Mapping entry for one edge tensor in the flat parts tensor dict.
 */
struct EdgeTensorSpec {
   int edge_type = -1;
   std::string attr;
   std::string key;
   std::string part;
};

/**
 * @brief Versioned schema for normalized encoder parts payloads.
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
   std::map< std::string, bool > flags;

   Schema();

   virtual ~Schema() = default;

   /**
    * @brief Validate schema invariants.
    *
    * Override in derived schemas to apply additional encoder-specific checks.
    */
   virtual void validate() const;

   /// Serialize to Python dictionary form.
   nb::dict to_dict() const;
   /// Parse schema from Python dictionary form.
   static Schema from_dict(const nb::dict& schema);

  protected:
   /// Base validation shared by all schema variants.
   void validate_base() const;
};

}  // namespace mifrost
