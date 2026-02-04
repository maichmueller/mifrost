#pragma once

#include <nanobind/nanobind.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace mifrost {

namespace nb = nanobind;

struct EdgeType {
   std::string src;
   std::string rel;
   std::string dst;

   bool operator<(const EdgeType& other) const
   {
      return std::tie(src, rel, dst) < std::tie(other.src, other.rel, other.dst);
   }
};

struct NodeTensorSpec {
   std::string node_type;
   std::string attr;
   std::string key;
};

struct EdgeTensorSpec {
   int edge_type = -1;
   std::string attr;
   std::string key;
   std::string part;
};

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

   virtual void validate() const;
   nb::dict to_dict() const;
   static Schema from_dict(const nb::dict& schema);

  protected:
   void validate_base() const;
};

}  // namespace mifrost
