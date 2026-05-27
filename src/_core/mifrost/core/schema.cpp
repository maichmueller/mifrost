#include "schema.hpp"

#include <absl/container/btree_map.h>

#include <algorithm>
#include <set>
#include <stdexcept>

namespace mifrost {

void Schema::validate_base() const
{
   if(version <= 0) {
      throw std::invalid_argument("Schema version must be positive");
   }
   if(graph_kind.empty()) {
      throw std::invalid_argument("Schema graph_kind must be set");
   }
   absl::btree_map< int, std::set< std::string > > edge_index_components;
   for(const auto& spec : node_tensors) {
      if(spec.node_type.empty() or spec.attr.empty() or spec.key.empty()) {
         throw std::invalid_argument("Schema node_tensors contain empty fields");
      }
   }
   for(const auto& spec : edge_tensors) {
      if(spec.edge_type < 0 or static_cast< size_t >(spec.edge_type) >= edge_types.size()) {
         throw std::invalid_argument("Schema edge_tensors reference invalid edge_type index");
      }
      if(spec.attr.empty() or spec.key.empty()) {
         throw std::invalid_argument("Schema edge_tensors contain empty fields");
      }
      if(spec.attr == "edge_index") {
         if(spec.part.empty()) {
            throw std::invalid_argument("Schema edge_index tensors must define part");
         }
         if(spec.part != "0" and spec.part != "1") {
            throw std::invalid_argument("Schema edge_index components must be '0' or '1'");
         }
         edge_index_components[spec.edge_type].insert(spec.part);
      }
   }
   for(const auto& [edge_type, components] : edge_index_components) {
      if(components.count("0") == 0 or components.count("1") == 0) {
         throw std::invalid_argument("Schema edge_index must include both components '0' and '1'");
      }
   }
   for(const auto& spec : graph_tensors) {
      if(spec.attr.empty() or spec.key.empty()) {
         throw std::invalid_argument("Schema graph_tensors contain empty fields");
      }
      if(spec.mode == GraphFieldMode::RAGGED_CAT and spec.ptr_key.empty()) {
         throw std::invalid_argument("Schema ragged graph_tensors require ptr_key");
      }
      if(spec.mode != GraphFieldMode::RAGGED_CAT and not spec.ptr_key.empty()) {
         throw std::invalid_argument("Schema non-ragged graph_tensors must not define ptr_key");
      }
      if(spec.dim <= 0) {
         throw std::invalid_argument("Schema graph_tensors require dim > 0");
      }
      validate_graph_field_spec(
         spec.attr,
         GraphFieldSpec{
            .dtype = spec.dtype,
            .mode = spec.mode,
            .dim = spec.dim,
            .cat_dim = spec.cat_dim,
            .inc = spec.inc,
         }
      );
   }
}

void Schema::validate() const
{
   validate_base();
   if(graph_kind != "hetero" and graph_kind != "homo" and graph_kind != "flat") {
      throw std::invalid_argument("Schema graph_kind must be 'hetero', 'homo', or 'flat'");
   }
   validate_history();
}

void Schema::validate_history() const
{
   const auto it = flags.find("history");
   if(it == flags.end() or not it->second) {
      return;
   }

   if(graph_kind != "hetero") {
      throw std::invalid_argument("History encoding requires graph_kind='hetero'");
   }

   if(std::ranges::find(node_types, "history") == node_types.end()) {
      throw std::invalid_argument("History encoding requires 'history' node type");
   }

   const auto has_history_dt = std::ranges::any_of(node_tensors, [](const NodeTensorSpec& spec) {
      return spec.node_type == "history" and spec.attr == "history_dt";
   });
   if(not has_history_dt) {
      throw std::invalid_argument("History encoding requires history/history_dt tensor");
   }

   std::vector< int > history_edge_ids;
   history_edge_ids.reserve(edge_types.size());
   for(size_t idx = 0; idx < edge_types.size(); ++idx) {
      const auto& edge_type = edge_types[idx];
      if(edge_type.src == "history" or edge_type.dst == "history") {
         history_edge_ids.push_back(static_cast< int >(idx));
      }
   }

   if(history_edge_ids.empty()) {
      throw std::invalid_argument("History encoding requires history link edge types");
   }

   bool has_pair = false;
   for(const auto& edge_type : edge_types) {
      if(edge_type.src != "history") {
         continue;
      }
      const bool has_reverse = std::ranges::any_of(edge_types, [&](const EdgeType& candidate) {
         return candidate.src == edge_type.dst and candidate.dst == "history"
                and candidate.rel == edge_type.rel;
      });
      if(has_reverse) {
         has_pair = true;
         break;
      }
   }

   if(not has_pair) {
      throw std::invalid_argument("History encoding requires bidirectional history link edges");
   }

   absl::btree_map< int, std::set< std::string > > parts_by_edge;
   for(const auto& spec : edge_tensors) {
      if(spec.attr == "edge_index") {
         parts_by_edge[spec.edge_type].insert(spec.part);
      }
   }
   for(const auto edge_id : history_edge_ids) {
      const auto iterator = parts_by_edge.find(edge_id);
      if(iterator == parts_by_edge.end() or not iterator->second.contains("0")
         or not iterator->second.contains("1")) {
         throw std::invalid_argument(
            "History encoding requires edge_index components for history link edges"
         );
      }
   }
}

}  // namespace mifrost
