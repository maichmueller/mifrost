#include "schema.hpp"

#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

namespace mifrost {

Schema::Schema() = default;

void Schema::validate_base() const
{
   if(version <= 0) {
      throw std::invalid_argument("Schema version must be positive");
   }
   if(graph_kind.empty()) {
      throw std::invalid_argument("Schema graph_kind must be set");
   }
   for(const auto& spec : node_tensors) {
      if(spec.node_type.empty() || spec.attr.empty() || spec.key.empty()) {
         throw std::invalid_argument("Schema node_tensors contain empty fields");
      }
   }
   for(const auto& spec : edge_tensors) {
      if(spec.edge_type < 0 || static_cast< size_t >(spec.edge_type) >= edge_types.size()) {
         throw std::invalid_argument("Schema edge_tensors reference invalid edge_type index");
      }
      if(spec.attr.empty() || spec.key.empty()) {
         throw std::invalid_argument("Schema edge_tensors contain empty fields");
      }
   }
}

void Schema::validate() const
{
   validate_base();
   if(graph_kind != "hetero" && graph_kind != "homo") {
      throw std::invalid_argument("Schema graph_kind must be 'hetero' or 'homo'");
   }
}

nb::dict Schema::to_dict() const
{
   nb::dict out;
   out["version"] = version;
   out["graph_kind"] = graph_kind;

   nb::list node_type_list;
   for(const auto& node_type : node_types) {
      node_type_list.append(node_type);
   }
   out["node_types"] = node_type_list;

   nb::list edge_type_list;
   for(const auto& [src, rel, dst] : edge_types) {
      nb::dict entry;
      entry["src"] = src;
      entry["rel"] = rel;
      entry["dst"] = dst;
      edge_type_list.append(entry);
   }
   out["edge_types"] = edge_type_list;

   nb::list node_tensor_list;
   for(const auto& [node_type, attr, key] : node_tensors) {
      nb::dict entry;
      entry["node_type"] = node_type;
      entry["attr"] = attr;
      entry["key"] = key;
      node_tensor_list.append(entry);
   }
   out["node_tensors"] = node_tensor_list;

   nb::list edge_tensor_list;
   for(const auto& [edge_type, attr, key, part] : edge_tensors) {
      nb::dict entry;
      entry["edge_type"] = edge_type;
      entry["attr"] = attr;
      if(! part.empty()) {
         entry["part"] = part;
      }
      entry["key"] = key;
      edge_tensor_list.append(entry);
   }
   out["edge_tensors"] = edge_tensor_list;

   nb::dict flags_dict{};
   for(const auto& [key, value] : flags) {
      flags_dict[key.c_str()] = value;
   }
   out["flags"] = flags_dict;
   out["extensions"] = nb::dict{};

   return out;
}

Schema Schema::from_dict(const nb::dict& schema)
{
   Schema out;
   if(schema.contains("version")) {
      out.version = nb::cast< int >(schema["version"]);
   }
   if(schema.contains("graph_kind")) {
      out.graph_kind = nb::cast< std::string >(schema["graph_kind"]);
   }
   if(schema.contains("node_types")) {
      out.node_types = nb::cast< std::vector< std::string > >(schema["node_types"]);
   }
   if(schema.contains("edge_types")) {
      auto edge_type_list = nb::cast< nb::list >(schema["edge_types"]);
      out.edge_types.reserve(edge_type_list.size());
      for(nb::handle entry_handle : edge_type_list) {
         auto entry = nb::cast< nb::dict >(entry_handle);
         out.edge_types.emplace_back(
            EdgeType{
               .src = nb::cast< std::string >(entry["src"]),
               .rel = nb::cast< std::string >(entry["rel"]),
               .dst = nb::cast< std::string >(entry["dst"])
            }
         );
      }
   }
   if(schema.contains("node_tensors")) {
      auto node_tensor_list = nb::cast< nb::list >(schema["node_tensors"]);
      out.node_tensors.reserve(node_tensor_list.size());
      for(nb::handle entry_handle : node_tensor_list) {
         auto entry = nb::cast< nb::dict >(entry_handle);
         out.node_tensors.emplace_back(
            NodeTensorSpec{
               .node_type = nb::cast< std::string >(entry["node_type"]),
               .attr = nb::cast< std::string >(entry["attr"]),
               .key = nb::cast< std::string >(entry["key"])
            }
         );
      }
   }
   if(schema.contains("edge_tensors")) {
      auto edge_tensor_list = nb::cast< nb::list >(schema["edge_tensors"]);
      out.edge_tensors.reserve(edge_tensor_list.size());
      for(nb::handle entry_handle : edge_tensor_list) {
         auto entry = nb::cast< nb::dict >(entry_handle);
         out.edge_tensors.emplace_back(
            EdgeTensorSpec{
               .edge_type = nb::cast< int >(entry["edge_type"]),
               .attr = nb::cast< std::string >(entry["attr"]),
               .key = nb::cast< std::string >(entry["key"]),
               .part = entry.contains("part") ? nb::cast< std::string >(entry["part"]) : ""
            }
         );
      }
   }
   if(schema.contains("flags")) {
      out.flags = nb::cast< std::map< std::string, bool > >(schema["flags"]);
   }
   out.validate();
   return out;
}

}  // namespace mifrost
