#include "schema.hpp"

#include <absl/container/btree_map.h>

#include <algorithm>
#include <set>
#include <stdexcept>

#if defined(MIFROST_ENABLE_PYTHON_API)
   #include <nanobind/stl/map.h>
   #include <nanobind/stl/string.h>
   #include <nanobind/stl/vector.h>
#endif

namespace mifrost {

#if defined(MIFROST_ENABLE_PYTHON_API)
namespace {

std::string py_string(nb::handle value)
{
   return {nb::str(value).c_str()};
}

}  // namespace
#endif

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

#if defined(MIFROST_ENABLE_PYTHON_API)
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
      if(not part.empty()) {
         entry["part"] = part;
      }
      entry["key"] = key;
      edge_tensor_list.append(entry);
   }
   out["edge_tensors"] = edge_tensor_list;

   nb::list graph_tensor_list;
   for(const auto& spec : graph_tensors) {
      nb::dict entry;
      entry["attr"] = spec.attr;
      entry["key"] = spec.key;
      if(not spec.ptr_key.empty()) {
         entry["ptr_key"] = spec.ptr_key;
      }
      entry["mode"] = graph_field_mode_name(spec.mode);
      entry["dtype"] = graph_field_dtype_name(spec.dtype);
      entry["dim"] = spec.dim;
      entry["cat_dim"] = spec.cat_dim;
      nb::dict inc;
      inc["kind"] = graph_field_inc_kind_name(spec.inc.kind);
      if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
         inc["node_type"] = spec.inc.node_type;
      } else if(spec.inc.kind == GraphFieldInc::Kind::FIELD_OFFSET) {
         inc["field_key"] = spec.inc.field_key;
      }
      entry["inc"] = std::move(inc);
      graph_tensor_list.append(entry);
   }
   if(not graph_tensors.empty()) {
      out["graph_tensors"] = std::move(graph_tensor_list);
   }

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
#endif
      try {
         out.version = nb::cast< int >(schema["version"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Schema key 'version' parse failed: " + std::string(ex.what())
         );
      }
   }
   if(schema.contains("graph_kind")) {
      try {
         out.graph_kind = nb::str(schema["graph_kind"]).c_str();
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Schema key 'graph_kind' parse failed: " + std::string(ex.what())
         );
      }
   }
   if(schema.contains("node_types")) {
      try {
         auto node_types = nb::cast< nb::list >(schema["node_types"]);
         out.node_types.clear();
         out.node_types.reserve(node_types.size());
         for(nb::handle entry : node_types) {
            out.node_types.push_back(py_string(entry));
         }
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Schema key 'node_types' parse failed: " + std::string(ex.what())
         );
      }
   }
   if(schema.contains("edge_types")) {
      auto edge_type_list = nb::cast< nb::list >(schema["edge_types"]);
      out.edge_types.reserve(edge_type_list.size());
      for(nb::handle entry_handle : edge_type_list) {
         auto entry = nb::cast< nb::dict >(entry_handle);
         out.edge_types.emplace_back(
            EdgeType{
               .src = py_string(entry["src"]),
               .rel = py_string(entry["rel"]),
               .dst = py_string(entry["dst"])
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
               .node_type = py_string(entry["node_type"]),
               .attr = py_string(entry["attr"]),
               .key = py_string(entry["key"])
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
               .attr = py_string(entry["attr"]),
               .key = py_string(entry["key"]),
               .part = entry.contains("part") ? py_string(entry["part"]) : ""
            }
         );
      }
   }
   if(schema.contains("graph_tensors")) {
      auto graph_tensor_list = nb::cast< nb::list >(schema["graph_tensors"]);
      out.graph_tensors.reserve(graph_tensor_list.size());
      for(nb::handle entry_handle : graph_tensor_list) {
         auto entry = nb::cast< nb::dict >(entry_handle);
         GraphFieldInc inc{};
         if(entry.contains("inc")) {
            auto inc_entry = nb::cast< nb::dict >(entry["inc"]);
            const auto inc_kind = nb::str(inc_entry["kind"]);
            inc.kind = graph_field_inc_kind_from_name(inc_kind.c_str());
            if(inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
               inc.node_type = nb::str(inc_entry["node_type"]).c_str();
            } else if(inc.kind == GraphFieldInc::Kind::FIELD_OFFSET) {
               inc.field_key = nb::str(inc_entry["field_key"]).c_str();
            }
         }
         const auto mode = nb::str(entry["mode"]);
         const auto dtype = nb::str(entry["dtype"]);
         out.graph_tensors.emplace_back(
            GraphTensorSpec{
               .attr = py_string(entry["attr"]),
               .key = py_string(entry["key"]),
               .ptr_key = entry.contains("ptr_key") ? py_string(entry["ptr_key"]) : "",
               .mode = graph_field_mode_from_name(mode.c_str()),
               .dtype = graph_field_dtype_from_name(dtype.c_str()),
               .dim = entry.contains("dim") ? nb::cast< int >(entry["dim"]) : 1,
               .cat_dim = entry.contains("cat_dim")
                             ? normalize_graph_field_cat_dim(nb::cast< int >(entry["cat_dim"]))
                             : 0,
               .inc = std::move(inc),
            }
         );
      }
   }
   if(schema.contains("flags")) {
      if(schema.contains("flags")) {
         auto flags = nb::cast< nb::dict >(schema["flags"]);
         for(auto [key, value] : flags) {
            out.flags[py_string(key)] = nb::cast< bool >(value);
         }
      }
   }
   out.validate();
   return out;
}

}  // namespace mifrost
