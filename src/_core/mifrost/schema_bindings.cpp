/**
 * @file schema_bindings.cpp
 * @brief Implements Schema::to_dict and Schema::from_dict (Python API methods).
 *
 * This file is compiled only into mifrost_core_module (the nanobind extension),
 * never into libmifrost_core. That guarantees libmifrost_core.so carries no
 * nanobind symbols and the undefined-symbol runtime failure cannot occur.
 * MIFROST_ENABLE_PYTHON_API is provided by the mifrost_core_module compile defs.
 */

#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "mifrost/core/schema.hpp"

namespace mifrost {

namespace {

std::string py_string(nb::handle value)
{
   return {nb::str(value).c_str()};
}

}  // namespace

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
      auto flags = nb::cast< nb::dict >(schema["flags"]);
      for(auto [key, value] : flags) {
         out.flags[py_string(key)] = nb::cast< bool >(value);
      }
   }
   out.validate();
   return out;
}

}  // namespace mifrost
