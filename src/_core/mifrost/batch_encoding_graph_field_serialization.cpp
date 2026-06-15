#include "mifrost/batch_encoding_graph_field_serialization.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "mifrost/common.hpp"

namespace nb = nanobind;

namespace mifrost {

GraphFieldSpec graph_field_spec_from_dict(const nb::dict& spec_dict)
{
   GraphFieldSpec spec;
   if(not spec_dict.contains("dtype")) {
      throw std::invalid_argument("field spec requires 'dtype'");
   }
   if(not spec_dict.contains("mode")) {
      throw std::invalid_argument("field spec requires 'mode'");
   }
   const auto dtype = py::to_std_string(spec_dict["dtype"]);
   const auto mode = py::to_std_string(spec_dict["mode"]);
   spec.dtype = graph_field_dtype_from_name(dtype);
   spec.mode = graph_field_mode_from_name(mode);
   if(spec_dict.contains("dim")) {
      spec.dim = nb::cast< int >(spec_dict["dim"]);
   }
   if(spec_dict.contains("cat_dim")) {
      spec.cat_dim = normalize_graph_field_cat_dim(nb::cast< int >(spec_dict["cat_dim"]));
   }
   if(spec_dict.contains("inc") and not spec_dict["inc"].is_none()) {
      if(not nb::isinstance< nb::dict >(spec_dict["inc"])) {
         throw std::invalid_argument("field spec 'inc' must be a dict");
      }
      const auto inc_dict = nb::cast< nb::dict >(spec_dict["inc"]);
      if(inc_dict.contains("kind")) {
         const auto kind = py::to_std_string(inc_dict["kind"]);
         spec.inc.kind = graph_field_inc_kind_from_name(kind);
      }
      if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
         if(not inc_dict.contains("node_type")) {
            throw std::invalid_argument("field spec inc NODE_OFFSET requires node_type");
         }
         spec.inc.node_type = py::to_std_string(inc_dict["node_type"]);
      } else if(spec.inc.kind == GraphFieldInc::Kind::FIELD_OFFSET) {
         if(not inc_dict.contains("field_key")) {
            throw std::invalid_argument("field spec inc FIELD_OFFSET requires field_key");
         }
         spec.inc.field_key = py::to_std_string(inc_dict["field_key"]);
      }
   }
   return spec;
}

nb::dict graph_field_spec_to_dict(const GraphFieldSpec& spec)
{
   nb::dict out;
   out["dtype"] = graph_field_dtype_name(spec.dtype);
   out["mode"] = graph_field_mode_name(spec.mode);
   out["dim"] = spec.dim;
   out["cat_dim"] = spec.cat_dim;
   nb::dict inc;
   inc["kind"] = graph_field_inc_kind_name(spec.inc.kind);
   if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
      inc["node_type"] = spec.inc.node_type;
   } else if(spec.inc.kind == GraphFieldInc::Kind::FIELD_OFFSET) {
      inc["field_key"] = spec.inc.field_key;
   }
   out["inc"] = std::move(inc);
   return out;
}

nb::dict graph_field_map_to_dict(const hash_map< std::string, GraphField >& fields)
{
   nb::dict out;
   for(const auto& [key, field] : fields) {
      nb::dict entry;
      entry["spec"] = graph_field_spec_to_dict(field.spec);
      std::visit(
         [&](const auto& data) {
            using T = std::decay_t< decltype(data) >::value_type;
            if constexpr(std::is_same_v< T, float >) {
               entry["dtype"] = "f32";
            } else {
               entry["dtype"] = "i64";
            }
            entry["length"] = static_cast< int64_t >(data.size());
            const auto* ptr = reinterpret_cast< const char* >(data.data());
            entry["raw"] = nb::bytes(ptr, data.size() * sizeof(T));
         },
         field.values
      );
      entry["ptr"] = nb::cast(field.ptr);
      out[key.c_str()] = std::move(entry);
   }
   return out;
}

hash_map< std::string, GraphField > graph_field_map_from_dict(const nb::dict& payload)
{
   hash_map< std::string, GraphField > out;
   out.reserve(payload.size());
   for(auto [key_obj, field_obj] : payload) {
      const auto key = py::to_std_string(key_obj);
      const auto entry = nb::cast< nb::dict >(field_obj);
      GraphField field;
      field.spec = graph_field_spec_from_dict(nb::cast< nb::dict >(entry["spec"]));
      field.ptr = nb::cast< std::vector< int64_t > >(entry["ptr"]);

      const auto dtype = py::to_std_string(entry["dtype"]);
      const auto length = static_cast< size_t >(nb::cast< int64_t >(entry["length"]));
      const auto raw_bytes = nb::cast< nb::bytes >(entry["raw"]);
      const std::string_view raw(raw_bytes.c_str(), raw_bytes.size());
      if(dtype == "f32") {
         if(raw.size() != length * sizeof(float)) {
            throw std::invalid_argument("Malformed graph field f32 payload");
         }
         std::vector< float > data(length);
         if(length > 0) {
            std::memcpy(data.data(), raw.data(), raw.size());
         }
         field.values = std::move(data);
      } else if(dtype == "i64") {
         if(raw.size() != length * sizeof(int64_t)) {
            throw std::invalid_argument("Malformed graph field i64 payload");
         }
         std::vector< int64_t > data(length);
         if(length > 0) {
            std::memcpy(data.data(), raw.data(), raw.size());
         }
         field.values = std::move(data);
      } else {
         throw std::invalid_argument("Unsupported graph field dtype payload");
      }
      out[key] = std::move(field);
   }
   return out;
}

}  // namespace mifrost
