#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mifrost {

using NumericColumnData = std::variant< std::vector< float >, std::vector< int64_t > >;

enum class GraphFieldDType { F32, I64 };
enum class GraphFieldMode { STACK, CAT, RAGGED_CAT, CONST };

struct GraphFieldInc {
   enum class Kind { NONE, NODE_OFFSET };
   Kind kind = Kind::NONE;
   std::string node_type;

   auto operator<=>(const GraphFieldInc&) const noexcept = default;
};

struct GraphFieldSpec {
   GraphFieldDType dtype = GraphFieldDType::F32;
   GraphFieldMode mode = GraphFieldMode::STACK;
   int dim = 1;
   int cat_dim = 0;
   GraphFieldInc inc{};

   auto operator<=>(const GraphFieldSpec&) const noexcept = default;
};

struct GraphField {
   GraphFieldSpec spec;
   NumericColumnData values;
   std::vector< int64_t > ptr;
   std::optional< NumericColumnData > pending;
};

inline NumericColumnData make_numeric_column_data(GraphFieldDType dtype)
{
   if(dtype == GraphFieldDType::F32) {
      return std::vector< float >{};
   }
   return std::vector< int64_t >{};
}

inline const char* graph_field_dtype_name(GraphFieldDType dtype)
{
   switch(dtype) {
      case GraphFieldDType::F32: return "f32";
      case GraphFieldDType::I64: return "i64";
   }
   throw std::logic_error("Unknown GraphFieldDType");
}

inline const char* graph_field_mode_name(GraphFieldMode mode)
{
   switch(mode) {
      case GraphFieldMode::STACK: return "stack";
      case GraphFieldMode::CAT: return "cat";
      case GraphFieldMode::RAGGED_CAT: return "ragged_cat";
      case GraphFieldMode::CONST: return "const";
   }
   throw std::logic_error("Unknown GraphFieldMode");
}

inline const char* graph_field_inc_kind_name(GraphFieldInc::Kind kind)
{
   switch(kind) {
      case GraphFieldInc::Kind::NONE: return "none";
      case GraphFieldInc::Kind::NODE_OFFSET: return "node_offset";
   }
   throw std::logic_error("Unknown GraphFieldInc::Kind");
}

inline char ascii_lower(char c)
{
   if(c >= 'A' && c <= 'Z') {
      return static_cast< char >(c - 'A' + 'a');
   }
   return c;
}

inline bool ascii_iequals(const std::string_view lhs, const std::string_view rhs)
{
   if(lhs.size() != rhs.size()) {
      return false;
   }
   for(size_t i = 0; i < lhs.size(); ++i) {
      if(ascii_lower(lhs[i]) != ascii_lower(rhs[i])) {
         return false;
      }
   }
   return true;
}

inline GraphFieldDType graph_field_dtype_from_name(const std::string_view value)
{
   if(ascii_iequals(value, "f32")) {
      return GraphFieldDType::F32;
   }
   if(ascii_iequals(value, "i64")) {
      return GraphFieldDType::I64;
   }
   throw std::invalid_argument("Unknown graph field dtype: " + std::string(value));
}

inline GraphFieldMode graph_field_mode_from_name(const std::string_view value)
{
   if(ascii_iequals(value, "stack")) {
      return GraphFieldMode::STACK;
   }
   if(ascii_iequals(value, "cat")) {
      return GraphFieldMode::CAT;
   }
   if(ascii_iequals(value, "ragged_cat")) {
      return GraphFieldMode::RAGGED_CAT;
   }
   if(ascii_iequals(value, "const")) {
      return GraphFieldMode::CONST;
   }
   throw std::invalid_argument("Unknown graph field mode: " + std::string(value));
}

inline GraphFieldInc::Kind graph_field_inc_kind_from_name(const std::string_view value)
{
   if(ascii_iequals(value, "none")) {
      return GraphFieldInc::Kind::NONE;
   }
   if(ascii_iequals(value, "node_offset")) {
      return GraphFieldInc::Kind::NODE_OFFSET;
   }
   throw std::invalid_argument("Unknown graph field inc kind: " + std::string(value));
}

inline int normalize_graph_field_cat_dim(int cat_dim)
{
   if(cat_dim == -1) {
      return 1;
   }
   return cat_dim;
}

inline bool graph_field_cat_dim_is_one(int cat_dim)
{
   const int normalized = normalize_graph_field_cat_dim(cat_dim);
   return normalized == 1;
}

inline void validate_graph_field_spec(const std::string_view key, const GraphFieldSpec& spec)
{
   const std::string key_str(key);
   if(spec.dim <= 0) {
      throw std::invalid_argument("Graph field '" + key_str + "' requires dim > 0");
   }
   const int cat_dim = normalize_graph_field_cat_dim(spec.cat_dim);
   if(spec.mode == GraphFieldMode::CAT || spec.mode == GraphFieldMode::RAGGED_CAT) {
      if(cat_dim != 0 && cat_dim != 1) {
         throw std::invalid_argument(
            "Graph field '" + key_str + "' CAT/RAGGED_CAT requires cat_dim in {0, 1, -1}"
         );
      }
   } else if(cat_dim != 0) {
      throw std::invalid_argument(
         "Graph field '" + key_str + "' non-concat modes require cat_dim == 0"
      );
   }
   if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET && spec.dtype != GraphFieldDType::I64) {
      throw std::invalid_argument(
         "Graph field '" + key_str + "' NODE_OFFSET increment requires dtype=i64"
      );
   }
   if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET && spec.inc.node_type.empty()) {
      throw std::invalid_argument(
         "Graph field '" + key_str + "' NODE_OFFSET increment requires non-empty node_type"
      );
   }
}

}  // namespace mifrost
