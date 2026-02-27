#pragma once

#include <absl/container/btree_map.h>
#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"
#include "mifrost/core/batch_builder.hpp"

namespace mifrost {

enum class PythonFieldDType { PYOBJ, STR, F32, I64 };

struct PythonFieldSpec {
   PythonFieldDType dtype = PythonFieldDType::PYOBJ;
   GraphFieldMode mode = GraphFieldMode::STACK;
   int dim = 1;
   int cat_dim = 0;
   GraphFieldInc inc{};
   bool inferred = false;

   auto operator<=>(const PythonFieldSpec&) const noexcept = default;
};

using PythonFieldSpecMap = absl::btree_map< std::string, PythonFieldSpec >;

nanobind::dict batch_encoding_python_attrs(nanobind::handle self);

nanobind::dict batch_encoding_python_attrs_copy(nanobind::handle self);

bool is_reserved_python_attr_key(std::string_view key);

void batch_encoding_clear_python_attrs(nanobind::handle self);

nanobind::dict batch_encoding_field_specs(nanobind::handle self);

void batch_encoding_apply_python_attrs_from_state(
   nanobind::handle self,
   const nanobind::dict& state,
   nanobind::dict& dst
);

void batch_encoding_apply_python_attrs_from_state(
   nanobind::handle self,
   const nanobind::dict& state
);

PythonFieldSpecMap canonicalize_python_field_specs(const nanobind::dict& specs);

void merge_python_field_specs(PythonFieldSpecMap& dst, const PythonFieldSpecMap& src);

nanobind::dict python_field_specs_to_dict(const PythonFieldSpecMap& specs);

std::tuple< PythonFieldSpecMap, std::vector< nanobind::dict > > build_python_collation_inputs(
   const nb::sequence& source_objects,
   const nb::object& field_specs_obj
);

PythonFieldSpecMap filter_python_field_specs_for_native_collisions(
   const PythonFieldSpecMap& field_specs,
   const std::set< std::string >& reserved_native_keys
);

inline bool is_torch_tensor(nb::handle value)
{
   return nb::isinstance(value, py::torch_tensor_type());
}

void register_batch_encoding_field_specs(nanobind::handle self, const nanobind::dict& specs);

void copy_python_attrs_to_object(
   nanobind::handle src,
   nanobind::handle dst,
   std::optional< bool > as_batch,
   const BatchBuilder::BatchEncoding& encoding
);

nb::object torch_dtype_for_python_field_dtype(PythonFieldDType dtype);
bool try_get_python_attr(const nb::dict& attrs, nb::handle key_obj, nb::object& out);
nb::object
normalize_numeric_tensor(const std::string& key, const PythonFieldSpec& spec, nb::handle value);
bool python_objects_equal_for_const(const nb::object& lhs, const nb::object& rhs);
int64_t rows_for_tensor_piece(const PythonFieldSpec& spec, const nb::object& tensor);
void validate_string_value(const std::string& key, GraphFieldMode mode, nb::handle value);
nb::list
collate_python_stack_values(const std::vector< nb::dict >& source_attrs, nb::handle key_obj);
struct PythonRaggedCollation {
   nb::list values;
   nb::list ptr;
};

PythonRaggedCollation
collate_python_ragged_values(const std::vector< nb::dict >& source_attrs, nb::handle key_obj);
nb::object collate_python_const_value(
   const std::string& key,
   const std::vector< nb::dict >& source_attrs,
   nb::handle key_obj
);

////////////////////////////////////////////////////////////
/// Implementations ////////////////////////////////////////
////////////////////////////////////////////////////////////

using namespace nb::literals;

template < typename T >
concept BatchEncodingPointerRange = std::ranges::range< T >
                                    and std::same_as<
                                       std::ranges::range_value_t< T >,
                                       BatchBuilder::BatchEncoding* >;

template < typename BERange >
   requires BatchEncodingPointerRange< BERange >
nb::object collate_numeric_field(
   const std::string& key,
   const PythonFieldSpec& spec,
   const std::vector< nb::dict >& source_attrs,
   const BERange& source_encodings,
   nb::object& out_ptr
)
{
   nb::handle torch = py::torch_module();
   const nb::str key_obj(key.c_str());
   const int64_t cat_dim = graph_field_cat_dim_is_one(spec.cat_dim) ? 1 : 0;

   std::vector< int64_t > offsets(source_attrs.size(), 0);
   if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
      offsets.reserve(source_encodings.size());
      int64_t running = 0;
      for(const auto* encoding : source_encodings) {
         offsets.push_back(running);
         if(const auto it = encoding->node_counts.find(spec.inc.node_type);
            it != encoding->node_counts.end()) {
            running += std::max< int64_t >(0, it->second);
         }
      }
   }

   nb::list pieces;
   std::vector< int64_t > ptr{0};
   int64_t ptr_offset = 0;
   nb::object first_const = nb::none();
   bool have_const = false;

   for(size_t source_idx = 0; source_idx < source_attrs.size(); ++source_idx) {
      const auto& attrs = source_attrs[source_idx];
      nb::object value;
      if(not try_get_python_attr(attrs, key_obj, value)) {
         if(spec.mode == GraphFieldMode::STACK or spec.mode == GraphFieldMode::CONST) {
            throw std::invalid_argument(
               "Field '" + key + "' missing value for encoding index " + std::to_string(source_idx)
            );
         }
         if(spec.mode == GraphFieldMode::RAGGED_CAT) {
            ptr.push_back(ptr_offset);
         }
         continue;
      }

      nb::object tensor = normalize_numeric_tensor(key, spec, value);
      if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET and offsets[source_idx] != 0) {
         tensor = tensor.attr("__add__")(offsets[source_idx]);
      }

      if(spec.mode == GraphFieldMode::CONST) {
         if(not have_const) {
            first_const = tensor;
            have_const = true;
         } else if(not python_objects_equal_for_const(tensor, first_const)) {
            throw std::invalid_argument(
               "Field '" + key + "' CONST has non-constant values across encodings"
            );
         }
         continue;
      }

      if(spec.mode == GraphFieldMode::STACK) {
         if(spec.dim == 1) {
            pieces.append(tensor.attr("reshape")(nb::make_tuple(1)));
         } else {
            pieces.append(tensor.attr("reshape")(nb::make_tuple(1, spec.dim)));
         }
         continue;
      }

      pieces.append(tensor);
      if(spec.mode == GraphFieldMode::RAGGED_CAT) {
         ptr_offset += rows_for_tensor_piece(spec, tensor);
         ptr.push_back(ptr_offset);
      }
   }

   if(spec.mode == GraphFieldMode::CONST) {
      if(not have_const) {
         throw std::invalid_argument("Field '" + key + "' CONST requires at least one value");
      }
      out_ptr = nb::none();
      return first_const;
   }

   if(nb::len(pieces) == 0) {
      nb::object empty;
      if(spec.dim == 1) {
         empty = torch.attr("empty")(
            nb::make_tuple(0), "dtype"_a = torch_dtype_for_python_field_dtype(spec.dtype)
         );
      } else if(cat_dim == 0 or spec.mode == GraphFieldMode::STACK) {
         empty = torch.attr("empty")(
            nb::make_tuple(0, spec.dim), "dtype"_a = torch_dtype_for_python_field_dtype(spec.dtype)
         );
      } else {
         empty = torch.attr("empty")(
            nb::make_tuple(spec.dim, 0), "dtype"_a = torch_dtype_for_python_field_dtype(spec.dtype)
         );
      }
      if(spec.mode == GraphFieldMode::RAGGED_CAT) {
         out_ptr = torch.attr("as_tensor")(ptr, "dtype"_a = torch.attr("int64"));
      } else {
         out_ptr = nb::none();
      }
      return empty;
   }

   int64_t dim = cat_dim;
   if(spec.mode == GraphFieldMode::STACK) {
      dim = 0;
   }
   nb::object out = torch.attr("cat")(pieces, "dim"_a = dim);
   if(spec.mode == GraphFieldMode::RAGGED_CAT) {
      out_ptr = torch.attr("as_tensor")(ptr, "dtype"_a = torch.attr("int64"));
   } else {
      out_ptr = nb::none();
   }
   return out;
}

template < typename BERange >
   requires BatchEncodingPointerRange< BERange >
nb::dict apply_python_collation(
   const PythonFieldSpecMap& field_specs,
   const std::vector< nb::dict >& source_attrs,
   const BERange& source_encodings
)
{
   nb::dict out_attrs;
   for(const auto& [key, spec] : field_specs) {
      const nb::str key_obj(key.c_str());
      if(spec.dtype == PythonFieldDType::F32 or spec.dtype == PythonFieldDType::I64) {
         nb::object ptr = nb::none();
         nb::object values = collate_numeric_field(key, spec, source_attrs, source_encodings, ptr);
         out_attrs[key_obj] = std::move(values);
         if(spec.mode == GraphFieldMode::RAGGED_CAT) {
            out_attrs[(key + "_ptr").c_str()] = std::move(ptr);
         }
         continue;
      }

      if(spec.dtype == PythonFieldDType::PYOBJ and spec.mode == GraphFieldMode::STACK) {
         out_attrs[key_obj] = collate_python_stack_values(source_attrs, key_obj);
         continue;
      }

      if((spec.dtype == PythonFieldDType::PYOBJ or spec.dtype == PythonFieldDType::STR)
         and spec.mode == GraphFieldMode::RAGGED_CAT) {
         if(spec.dtype == PythonFieldDType::STR) {
            for(size_t source_idx = 0; source_idx < source_attrs.size(); ++source_idx) {
               nb::object value;
               if(try_get_python_attr(source_attrs[source_idx], key_obj, value)) {
                  validate_string_value(key, GraphFieldMode::RAGGED_CAT, value);
               }
            }
         }
         auto ragged = collate_python_ragged_values(source_attrs, key_obj);
         out_attrs[key_obj] = std::move(ragged.values);
         const std::string key_with_ptr = key + "_ptr";
         out_attrs[key_with_ptr.c_str()] = std::move(ragged.ptr);
         continue;
      }

      if(spec.dtype == PythonFieldDType::STR and spec.mode == GraphFieldMode::STACK) {
         nb::list values;
         for(const auto& attrs : source_attrs) {
            nb::object value;
            if(try_get_python_attr(attrs, key_obj, value)) {
               validate_string_value(key, GraphFieldMode::STACK, value);
               values.append(value);
            } else {
               values.append(nb::none());
            }
         }
         out_attrs[key_obj] = std::move(values);
         continue;
      }

      if(spec.dtype == PythonFieldDType::STR and spec.mode == GraphFieldMode::CONST) {
         for(const auto& attrs : source_attrs) {
            nb::object value;
            if(try_get_python_attr(attrs, key_obj, value)) {
               validate_string_value(key, GraphFieldMode::CONST, value);
            }
         }
         out_attrs[key_obj] = collate_python_const_value(key, source_attrs, key_obj);
         continue;
      }

      out_attrs[key_obj] = collate_python_const_value(key, source_attrs, key_obj);
   }
   return out_attrs;
}

}  // namespace mifrost
