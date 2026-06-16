#include "mifrost/batch_encoding_graph_field_mutation.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mifrost/common.hpp"
#include "mifrost/core/graph_fields.hpp"

namespace nb = nanobind;

namespace mifrost {

namespace {

template < typename T >
struct NumericFieldInput {
   std::vector< T > values;
   int ndim = 0;
   size_t rows = 0;
   size_t cols = 0;
};

template < typename T >
NumericFieldInput< T > coerce_numeric_values(nb::handle value)
{
   NumericFieldInput< T > out;
   const auto is_string_like = [](nb::handle handle) {
      return nb::isinstance< nb::str >(handle) or nb::isinstance< nb::bytes >(handle);
   };
   const auto try_scalar_from_zero_dim_arraylike = [&](nb::handle handle) -> std::optional< T > {
      if(not nb::hasattr(handle, "ndim")) {
         return std::nullopt;
      }
      int ndim = 0;
      try {
         ndim = nb::cast< int >(handle.attr("ndim"));
      } catch(...) {
         return std::nullopt;
      }
      if(ndim != 0) {
         return std::nullopt;
      }
      if(nb::hasattr(handle, "item")) {
         return nb::cast< T >(handle.attr("item")());
      }
      return nb::cast< T >(handle);
   };

   if(nb::isinstance< nb::bool_ >(value) or nb::isinstance< nb::int_ >(value)
      or nb::isinstance< nb::float_ >(value)) {
      out.values.push_back(nb::cast< T >(value));
      out.ndim = 0;
      out.rows = 1;
      out.cols = 1;
      return out;
   }
   if(const auto scalar = try_scalar_from_zero_dim_arraylike(value); scalar.has_value()) {
      out.values.push_back(*scalar);
      out.ndim = 0;
      out.rows = 1;
      out.cols = 1;
      return out;
   }

   if(not nb::isinstance< nb::iterable >(value) or is_string_like(value)) {
      throw std::invalid_argument("Graph field value must be a scalar or iterable");
   }

   bool has_nested = false;
   bool has_scalar = false;
   bool nested_cols_set = false;
   size_t nested_cols = 0;
   size_t nested_rows = 0;

   nb::object iterable_obj = nb::borrow< nb::object >(value);
   for(nb::handle item : iterable_obj) {
      if(const auto scalar = try_scalar_from_zero_dim_arraylike(item); scalar.has_value()) {
         has_scalar = true;
         out.values.push_back(*scalar);
         continue;
      }
      if(nb::isinstance< nb::iterable >(item) and not is_string_like(item)) {
         has_nested = true;
         size_t row_size = 0;
         nb::object nested_obj = nb::borrow< nb::object >(item);
         for(nb::handle nested : nested_obj) {
            out.values.push_back(nb::cast< T >(nested));
            row_size++;
         }
         if(not nested_cols_set) {
            nested_cols = row_size;
            nested_cols_set = true;
         } else if(row_size != nested_cols) {
            throw std::invalid_argument(
               "Graph field nested iterable rows must have consistent lengths"
            );
         }
         nested_rows++;
      } else {
         has_scalar = true;
         out.values.push_back(nb::cast< T >(item));
      }
   }
   if(has_nested and has_scalar) {
      throw std::invalid_argument(
         "Graph field value must be consistently 1D or 2D, not mixed nested/scalar"
      );
   }
   if(has_nested) {
      out.ndim = 2;
      out.rows = nested_rows;
      out.cols = nested_cols;
      return out;
   }

   out.ndim = 1;
   out.rows = out.values.size();
   out.cols = 1;
   return out;
}

template < typename T >
std::vector< T > normalize_graph_field_input(
   const std::string& key,
   const GraphFieldSpec& spec,
   NumericFieldInput< T > input
)
{
   const int cat_dim = normalize_graph_field_cat_dim(spec.cat_dim);
   const bool is_concat_mode = spec.mode == GraphFieldMode::CAT
                               or spec.mode == GraphFieldMode::RAGGED_CAT;
   if(is_concat_mode and spec.dim > 1) {
      if(cat_dim == 0) {
         if(input.ndim == 2 and input.cols != static_cast< size_t >(spec.dim)) {
            throw std::invalid_argument(
               "Graph field '" + key + "' with cat_dim=0 expects 2D shape [N, dim]"
            );
         }
      } else {
         if(input.ndim != 2) {
            throw std::invalid_argument(
               "Graph field '" + key
               + "' with cat_dim=1 and dim>1 requires a 2D value shaped [dim, N]"
            );
         }
         if(input.rows != static_cast< size_t >(spec.dim)) {
            throw std::invalid_argument(
               "Graph field '" + key + "' with cat_dim=1 expects leading dimension == dim"
            );
         }
      }
   }
   return std::move(input.values);
}

template < typename T >
void set_builder_field_values(
   BatchBuilder& builder,
   const std::string& key,
   const GraphFieldSpec& spec,
   nb::handle value
)
{
   auto input = coerce_numeric_values< T >(value);
   auto values = normalize_graph_field_input(key, spec, std::move(input));
   builder.set_field(key, std::span< const T >(values.data(), values.size()));
}

template < typename T >
void assign_batch_encoding_graph_field_values(
   BatchBuilder::BatchEncoding& encoding,
   const std::string& key,
   std::vector< T > values
)
{
   auto it = encoding.graph_fields.find(key);
   if(it == encoding.graph_fields.end()) {
      throw std::invalid_argument("Graph field '" + key + "' is not registered");
   }
   auto& field = it->second;
   if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
      throw std::invalid_argument(
         "Graph field '" + key + "' in RAGGED_CAT mode expects assignment as (values, ptr)"
      );
   }
   field.ptr.clear();
   field.values = NumericColumnData{std::move(values)};
   validate_graph_field_storage(key, field, encoding.num_graphs);
}

}  // namespace

bool is_native_graph_field_ptr_key(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view key
)
{
   constexpr std::string_view kPtrSuffix = "_ptr";
   if(key.size() <= kPtrSuffix.size()
      or key.compare(key.size() - kPtrSuffix.size(), kPtrSuffix.size(), kPtrSuffix) != 0) {
      return false;
   }
   std::string base(key.substr(0, key.size() - kPtrSuffix.size()));
   if(const auto it = encoding.graph_fields.find(base); it != encoding.graph_fields.end()) {
      return it->second.spec.mode == GraphFieldMode::RAGGED_CAT;
   }
   return false;
}

void set_batch_builder_graph_field(BatchBuilder& builder, const std::string& key, nb::handle value)
{
   const auto spec = builder.get_graph_field_spec(key);
   if(spec.dtype == GraphFieldDType::F32) {
      set_builder_field_values< float >(builder, key, spec, value);
      return;
   }
   set_builder_field_values< int64_t >(builder, key, spec, value);
}

void set_batch_builder_graph_fields(BatchBuilder& builder, const nb::dict& values)
{
   for(auto [key_obj, value_obj] : values) {
      set_batch_builder_graph_field(builder, py::to_std_string(key_obj), value_obj);
   }
}

void set_batch_encoding_graph_field(
   BatchBuilder::BatchEncoding& encoding,
   const std::string& key,
   nb::handle value
)
{
   if(encoding.graph_fields.find(key) == encoding.graph_fields.end()) {
      throw std::invalid_argument("Graph field '" + key + "' is not registered");
   }
   if(is_native_graph_field_ptr_key(encoding, key)) {
      throw std::invalid_argument(
         "Direct assignment to ragged ptr key '" + key
         + "' is not supported; assign the base field as (values, ptr)"
      );
   }
   auto& field = encoding.graph_fields.at(key);
   const auto spec = field.spec;

   if(spec.mode == GraphFieldMode::RAGGED_CAT) {
      if(not nb::isinstance< nb::tuple >(value)) {
         throw std::invalid_argument(
            "Graph field '" + key + "' in RAGGED_CAT mode expects assignment as (values, ptr)"
         );
      }
      const nb::tuple payload = nb::cast< nb::tuple >(value);
      if(nb::len(payload) != 2) {
         throw std::invalid_argument(
            "Graph field '" + key + "' in RAGGED_CAT mode expects exactly 2 elements: (values, ptr)"
         );
      }
      const nb::handle values_obj = payload[0];
      const nb::handle ptr_obj = payload[1];
      auto ptr_input = coerce_numeric_values< int64_t >(ptr_obj);
      if(ptr_input.ndim != 1) {
         throw std::invalid_argument(
            "Graph field '" + key + "' RAGGED_CAT ptr must be a 1D iterable of int64 values"
         );
      }
      field.ptr = std::move(ptr_input.values);

      if(spec.dtype == GraphFieldDType::F32) {
         auto input = coerce_numeric_values< float >(values_obj);
         auto values = normalize_graph_field_input(key, spec, std::move(input));
         field.values = NumericColumnData{std::move(values)};
      } else {
         auto input = coerce_numeric_values< int64_t >(values_obj);
         auto values = normalize_graph_field_input(key, spec, std::move(input));
         field.values = NumericColumnData{std::move(values)};
      }
      validate_graph_field_storage(key, field, encoding.num_graphs);
      return;
   }

   if(spec.dtype == GraphFieldDType::F32) {
      auto input = coerce_numeric_values< float >(value);
      auto values = normalize_graph_field_input(key, spec, std::move(input));
      assign_batch_encoding_graph_field_values(encoding, key, std::move(values));
      return;
   }
   auto input = coerce_numeric_values< int64_t >(value);
   auto values = normalize_graph_field_input(key, spec, std::move(input));
   assign_batch_encoding_graph_field_values(encoding, key, std::move(values));
}

void set_batch_encoding_graph_fields(BatchBuilder::BatchEncoding& encoding, const nb::dict& values)
{
   for(auto [key_obj, value_obj] : values) {
      set_batch_encoding_graph_field(
         encoding, py::to_std_string(key_obj), nb::borrow< nb::object >(value_obj)
      );
   }
}

}  // namespace mifrost
