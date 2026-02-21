#include <absl/container/btree_map.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/axiom_evaluators/interface.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <optional>
#include <ranges>
#include <string_view>
#include <type_traits>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_python_collation.hpp"
#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/default_relations.hpp"
#include "mifrost/core/dlpack_utils.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/horizon_hgraph_encoder.hpp"
#include "mifrost/core/map_view.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/core/nb_instance.hpp"
#include "mifrost/core/successor_hgraph_encoder.hpp"
#include "mifrost/core/transition_dag.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

struct ReprQuoted {
   std::string_view value;
};

struct ReprEdgeType {
   const EdgeType* value = nullptr;
};

struct DisplayEdgeType {
   const EdgeType* value = nullptr;
};

}  // namespace mifrost

template <>
struct fmt::formatter< mifrost::ReprQuoted > {
   constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

   template < typename FormatContext >
   auto format(const mifrost::ReprQuoted& quoted, FormatContext& ctx) const
   {
      auto out = ctx.out();
      *out++ = '\'';
      for(char ch : quoted.value) {
         if(ch == '\'' or ch == '\\') {
            *out++ = '\\';
         }
         *out++ = ch;
      }
      *out++ = '\'';
      return out;
   }
};

template <>
struct fmt::formatter< mifrost::ReprEdgeType > {
   constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

   template < typename FormatContext >
   auto format(const mifrost::ReprEdgeType& edge_type, FormatContext& ctx) const
   {
      if(edge_type.value == nullptr) {
         return fmt::format_to(ctx.out(), "(None)");
      }
      return fmt::format_to(
         ctx.out(),
         "({}, {}, {})",
         mifrost::ReprQuoted{edge_type.value->src},
         mifrost::ReprQuoted{edge_type.value->rel},
         mifrost::ReprQuoted{edge_type.value->dst}
      );
   }
};

template <>
struct fmt::formatter< mifrost::DisplayEdgeType > {
   constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

   template < typename FormatContext >
   auto format(const mifrost::DisplayEdgeType& edge_type, FormatContext& ctx) const
   {
      if(edge_type.value == nullptr) {
         return fmt::format_to(ctx.out(), "(None)");
      }
      return fmt::format_to(
         ctx.out(), "({}, {}, {})", edge_type.value->src, edge_type.value->rel, edge_type.value->dst
      );
   }
};

namespace mifrost {

namespace {

std::string py_string(nb::handle value)
{
   return {nb::str(value).c_str()};
}

constexpr std::string_view kPythonTensorDeviceAttr = "__mifrost_tensor_device__";
constexpr std::string_view kPythonTensorCacheAttr = "__mifrost_tensor_cache__";

void clear_owner_tensor_cache(nb::handle owner);

nb::handle torch_module_handle()
{
   static nb::object* module = []() { return new nb::object(nb::module_::import_("torch")); }();
   return *module;
}

nb::handle builtins_module_handle()
{
   static nb::object* module = []() { return new nb::object(nb::module_::import_("builtins")); }();
   return *module;
}

nb::handle builtins_object_setattr_handle()
{
   static nb::object* fn = []() {
      return new nb::object(builtins_module_handle().attr("object").attr("__setattr__"));
   }();
   return *fn;
}

nb::handle builtins_open_handle()
{
   static nb::object* fn = []() { return new nb::object(builtins_module_handle().attr("open")); }();
   return *fn;
}

nb::handle builtins_tuple_ctor_handle()
{
   static nb::object* fn = []() {
      return new nb::object(builtins_module_handle().attr("tuple"));
   }();
   return *fn;
}

nb::handle pickle_module_handle()
{
   static nb::object* module = []() { return new nb::object(nb::module_::import_("pickle")); }();
   return *module;
}

nb::handle pickle_dumps_handle()
{
   static nb::object* fn = []() { return new nb::object(pickle_module_handle().attr("dumps")); }();
   return *fn;
}

nb::handle pickle_loads_handle()
{
   static nb::object* fn = []() { return new nb::object(pickle_module_handle().attr("loads")); }();
   return *fn;
}

nb::handle types_module_handle()
{
   static nb::object* module = []() { return new nb::object(nb::module_::import_("types")); }();
   return *module;
}

nb::handle mapping_proxy_type_ctor_handle()
{
   static nb::object* ctor = []() {
      return new nb::object(types_module_handle().attr("MappingProxyType"));
   }();
   return *ctor;
}

nb::handle torch_geometric_data_module_handle()
{
   static nb::object* module = []() {
      return new nb::object(nb::module_::import_("torch_geometric.data"));
   }();
   return *module;
}

nb::handle torch_geometric_heterodata_ctor_handle()
{
   static nb::object* ctor = []() {
      return new nb::object(torch_geometric_data_module_handle().attr("HeteroData"));
   }();
   return *ctor;
}

nb::handle torch_geometric_data_ctor_handle()
{
   static nb::object* ctor = []() {
      return new nb::object(torch_geometric_data_module_handle().attr("Data"));
   }();
   return *ctor;
}

nb::handle mifrost_core_batch_encoding_cls_handle()
{
   static nb::object* cls = []() {
      return new nb::object(nb::module_::import_("mifrost._core").attr("BatchEncoding"));
   }();
   return *cls;
}

nb::handle mifrost_batch_encoding_loader_handle()
{
   static nb::object* loader = []() {
      return new nb::object(nb::module_::import_("mifrost").attr("_batch_encoding_from_payload"));
   }();
   return *loader;
}

GraphFieldSpec graph_field_spec_from_dict(const nb::dict& spec_dict)
{
   GraphFieldSpec spec;
   if(not spec_dict.contains("dtype")) {
      throw std::invalid_argument("field spec requires 'dtype'");
   }
   if(not spec_dict.contains("mode")) {
      throw std::invalid_argument("field spec requires 'mode'");
   }
   const auto dtype = py_string(spec_dict["dtype"]);
   const auto mode = py_string(spec_dict["mode"]);
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
         const auto kind = py_string(inc_dict["kind"]);
         spec.inc.kind = graph_field_inc_kind_from_name(kind);
      }
      if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
         if(not inc_dict.contains("node_type")) {
            throw std::invalid_argument("field spec inc NODE_OFFSET requires node_type");
         }
         spec.inc.node_type = py_string(inc_dict["node_type"]);
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
      const auto key = py_string(key_obj);
      const auto entry = nb::cast< nb::dict >(field_obj);
      GraphField field;
      field.spec = graph_field_spec_from_dict(nb::cast< nb::dict >(entry["spec"]));
      field.ptr = nb::cast< std::vector< int64_t > >(entry["ptr"]);

      const auto dtype = py_string(entry["dtype"]);
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
         encoding, py_string(key_obj), nb::borrow< nb::object >(value_obj)
      );
   }
}

void set_python_attribute(nb::handle self, const std::string& key, nb::handle value)
{
   builtins_object_setattr_handle()(self, key, value);
}

template < typename T >
nb::object vector_to_1d_tensor_view(std::vector< T >& vec, nb::handle owner)
{
   return dlpack_utils::vector_to_dlpack_view_1d(vec, owner);
}

template < typename T >
nb::object
vector_to_2d_tensor_view(std::vector< T >& vec, size_t rows, size_t cols, nb::handle owner)
{
   return dlpack_utils::vector_to_dlpack_view_2d(vec, rows, cols, owner);
}

template < typename T >
nb::object vector_to_1d_tensor_owned(std::vector< T >&& vec)
{
   return dlpack_utils::vector_to_dlpack_owned_1d(std::move(vec));
}

template < typename T >
nb::object vector_to_2d_tensor_owned(std::vector< T >&& vec, size_t rows, size_t cols)
{
   return dlpack_utils::vector_to_dlpack_owned_2d(std::move(vec), rows, cols);
}

std::vector< int64_t > ptr_to_batch(const std::vector< int64_t >& ptr)
{
   std::vector< int64_t > batch;
   if(ptr.size() < 2) {
      return batch;
   }
   batch.reserve(static_cast< size_t >(std::max< int64_t >(0, ptr.back())));
   for(size_t idx = 0; idx + 1 < ptr.size(); ++idx) {
      const int64_t count = std::max< int64_t >(0, ptr[idx + 1] - ptr[idx]);
      batch.insert(batch.end(), static_cast< size_t >(count), static_cast< int64_t >(idx));
   }
   return batch;
}

nb::object flatten_single_graph_metadata_list(nb::handle value)
{
   if(not nb::isinstance< nb::list >(value)) {
      return nb::borrow< nb::object >(value);
   }
   nb::list outer = nb::borrow< nb::list >(value);
   if(nb::len(outer) == 1 and nb::isinstance< nb::list >(outer[0])) {
      return nb::borrow< nb::object >(outer[0]);
   }
   return nb::borrow< nb::object >(value);
}

void copy_store_attrs_without_batch(nb::object& dst_store, nb::object& src_store)
{
   for(auto key_obj : src_store.attr("keys")()) {
      const std::string key = py_string(key_obj);
      if(key == "ptr" or key == "batch") {
         continue;
      }
      dst_store.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
   }
}

void copy_global_attrs_for_single(nb::object& dst, nb::object& src)
{
   nb::object global_store = src.attr("_global_store");
   for(auto key_obj : global_store.attr("keys")()) {
      const std::string key = py_string(key_obj);
      if(key == "_num_graphs") {
         continue;
      }
      nb::object value = global_store.attr("__getitem__")(key_obj);
      if(key == "object_names") {
         value = flatten_single_graph_metadata_list(value);
      }
      dst.attr(key.c_str()) = value;
   }
}

nb::object batch_to_single_hetero_data(nb::object& pyg_batch)
{
   nb::object out = torch_geometric_heterodata_ctor_handle()();

   for(auto node_type_obj : pyg_batch.attr("node_types")) {
      std::string node_type = py_string(node_type_obj);
      nb::object src_store = pyg_batch.attr("__getitem__")(node_type);
      nb::object dst_store = out.attr("__getitem__")(node_type);
      copy_store_attrs_without_batch(dst_store, src_store);
      if(nb::cast< bool >(src_store.attr("__contains__")("node_names"))) {
         nb::object flat_names = flatten_single_graph_metadata_list(
            src_store.attr("__getitem__")("node_names")
         );
         dst_store.attr("node_names") = flat_names;
         dst_store.attr("num_nodes") = nb::len(flat_names);
      }
   }

   for(auto edge_type_obj : pyg_batch.attr("edge_types")) {
      nb::object src_store = pyg_batch.attr("__getitem__")(edge_type_obj);
      nb::object dst_store = out.attr("__getitem__")(edge_type_obj);
      for(auto key_obj : src_store.attr("keys")()) {
         dst_store.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
   }

   copy_global_attrs_for_single(out, pyg_batch);
   return out;
}

nb::object batch_to_single_homo_data(nb::object& pyg_batch)
{
   nb::object out = torch_geometric_data_ctor_handle()();

   nb::list node_types = nb::cast< nb::list >(pyg_batch.attr("node_types"));
   if(nb::len(node_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=False) for homo expects a single node type"
      );
   }

   if(nb::len(node_types) == 1) {
      std::string node_type = py_string(node_types[0]);
      nb::object src_store = pyg_batch.attr("__getitem__")(node_type);
      for(auto key_obj : src_store.attr("keys")()) {
         const std::string key = py_string(key_obj);
         if(key == "ptr" or key == "batch") {
            continue;
         }
         out.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
      if(nb::cast< bool >(src_store.attr("__contains__")("node_names"))) {
         nb::object flat_names = flatten_single_graph_metadata_list(
            src_store.attr("__getitem__")("node_names")
         );
         out.attr("node_names") = flat_names;
      }
   }

   nb::list edge_types = nb::cast< nb::list >(pyg_batch.attr("edge_types"));
   if(nb::len(edge_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=False) for homo expects a single edge type"
      );
   }
   if(nb::len(edge_types) == 1) {
      nb::object src_store = pyg_batch.attr("__getitem__")(edge_types[0]);
      for(auto key_obj : src_store.attr("keys")()) {
         out.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
   }

   copy_global_attrs_for_single(out, pyg_batch);
   return out;
}

nb::dict
batch_encoding_to_state_dict(const BatchBuilder::BatchEncoding& encoding, bool include_metadata)
{
   auto map_to_dict =
      []< typename value_t >(const absl::btree_map< std::string, value_t >& values) {
         nb::dict out;
         for(const auto& [key, value] : values) {
            out[key.c_str()] = value;
         }
         return out;
      };

   nb::dict state;
   state["format_version"] = 1;
   state["graph_kind"] = encoding.graph_kind;
   state["num_graphs"] = encoding.num_graphs;
   state["schema_flags"] = map_to_dict(encoding.schema_flags);
   state["node_feature_dims"] = encoding.node_feature_dims;
   state["graph_attrs"] = encoding.graph_attrs;
   state["graph_fields"] = graph_field_map_to_dict(encoding.graph_fields);
   state["ptrs"] = encoding.ptrs;
   state["node_counts"] = map_to_dict(encoding.node_counts);
   state["schema"] = encoding.schema.to_dict();
   if(include_metadata) {
      state["node_names"] = encoding.node_names;
      state["object_names"] = encoding.object_names;
   } else {
      state["node_names"] = hash_map< std::string, std::vector< std::string > >{};
      state["object_names"] = std::vector< std::string >{};
   }

   nb::dict columns;
   for(const auto& [key, column] : encoding.columns) {
      nb::dict c;
      c["dim"] = column.dim;
      std::visit(
         [&]< typename T >(const std::vector< T >& data) {
            if constexpr(std::is_same_v< T, float >) {
               c["dtype"] = "f32";
            } else {
               c["dtype"] = "i64";
            }
            c["length"] = static_cast< int64_t >(data.size());
            const auto* ptr = reinterpret_cast< const char* >(data.data());
            c["raw"] = nb::bytes(ptr, data.size() * sizeof(T));
         },
         column.data
      );
      columns[key.c_str()] = std::move(c);
   }
   state["columns"] = std::move(columns);
   return state;
}

template < typename value_type >
auto map_from_dict(const nb::dict& source)
{
   absl::btree_map< std::string, value_type > out;
   for(auto [key_obj, value_obj] : source) {
      out.emplace(py_string(key_obj), nb::cast< value_type >(value_obj));
   }
   return out;
};

BatchBuilder::BatchEncoding batch_encoding_from_state_dict(const nb::dict& state)
{
   const int version = nb::cast< int >(state["format_version"]);
   if(version != 1) {
      throw std::invalid_argument("Unsupported BatchEncoding format version");
   }

   BatchBuilder::BatchEncoding encoding;
   try {
      encoding.graph_kind = py_string(state["graph_kind"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['graph_kind']: " + std::string(ex.what()));
   }
   try {
      encoding.num_graphs = nb::cast< int64_t >(state["num_graphs"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['num_graphs']: " + std::string(ex.what()));
   }
   try {
      nb::dict schema_flags = nb::cast< nb::dict >(state["schema_flags"]);
      encoding.schema_flags = map_from_dict< bool >(schema_flags);
   } catch(const std::exception& ex) {
      throw std::invalid_argument(
         "Failed to parse state['schema_flags']: " + std::string(ex.what())
      );
   }
   {
      nb::dict node_feature_dims;
      try {
         node_feature_dims = nb::cast< nb::dict >(state["node_feature_dims"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_feature_dims']: " + std::string(ex.what())
         );
      }
      encoding.node_feature_dims.clear();
      encoding.node_feature_dims.reserve(node_feature_dims.size());
      for(auto [key_obj, value_obj] : node_feature_dims) {
         try {
            encoding.node_feature_dims.emplace(py_string(key_obj), nb::cast< int >(value_obj));
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['node_feature_dims'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   {
      nb::dict graph_attrs;
      try {
         graph_attrs = nb::cast< nb::dict >(state["graph_attrs"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['graph_attrs']: " + std::string(ex.what())
         );
      }
      encoding.graph_attrs.clear();
      encoding.graph_attrs.reserve(graph_attrs.size());
      for(auto [key_obj, value_obj] : graph_attrs) {
         try {
            encoding.graph_attrs.emplace(
               py_string(key_obj), nb::cast< BatchBuilder::GraphAttrValue >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['graph_attrs'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   if(state.contains("graph_fields")) {
      try {
         encoding.graph_fields = graph_field_map_from_dict(
            nb::cast< nb::dict >(state["graph_fields"])
         );
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['graph_fields']: " + std::string(ex.what())
         );
      }
   }
   {
      nb::dict ptrs;
      try {
         ptrs = nb::cast< nb::dict >(state["ptrs"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument("Failed to parse state['ptrs']: " + std::string(ex.what()));
      }
      encoding.ptrs.clear();
      encoding.ptrs.reserve(ptrs.size());
      for(auto [key_obj, value_obj] : ptrs) {
         try {
            encoding.ptrs.emplace(
               py_string(key_obj), nb::cast< std::vector< int64_t > >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['ptrs'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   {
      try {
         nb::dict node_counts = nb::cast< nb::dict >(state["node_counts"]);
         encoding.node_counts = map_from_dict< int64_t >(node_counts);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_counts']: " + std::string(ex.what())
         );
      }
   }
   {
      try {
         nb::dict schema = nb::cast< nb::dict >(state["schema"]);
         encoding.schema = Schema::from_dict(schema);
      } catch(const std::exception& ex) {
         throw std::invalid_argument("Failed to parse state['schema']: " + std::string(ex.what()));
      }
   }
   {
      nb::dict node_names;
      try {
         node_names = nb::cast< nb::dict >(state["node_names"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_names']: " + std::string(ex.what())
         );
      }
      encoding.node_names.clear();
      encoding.node_names.reserve(node_names.size());
      for(auto [key_obj, value_obj] : node_names) {
         try {
            encoding.node_names.emplace(
               py_string(key_obj), nb::cast< std::vector< std::string > >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['node_names'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   try {
      encoding.object_names = nb::cast< std::vector< std::string > >(state["object_names"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument(
         "Failed to parse state['object_names']: " + std::string(ex.what())
      );
   }

   nb::dict columns;
   try {
      columns = nb::cast< nb::dict >(state["columns"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['columns']: " + std::string(ex.what()));
   }
   for(auto [key_obj, col_obj] : columns) {
      auto col = nb::cast< nb::dict >(col_obj);
      const auto key = py_string(key_obj);
      const auto dim = nb::cast< int >(col["dim"]);
      const auto dtype = py_string(col["dtype"]);
      const auto length = nb::cast< int64_t >(col["length"]);
      const auto raw_bytes = nb::cast< nb::bytes >(col["raw"]);
      const std::string_view raw(raw_bytes.c_str(), raw_bytes.size());

      if(dtype == "f32") {
         if(raw.size() != length * sizeof(float)) {
            throw std::invalid_argument("Malformed f32 column payload");
         }
         std::vector< float > data(length);
         if(length > 0) {
            std::memcpy(data.data(), raw.data(), raw.size());
         }
         encoding.columns[key] = BatchBuilder::Column{std::move(data), dim};
      } else if(dtype == "i64") {
         if(raw.size() != length * sizeof(int64_t)) {
            throw std::invalid_argument("Malformed i64 column payload");
         }
         std::vector< int64_t > data(length);
         if(length > 0) {
            std::memcpy(data.data(), raw.data(), raw.size());
         }
         encoding.columns[key] = BatchBuilder::Column{std::move(data), dim};
      } else {
         throw std::invalid_argument("Unsupported BatchEncoding column dtype");
      }
   }

   return encoding;
}

nb::dict batch_encoding_state_from_instance(nb::handle self, bool include_metadata)
{
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      self, "BatchEncoding state extraction called with invalid instance"
   );
   validate_batch_encoding_graph_fields(*encoding, "BatchEncoding state extraction");
   nb::dict state = batch_encoding_to_state_dict(*encoding, include_metadata);
   nb::dict py_attrs = batch_encoding_python_attrs_copy(self);
   if(py_attrs.contains(kPythonTensorCacheAttr.data())) {
      py_attrs.attr("pop")(kPythonTensorCacheAttr.data());
   }
   if(nb::len(py_attrs) > 0) {
      state["python_attrs"] = std::move(py_attrs);
   }
   return state;
}

nb::object batch_encoding_object_from_state(const nb::dict& state)
{
   nb::object obj = mifrost_core_batch_encoding_cls_handle()();
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      obj, "Failed to instantiate BatchEncoding during state load"
   );
   *encoding = batch_encoding_from_state_dict(state);
   auto attrs = nb::cast< nb::dict >(obj.attr("__dict__"));
   attrs.clear();
   batch_encoding_apply_python_attrs_from_state(obj, state, attrs);
   clear_owner_tensor_cache(obj);
   return obj;
}

uint64_t schema_fingerprint(const BatchBuilder::BatchEncoding& encoding)
{
   constexpr uint64_t kOffset = 1469598103934665603ULL;
   constexpr uint64_t kPrime = 1099511628211ULL;
   auto fnv_mix_byte = [&](uint64_t& h, unsigned char c) {
      h ^= static_cast< uint64_t >(c);
      h *= kPrime;
   };
   auto fnv_mix_string = [&](uint64_t& h, std::string_view text) {
      for(const unsigned char c : text) {
         fnv_mix_byte(h, c);
      }
      fnv_mix_byte(h, 0xFF);
   };
   auto fnv_mix_int = [&](uint64_t& h, int64_t value) {
      const uint64_t u = static_cast< uint64_t >(value);
      for(int shift = 0; shift < 64; shift += 8) {
         fnv_mix_byte(h, static_cast< unsigned char >((u >> shift) & 0xFFULL));
      }
   };

   uint64_t h = kOffset;
   fnv_mix_string(h, encoding.graph_kind);
   for(const auto& [key, value] : encoding.schema_flags) {
      fnv_mix_string(h, key);
      fnv_mix_byte(h, value ? 1 : 0);
   }

   std::vector< std::pair< std::string, int > > node_feature_dims(
      encoding.node_feature_dims.begin(), encoding.node_feature_dims.end()
   );
   std::ranges::sort(node_feature_dims, [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
   });
   for(const auto& [node_type, dim] : node_feature_dims) {
      fnv_mix_string(h, node_type);
      fnv_mix_int(h, dim);
   }

   const auto& schema = encoding.schema;
   fnv_mix_int(h, schema.version);
   fnv_mix_string(h, schema.graph_kind);
   for(const auto& node_type : schema.node_types) {
      fnv_mix_string(h, node_type);
   }
   for(const auto& edge_type : schema.edge_types) {
      fnv_mix_string(h, edge_type.src);
      fnv_mix_string(h, edge_type.rel);
      fnv_mix_string(h, edge_type.dst);
   }
   for(const auto& spec : schema.node_tensors) {
      fnv_mix_string(h, spec.node_type);
      fnv_mix_string(h, spec.attr);
      fnv_mix_string(h, spec.key);
   }
   for(const auto& spec : schema.edge_tensors) {
      fnv_mix_int(h, spec.edge_type);
      fnv_mix_string(h, spec.attr);
      fnv_mix_string(h, spec.key);
      fnv_mix_string(h, spec.part);
   }
   for(const auto& spec : schema.graph_tensors) {
      fnv_mix_string(h, spec.attr);
      fnv_mix_string(h, spec.key);
      fnv_mix_string(h, spec.ptr_key);
      fnv_mix_string(h, graph_field_mode_name(spec.mode));
      fnv_mix_string(h, graph_field_dtype_name(spec.dtype));
      fnv_mix_int(h, spec.dim);
      fnv_mix_int(h, spec.cat_dim);
      fnv_mix_string(h, graph_field_inc_kind_name(spec.inc.kind));
      fnv_mix_string(h, spec.inc.node_type);
   }
   for(const auto& [key, value] : schema.flags) {
      fnv_mix_string(h, key);
      fnv_mix_byte(h, value ? 1 : 0);
   }
   return h;
}

int64_t batch_encoding_num_nodes(const BatchBuilder::BatchEncoding& encoding)
{
   int64_t total = 0;
   for(const auto& count : std::views::values(encoding.node_counts)) {
      total += count;
   }
   return total;
}

int64_t batch_encoding_num_edges(const BatchBuilder::BatchEncoding& encoding)
{
   int64_t total = 0;
   for(const auto& [key, col] : encoding.columns) {
      if(key.find("/edge_index_0") == std::string::npos) {
         continue;
      }
      std::visit([&](const auto& data) { total += static_cast< int64_t >(data.size()); }, col.data);
   }
   return total;
}

std::vector< std::string > batch_encoding_node_types(const BatchBuilder::BatchEncoding& encoding)
{
   return encoding.schema.node_types;
}

nb::list batch_encoding_edge_types(const BatchBuilder::BatchEncoding& encoding)
{
   nb::list out;
   for(const auto& edge_type : encoding.schema.edge_types) {
      out.append(nb::make_tuple(edge_type.src, edge_type.rel, edge_type.dst));
   }
   return out;
}

nb::dict batch_encoding_as_dict(BatchBuilder::BatchEncoding& encoding, nb::handle owner)
{
   nb::dict tensors;

   for(auto& [key, col] : encoding.columns) {
      const bool is_edge_index = key.find("/edge_index_") != std::string::npos;
      std::visit(
         [&]< typename T >(std::vector< T >& data) {
            if(is_edge_index) {
               tensors[key.c_str()] = vector_to_1d_tensor_view(data, owner);
               return;
            }
            const size_t rows = col.dim > 0 ? data.size() / static_cast< size_t >(col.dim) : 0;
            tensors[key.c_str()] = vector_to_2d_tensor_view(
               data, rows, static_cast< size_t >(col.dim), owner
            );
         },
         col.data
      );
   }

   bool exported_ptr = false;
   for(auto& [node_type, ptr] : encoding.ptrs) {
      if(ptr.size() < 2) {
         continue;
      }
      exported_ptr = true;
      tensors[(node_type + "/ptr").c_str()] = vector_to_1d_tensor_view(ptr, owner);
      tensors[(node_type + "/batch").c_str()] = vector_to_1d_tensor_owned(ptr_to_batch(ptr));
   }
   if(not exported_ptr) {
      for(const auto& [node_type, count] : encoding.node_counts) {
         if(count <= 0) {
            continue;
         }
         std::vector< int64_t > ptr{0, count};
         tensors[(node_type + "/ptr").c_str()] = vector_to_1d_tensor_owned(std::move(ptr));
         tensors[(node_type + "/batch").c_str()] = vector_to_1d_tensor_owned(
            std::vector< int64_t >(count, 0)
         );
      }
   }

   for(auto& [attr, field] : encoding.graph_fields) {
      const std::string key = "__graph__/" + attr;
      std::visit(
         [&]< typename T >(std::vector< T >& data) {
            if(field.spec.dim == 1) {
               tensors[key.c_str()] = vector_to_1d_tensor_view(data, owner);
               return;
            }
            const bool cat_dim_one = (field.spec.mode == GraphFieldMode::CAT
                                      or field.spec.mode == GraphFieldMode::RAGGED_CAT)
                                     and graph_field_cat_dim_is_one(field.spec.cat_dim);
            const size_t rows = cat_dim_one ? static_cast< size_t >(field.spec.dim)
                                            : data.size() / static_cast< size_t >(field.spec.dim);
            const size_t cols = cat_dim_one ? data.size() / static_cast< size_t >(field.spec.dim)
                                            : static_cast< size_t >(field.spec.dim);
            tensors[key.c_str()] = vector_to_2d_tensor_view(data, rows, cols, owner);
         },
         field.values
      );
      if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
         tensors[(key + "/ptr").c_str()] = vector_to_1d_tensor_view(field.ptr, owner);
      }
   }

   nb::dict out;
   out["tensors"] = std::move(tensors);
   out["schema"] = encoding.schema.to_dict();

   nb::dict node_names_dict;
   for(const auto& [node_type, names] : encoding.node_names) {
      node_names_dict[node_type.c_str()] = nb::cast(names);
   }
   out["node_names"] = std::move(node_names_dict);

   nb::dict dims_dict;
   for(const auto& [node_type, dim] : encoding.node_feature_dims) {
      dims_dict[node_type.c_str()] = dim;
   }
   out["node_feature_dims"] = std::move(dims_dict);
   out["object_names"] = nb::cast(encoding.object_names);
   out["num_graphs"] = encoding.num_graphs;

   if(not encoding.graph_attrs.empty()) {
      nb::dict graph_attrs_dict;
      for(const auto& [key, value] : encoding.graph_attrs) {
         std::visit([&](const auto& v) { graph_attrs_dict[key.c_str()] = nb::cast(v); }, value);
      }
      out["graph_attrs"] = std::move(graph_attrs_dict);
   }

   return out;
}

nb::object
batch_encoding_as_pyg(const BatchBuilder::BatchEncoding& encoding, std::optional< bool > as_batch)
{
   validate_batch_encoding_graph_fields(encoding, "BatchEncoding.as_pyg");
   const bool want_batch = as_batch.value_or(encoding.num_graphs != 1);
   BatchBuilder builder;
   builder.set_graph_kind(encoding.graph_kind);
   builder.load_from_batch_encoding(encoding);
   nb::object pyg_batch = builder.build_pyg();

   if(not want_batch and encoding.num_graphs != 1) {
      throw std::invalid_argument("BatchEncoding.as_pyg(as_batch=False) requires num_graphs == 1");
   }

   if(not want_batch) {
      if(encoding.graph_kind == "homo") {
         return batch_to_single_homo_data(pyg_batch);
      }
      return batch_to_single_hetero_data(pyg_batch);
   }

   return pyg_batch;
}

nb::object owner_target_device(nb::handle owner)
{
   auto attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(not attrs.contains(kPythonTensorDeviceAttr.data())) {
      return nb::none();
   }
   return nb::borrow< nb::object >(attrs[kPythonTensorDeviceAttr.data()]);
}

std::optional< nb::dict > owner_tensor_cache_if_present(nb::handle owner)
{
   auto attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(not attrs.contains(kPythonTensorCacheAttr.data())) {
      return std::nullopt;
   }
   auto raw_cache = nb::borrow< nb::object >(attrs[kPythonTensorCacheAttr.data()]);
   if(not nb::isinstance< nb::dict >(raw_cache)) {
      throw std::invalid_argument("BatchEncoding internal tensor cache must be a dict");
   }
   return nb::cast< nb::dict >(raw_cache);
}

void clear_owner_tensor_cache(nb::handle owner)
{
   auto attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(attrs.contains(kPythonTensorCacheAttr.data())) {
      attrs.attr("pop")(kPythonTensorCacheAttr.data());
   }
}

bool is_torch_tensor(nb::handle value)
{
   nb::handle torch = torch_module_handle();
   return nb::isinstance(value, torch.attr("Tensor"));
}

nb::object move_object_to_device(nb::handle value, nb::handle device)
{
   if(device.is_none()) {
      return nb::borrow< nb::object >(value);
   }
   if(is_torch_tensor(value)) {
      return nb::borrow< nb::object >(value).attr("to")(device);
   }
   if(nb::isinstance< nb::dict >(value)) {
      nb::dict out;
      for(auto [k, v] : nb::borrow< nb::dict >(value)) {
         out[k] = move_object_to_device(nb::borrow< nb::object >(v), device);
      }
      return out;
   }
   if(nb::isinstance< nb::list >(value)) {
      nb::list out;
      for(nb::handle item : nb::borrow< nb::list >(value)) {
         out.append(move_object_to_device(item, device));
      }
      return out;
   }
   if(nb::isinstance< nb::tuple >(value)) {
      nb::list tmp;
      for(nb::handle item : nb::borrow< nb::tuple >(value)) {
         tmp.append(move_object_to_device(item, device));
      }
      return builtins_tuple_ctor_handle()(tmp);
   }
   return nb::borrow< nb::object >(value);
}

void set_owner_target_device(nb::handle owner, nb::handle device)
{
   nb::dict attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(device.is_none()) {
      if(attrs.contains(kPythonTensorDeviceAttr.data())) {
         attrs.attr("pop")(kPythonTensorDeviceAttr.data());
      }
      return;
   }
   nb::handle torch = torch_module_handle();
   attrs[kPythonTensorDeviceAttr.data()] = torch.attr("device")(device);
}

std::string batch_encoding_repr(nb::handle self, const BatchBuilder::BatchEncoding& encoding)
{
   const auto native_field_keys = batch_encoding_native_graph_field_keys(encoding);
   nb::dict attrs = batch_encoding_python_attrs(self);
   std::set< std::string > python_attr_keys;
   for(auto [key_obj, value_obj] : attrs) {
      (void) value_obj;
      const std::string key = py_string(key_obj);
      if(is_reserved_python_attr_key(key) or native_field_keys.contains(key)) {
         continue;
      }
      python_attr_keys.insert(key);
   }

   const auto node_type_reprs = encoding.schema.node_types
                                | std::views::transform([](const std::string& value) {
                                     return ReprQuoted{value};
                                  });
   const auto edge_type_reprs = encoding.schema.edge_types
                                | std::views::transform([](const EdgeType& value) {
                                     return ReprEdgeType{&value};
                                  });
   const auto field_key_reprs = native_field_keys | std::views::transform([](const auto& value) {
                                   return ReprQuoted{value};
                                });
   const auto python_attr_reprs = python_attr_keys | std::views::transform([](const auto& value) {
                                     return ReprQuoted{value};
                                  });

   nb::object device = owner_target_device(self);
   if(device.is_none()) {
      return fmt::format(
         "BatchEncoding(graph_kind={}, num_graphs={}, num_nodes={}, num_edges={}, "
         "node_types=[{}], edge_types=[{}], fields=[{}], python_attrs=[{}], device=None)",
         ReprQuoted{encoding.graph_kind},
         encoding.num_graphs,
         batch_encoding_num_nodes(encoding),
         batch_encoding_num_edges(encoding),
         fmt::join(node_type_reprs, ", "),
         fmt::join(edge_type_reprs, ", "),
         fmt::join(field_key_reprs, ", "),
         fmt::join(python_attr_reprs, ", ")
      );
   }
   const std::string device_repr = py_string(nb::str(device));
   return fmt::format(
      "BatchEncoding(graph_kind={}, num_graphs={}, num_nodes={}, num_edges={}, "
      "node_types=[{}], edge_types=[{}], fields=[{}], python_attrs=[{}], device={})",
      ReprQuoted{encoding.graph_kind},
      encoding.num_graphs,
      batch_encoding_num_nodes(encoding),
      batch_encoding_num_edges(encoding),
      fmt::join(node_type_reprs, ", "),
      fmt::join(edge_type_reprs, ", "),
      fmt::join(field_key_reprs, ", "),
      fmt::join(python_attr_reprs, ", "),
      ReprQuoted{device_repr}
   );
}

std::string batch_encoding_str(nb::handle self, const BatchBuilder::BatchEncoding& encoding)
{
   const auto native_field_keys = batch_encoding_native_graph_field_keys(encoding);
   nb::dict attrs = batch_encoding_python_attrs(self);
   std::set< std::string > python_attr_keys;
   for(auto [key_obj, value_obj] : attrs) {
      (void) value_obj;
      const std::string key = py_string(key_obj);
      if(is_reserved_python_attr_key(key) or native_field_keys.contains(key)) {
         continue;
      }
      python_attr_keys.insert(key);
   }

   const auto edge_type_views = encoding.schema.edge_types
                                | std::views::transform([](const EdgeType& value) {
                                     return DisplayEdgeType{&value};
                                  });
   nb::object device = owner_target_device(self);
   const std::string device_str = device.is_none() ? "None" : py_string(nb::str(device));

   return fmt::format(
      "BatchEncoding(graph_kind={}, num_graphs={}, num_nodes={}, num_edges={}, "
      "node_types=[{}], edge_types=[{}], fields=[{}], python_attrs=[{}], device={})",
      encoding.graph_kind,
      encoding.num_graphs,
      batch_encoding_num_nodes(encoding),
      batch_encoding_num_edges(encoding),
      fmt::join(encoding.schema.node_types, ", "),
      fmt::join(edge_type_views, ", "),
      fmt::join(native_field_keys, ", "),
      fmt::join(python_attr_keys, ", "),
      device_str
   );
}

void materialize_owner_tensor_cache(nb::handle owner, BatchBuilder::BatchEncoding& encoding)
{
   clear_owner_tensor_cache(owner);

   nb::dict cache;

   const auto native_tensor_keys = batch_encoding_native_tensor_keys(encoding);

   for(const auto& key : native_tensor_keys) {
      cache[key.c_str()] = batch_encoding_get_native_tensor(encoding, key, owner);
   }

   nb::dict attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   attrs[kPythonTensorCacheAttr.data()] = std::move(cache);
}

nb::object to_mapping_proxy(const nb::dict& mapping)
{
   return mapping_proxy_type_ctor_handle()(mapping);
}

nb::object zeros_f32_on_owner_device(nb::handle owner, int64_t rows, int64_t cols)
{
   nb::handle torch = torch_module_handle();
   nb::object device = owner_target_device(owner);
   if(device.is_none()) {
      return torch.attr("zeros")(nb::make_tuple(rows, cols), "dtype"_a = torch.attr("float32"));
   }
   return torch.attr("zeros")(
      nb::make_tuple(rows, cols), "dtype"_a = torch.attr("float32"), "device"_a = device
   );
}

std::optional< std::string >
find_node_attr_key(const Schema& schema, std::string_view node_type, std::string_view attr)
{
   for(const auto& spec : schema.node_tensors) {
      if(spec.node_type == node_type and spec.attr == attr) {
         return spec.key;
      }
   }
   return std::nullopt;
}

std::pair< std::optional< std::string >, std::optional< std::string > >
find_edge_index_keys(const Schema& schema, int edge_type_idx)
{
   std::optional< std::string > key0;
   std::optional< std::string > key1;
   for(const auto& spec : schema.edge_tensors) {
      if(spec.edge_type != edge_type_idx or spec.attr != "edge_index") {
         continue;
      }
      if(spec.part == "0") {
         key0 = spec.key;
      } else if(spec.part == "1") {
         key1 = spec.key;
      }
   }
   return {key0, key1};
}

std::optional< std::string > find_edge_attr_key(const Schema& schema, int edge_type_idx)
{
   for(const auto& spec : schema.edge_tensors) {
      if(spec.edge_type == edge_type_idx and spec.attr == "edge_attr") {
         return spec.key;
      }
   }
   return std::nullopt;
}

nb::tuple edge_type_to_tuple(const EdgeType& edge_type)
{
   return nb::make_tuple(edge_type.src, edge_type.rel, edge_type.dst);
}

class HeteroBatchEncodingView {
  public:
   explicit HeteroBatchEncodingView(nb::object owner) : owner_(std::move(owner))
   {
      encoding_ = require_instance_ptr< BatchBuilder::BatchEncoding >(
         owner_, "HeteroBatchEncodingView created with invalid BatchEncoding instance"
      );
   }

   [[nodiscard]] int64_t num_graphs() const { return encoding_->num_graphs; }
   [[nodiscard]] int64_t num_nodes() const { return batch_encoding_num_nodes(*encoding_); }
   [[nodiscard]] int64_t num_edges() const { return batch_encoding_num_edges(*encoding_); }
   [[nodiscard]] std::string graph_kind() const { return encoding_->graph_kind; }

   [[nodiscard]] std::vector< std::string > node_types() const
   {
      return encoding_->schema.node_types;
   }

   [[nodiscard]] nb::list edge_types() const { return batch_encoding_edge_types(*encoding_); }
   [[nodiscard]] std::vector< std::string > object_names() const { return encoding_->object_names; }

   nb::object x_dict()
   {
      if(x_dict_cache_.is_valid()) {
         return x_dict_cache_;
      }

      nb::dict out;
      for(const auto& node_type : encoding_->schema.node_types) {
         if(const auto key = find_node_attr_key(encoding_->schema, node_type, "x");
            key.has_value() and has_tensor(*key)) {
            out[node_type.c_str()] = tensor(*key);
            continue;
         }
         if(const auto it = encoding_->node_feature_dims.find(node_type);
            it != encoding_->node_feature_dims.end()) {
            int64_t count = 0;
            if(const auto count_it = encoding_->node_counts.find(node_type);
               count_it != encoding_->node_counts.end()) {
               count = std::max< int64_t >(0, count_it->second);
            }
            out[node_type.c_str()] = zeros_f32_on_owner_device(owner_, count, it->second);
         }
      }
      x_dict_cache_ = to_mapping_proxy(out);
      return x_dict_cache_;
   }

   nb::object edge_index_dict()
   {
      if(edge_index_dict_cache_.is_valid()) {
         return edge_index_dict_cache_;
      }

      nb::dict out;
      for(size_t idx = 0; idx < encoding_->schema.edge_types.size(); ++idx) {
         const auto [key0, key1] = find_edge_index_keys(encoding_->schema, static_cast< int >(idx));
         if(not key0.has_value() or not key1.has_value()) {
            continue;
         }
         if(not has_tensor(*key0) or not has_tensor(*key1)) {
            continue;
         }
         nb::list pair;
         pair.append(tensor(*key0));
         pair.append(tensor(*key1));
         nb::handle torch = torch_module_handle();
         out[edge_type_to_tuple(encoding_->schema.edge_types[idx])] = torch.attr("stack")(
            pair, "dim"_a = 0
         );
      }
      edge_index_dict_cache_ = to_mapping_proxy(out);
      return edge_index_dict_cache_;
   }

   nb::object batch_dict()
   {
      if(batch_dict_cache_.is_valid()) {
         return batch_dict_cache_;
      }

      nb::dict out;
      for(const auto& node_type : encoding_->schema.node_types) {
         const std::string key = node_type + "/batch";
         if(has_tensor(key)) {
            out[node_type.c_str()] = tensor(key);
         }
      }
      batch_dict_cache_ = to_mapping_proxy(out);
      return batch_dict_cache_;
   }

   nb::object ptr_dict()
   {
      if(ptr_dict_cache_.is_valid()) {
         return ptr_dict_cache_;
      }

      nb::dict out;
      for(const auto& node_type : encoding_->schema.node_types) {
         const std::string key = node_type + "/ptr";
         if(has_tensor(key)) {
            out[node_type.c_str()] = tensor(key);
         }
      }
      ptr_dict_cache_ = to_mapping_proxy(out);
      return ptr_dict_cache_;
   }

   nb::object edge_attr_dict()
   {
      if(edge_attr_dict_cache_.is_valid()) {
         return edge_attr_dict_cache_;
      }

      nb::dict out;
      for(size_t idx = 0; idx < encoding_->schema.edge_types.size(); ++idx) {
         const auto key = find_edge_attr_key(encoding_->schema, static_cast< int >(idx));
         if(not key.has_value() or not has_tensor(*key)) {
            continue;
         }
         out[edge_type_to_tuple(encoding_->schema.edge_types[idx])] = tensor(*key);
      }
      edge_attr_dict_cache_ = to_mapping_proxy(out);
      return edge_attr_dict_cache_;
   }

   void set_device(nb::handle device)
   {
      if(device.is_none()) {
         return;
      }
      set_owner_target_device(owner_, device);
      materialize_owner_tensor_cache(owner_, *encoding_);
      clear_caches();
      prewarm_caches();
   }

  private:
   void clear_caches()
   {
      tensor_cache_ = nb::dict();
      x_dict_cache_ = nb::object();
      edge_index_dict_cache_ = nb::object();
      batch_dict_cache_ = nb::object();
      ptr_dict_cache_ = nb::object();
      edge_attr_dict_cache_ = nb::object();
   }

   void prewarm_caches()
   {
      (void) x_dict();
      (void) edge_index_dict();
      (void) batch_dict();
      (void) ptr_dict();
      (void) edge_attr_dict();
   }

   bool has_tensor(const std::string& key) const
   {
      return batch_encoding_has_native_tensor(*encoding_, key);
   }

   nb::object tensor(const std::string& key) const
   {
      if(tensor_cache_.contains(key.c_str())) {
         return nb::borrow< nb::object >(tensor_cache_[key.c_str()]);
      }
      if(auto owner_cache = owner_tensor_cache_if_present(owner_);
         owner_cache.has_value() and owner_cache->contains(key.c_str())) {
         auto value = nb::borrow< nb::object >((*owner_cache)[key.c_str()]);
         tensor_cache_[key.c_str()] = value;
         return value;
      }
      nb::object value = batch_encoding_get_native_tensor(*encoding_, key, owner_);
      tensor_cache_[key.c_str()] = value;
      if(auto owner_cache = owner_tensor_cache_if_present(owner_); owner_cache.has_value()) {
         (*owner_cache)[key.c_str()] = value;
      }
      return value;
   }

   nb::object owner_;
   BatchBuilder::BatchEncoding* encoding_ = nullptr;
   nb::dict tensor_cache_;
   nb::object x_dict_cache_;
   nb::object edge_index_dict_cache_;
   nb::object batch_dict_cache_;
   nb::object ptr_dict_cache_;
   nb::object edge_attr_dict_cache_;
};

class HomoBatchEncodingView {
  public:
   explicit HomoBatchEncodingView(nb::object owner) : owner_(std::move(owner))
   {
      encoding_ = require_instance_ptr< BatchBuilder::BatchEncoding >(
         owner_, "HomoBatchEncodingView created with invalid BatchEncoding instance"
      );
   }

   [[nodiscard]] int64_t num_graphs() const { return encoding_->num_graphs; }
   [[nodiscard]] int64_t num_nodes() const { return batch_encoding_num_nodes(*encoding_); }
   [[nodiscard]] int64_t num_edges() const { return batch_encoding_num_edges(*encoding_); }
   [[nodiscard]] std::string graph_kind() const { return encoding_->graph_kind; }
   [[nodiscard]] std::vector< std::string > node_types() const
   {
      return encoding_->schema.node_types;
   }
   [[nodiscard]] nb::list edge_types() const { return batch_encoding_edge_types(*encoding_); }
   [[nodiscard]] std::vector< std::string > object_names() const { return encoding_->object_names; }

   nb::object x()
   {
      if(x_ready_) {
         return x_cache_;
      }
      x_ready_ = true;
      x_cache_ = nb::none();
      if(encoding_->schema.node_types.empty()) {
         return x_cache_;
      }
      const std::string& node_type = encoding_->schema.node_types.front();
      if(const auto key = find_node_attr_key(encoding_->schema, node_type, "x");
         key.has_value() and has_tensor(*key)) {
         x_cache_ = tensor(*key);
         return x_cache_;
      }
      if(const auto it = encoding_->node_feature_dims.find(node_type);
         it != encoding_->node_feature_dims.end()) {
         int64_t count = 0;
         if(const auto count_it = encoding_->node_counts.find(node_type);
            count_it != encoding_->node_counts.end()) {
            count = std::max< int64_t >(0, count_it->second);
         }
         x_cache_ = zeros_f32_on_owner_device(owner_, count, it->second);
      }
      return x_cache_;
   }

   nb::object edge_index()
   {
      if(edge_index_ready_) {
         return edge_index_cache_;
      }
      edge_index_ready_ = true;
      if(encoding_->schema.edge_types.empty()) {
         return edge_index_cache_ = nb::none();
      }
      const auto [key0, key1] = find_edge_index_keys(encoding_->schema, 0);
      if(not key0.has_value() or not key1.has_value() or not has_tensor(*key0)
         or not has_tensor(*key1)) {
         return edge_index_cache_ = nb::none();
      }
      nb::list pair;
      pair.append(tensor(*key0));
      pair.append(tensor(*key1));
      nb::handle torch = torch_module_handle();
      edge_index_cache_ = torch.attr("stack")(pair, "dim"_a = 0);
      return edge_index_cache_;
   }

   nb::object batch()
   {
      if(batch_ready_) {
         return batch_cache_;
      }
      batch_ready_ = true;
      std::string key;
      if(encoding_->schema.node_types.empty() or std::invoke([&] {
            key = encoding_->schema.node_types.front() + "/batch";
            return not has_tensor(key);
         })) {
         return batch_cache_ = nb::none();
      }
      return batch_cache_ = tensor(key);
   }

   nb::object ptr()
   {
      if(ptr_ready_) {
         return ptr_cache_;
      }
      ptr_ready_ = true;
      std::string key;
      if(encoding_->schema.node_types.empty() or std::invoke([&] {
            key = encoding_->schema.node_types.front() + "/ptr";
            return not has_tensor(key);
         })) {
         return ptr_cache_ = nb::none();
      }
      return ptr_cache_ = tensor(key);
   }

   nb::object edge_attr()
   {
      if(edge_attr_ready_) {
         return edge_attr_cache_;
      }
      edge_attr_ready_ = true;
      edge_attr_cache_ = nb::none();
      if(encoding_->schema.edge_types.empty()) {
         return edge_attr_cache_;
      }
      const auto key = find_edge_attr_key(encoding_->schema, 0);
      if(key.has_value() and has_tensor(*key)) {
         edge_attr_cache_ = tensor(*key);
      }
      return edge_attr_cache_;
   }

   void set_device(nb::handle device)
   {
      if(device.is_none()) {
         return;
      }
      set_owner_target_device(owner_, device);
      materialize_owner_tensor_cache(owner_, *encoding_);
      clear_caches();
      prewarm_caches();
   }

  private:
   void clear_caches()
   {
      tensor_cache_ = nb::dict();
      x_ready_ = false;
      edge_index_ready_ = false;
      batch_ready_ = false;
      ptr_ready_ = false;
      edge_attr_ready_ = false;
      x_cache_ = nb::object();
      edge_index_cache_ = nb::object();
      batch_cache_ = nb::object();
      ptr_cache_ = nb::object();
      edge_attr_cache_ = nb::object();
   }

   void prewarm_caches()
   {
      (void) x();
      (void) edge_index();
      (void) batch();
      (void) ptr();
      (void) edge_attr();
   }

   bool has_tensor(const std::string& key)
   {
      return batch_encoding_has_native_tensor(*encoding_, key);
   }

   nb::object tensor(const std::string& key)
   {
      if(tensor_cache_.contains(key.c_str())) {
         return nb::borrow< nb::object >(tensor_cache_[key.c_str()]);
      }
      if(auto owner_cache = owner_tensor_cache_if_present(owner_);
         owner_cache.has_value() and owner_cache->contains(key.c_str())) {
         auto value = nb::borrow< nb::object >((*owner_cache)[key.c_str()]);
         tensor_cache_[key.c_str()] = value;
         return value;
      }
      nb::object value = batch_encoding_get_native_tensor(*encoding_, key, owner_);
      tensor_cache_[key.c_str()] = value;
      if(auto owner_cache = owner_tensor_cache_if_present(owner_); owner_cache.has_value()) {
         (*owner_cache)[key.c_str()] = value;
      }
      return value;
   }

   nb::object owner_;
   BatchBuilder::BatchEncoding* encoding_ = nullptr;
   nb::dict tensor_cache_;
   bool x_ready_ = false;
   bool edge_index_ready_ = false;
   bool batch_ready_ = false;
   bool ptr_ready_ = false;
   bool edge_attr_ready_ = false;
   nb::object x_cache_;
   nb::object edge_index_cache_;
   nb::object batch_cache_;
   nb::object ptr_cache_;
   nb::object edge_attr_cache_;
};

}  // namespace

void init_batch_encoding(nb::module_& m)
{
   register_mapview_maybe< absl::btree_map< std::string, bool > >(m);
   register_mapview_maybe< hash_map< std::string, int > >(m);

   auto batch_builder_cls =
      nb::class_< BatchBuilder >(m, "BatchBuilder")
         .def(nb::init<>())
         .def(
            "add_node_features",
            [](BatchBuilder& builder,
               const std::string& node_type,
               const std::string& attr_name,
               nb::ndarray< nb::numpy, float > data) {
               if(data.ndim() != 1 and data.ndim() != 2) {
                  throw std::invalid_argument("add_node_features expects a 1D/2D array");
               }
               const int feature_dim = data.ndim() == 2 ? static_cast< int >(data.shape(1)) : 1;
               const auto count = static_cast< size_t >(data.size());
               builder.add_node_features(
                  node_type, attr_name, std::span< const float >(data.data(), count), feature_dim
               );
            }
         )
         .def(
            "add_edges",
            [](BatchBuilder& builder,
               const std::string& src_type,
               const std::string& rel_type,
               const std::string& dst_type,
               nb::ndarray< nb::numpy, int64_t > src,
               nb::ndarray< nb::numpy, int64_t > dst) {
               if(src.ndim() != 1 or dst.ndim() != 1) {
                  throw std::invalid_argument("add_edges expects 1D arrays for src/dst indices");
               }
               if(src.size() != dst.size()) {
                  throw std::invalid_argument("add_edges expects src/dst arrays of equal length");
               }
               builder.add_edges(
                  src_type,
                  rel_type,
                  dst_type,
                  std::span< const int64_t >(src.data(), src.size()),
                  std::span< const int64_t >(dst.data(), dst.size())
               );
            }
         )
         .def(
            "add_edge_features",
            [](BatchBuilder& builder,
               const std::string& src_type,
               const std::string& rel_type,
               const std::string& dst_type,
               const std::string& attr_name,
               nb::ndarray< nb::numpy, float > data) {
               if(data.ndim() != 1 and data.ndim() != 2) {
                  throw std::invalid_argument("add_edge_features expects a 1D/2D array");
               }
               const int feature_dim = data.ndim() == 2 ? static_cast< int >(data.shape(1)) : 1;
               const auto count = static_cast< size_t >(data.size());
               builder.add_edge_features(
                  src_type,
                  rel_type,
                  dst_type,
                  attr_name,
                  std::span< const float >(data.data(), count),
                  feature_dim
               );
            }
         )
         .def("add_nodes", &BatchBuilder::add_nodes)
         .def("add_edge", &BatchBuilder::add_edge)
         .def("set_node_names", &BatchBuilder::set_node_names)
         .def("set_object_names", &BatchBuilder::set_object_names)
         .def("build", &BatchBuilder::build)
         .def("build_pyg", &BatchBuilder::build_pyg)
         .def("append_batch_encoding", &BatchBuilder::append_batch_encoding)
         .def(
            "load_from_batch_encoding",
            nb::overload_cast< const BatchBuilder::BatchEncoding& >(
               &BatchBuilder::load_from_batch_encoding
            )
         )
         .def("next_graph", &BatchBuilder::next_graph)
         .def("set_graph_kind", &BatchBuilder::set_graph_kind, "kind"_a)
         .def("set_schema_flag", &BatchBuilder::set_schema_flag, "key"_a, "value"_a)
         .def(
            "schema_flags_view",
            [](nb::handle self) {
               auto* builder = require_instance_ptr< BatchBuilder >(
                  self, "BatchBuilder.schema_flags_view called with invalid instance"
               );
               return make_map_view(builder->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def(
            "node_feature_dims_view",
            [](nb::handle self) {
               auto* builder = require_instance_ptr< BatchBuilder >(
                  self, "BatchBuilder.node_feature_dims_view called with invalid instance"
               );
               return make_map_view(builder->node_feature_dims, self);
            },
            nb::rv_policy::move
         )
         .def("field_keys", &BatchBuilder::field_keys)
         .def(
            "field_specs",
            [](const BatchBuilder& builder) {
               nb::dict out;
               for(const auto& [key, spec] : builder.field_specs()) {
                  out[key.c_str()] = graph_field_spec_to_dict(spec);
               }
               return out;
            }
         )
         .def(
            "register_field",
            [](BatchBuilder& builder, const std::string& key, const nb::dict& spec) {
               builder.register_field(key, graph_field_spec_from_dict(spec));
            },
            "key"_a,
            "spec"_a
         )
         .def(
            "set_field",
            [](BatchBuilder& builder, const std::string& key, nb::handle value) {
               const auto spec = builder.get_graph_field_spec(key);
               if(spec.dtype == GraphFieldDType::F32) {
                  auto input = coerce_numeric_values< float >(value);
                  auto values = normalize_graph_field_input(key, spec, std::move(input));
                  builder.set_field(key, std::span< const float >(values.data(), values.size()));
               } else {
                  auto input = coerce_numeric_values< int64_t >(value);
                  auto values = normalize_graph_field_input(key, spec, std::move(input));
                  builder.set_field(key, std::span< const int64_t >(values.data(), values.size()));
               }
            },
            "key"_a,
            "value"_a
         )
         .def(
            "set_fields",
            [](BatchBuilder& builder, const nb::dict& values) {
               for(auto [key_obj, value_obj] : values) {
                  const std::string key = py_string(key_obj);
                  const auto spec = builder.get_graph_field_spec(key);
                  if(spec.dtype == GraphFieldDType::F32) {
                     auto input = coerce_numeric_values< float >(value_obj);
                     auto data = normalize_graph_field_input(key, spec, std::move(input));
                     builder.set_field(key, std::span< const float >(data.data(), data.size()));
                  } else {
                     auto input = coerce_numeric_values< int64_t >(value_obj);
                     auto data = normalize_graph_field_input(key, spec, std::move(input));
                     builder.set_field(key, std::span< const int64_t >(data.data(), data.size()));
                  }
               }
            },
            "values"_a
         );

   nb::class_< HeteroBatchEncodingView >(m, "HeteroBatchEncodingView")
      .def_prop_ro("num_graphs", &HeteroBatchEncodingView::num_graphs)
      .def_prop_ro("num_nodes", &HeteroBatchEncodingView::num_nodes)
      .def_prop_ro("num_edges", &HeteroBatchEncodingView::num_edges)
      .def_prop_ro("graph_kind", &HeteroBatchEncodingView::graph_kind)
      .def_prop_ro("node_types", &HeteroBatchEncodingView::node_types)
      .def_prop_ro("edge_types", &HeteroBatchEncodingView::edge_types)
      .def_prop_ro("object_names", &HeteroBatchEncodingView::object_names)
      .def_prop_ro(
         "x_dict",
         &HeteroBatchEncodingView::x_dict,
         nb::sig("def x_dict(self) -> collections.abc.Mapping[str, torch.Tensor]")
      )
      .def_prop_ro(
         "edge_index_dict",
         &HeteroBatchEncodingView::edge_index_dict,
         nb::sig(
            "def edge_index_dict(self) -> collections.abc.Mapping[tuple[str, str, str], "
            "torch.Tensor]"
         )
      )
      .def_prop_ro(
         "batch_dict",
         &HeteroBatchEncodingView::batch_dict,
         nb::sig("def batch_dict(self) -> collections.abc.Mapping[str, torch.Tensor]")
      )
      .def_prop_ro(
         "ptr_dict",
         &HeteroBatchEncodingView::ptr_dict,
         nb::sig("def ptr_dict(self) -> collections.abc.Mapping[str, torch.Tensor]")
      )
      .def_prop_ro(
         "edge_attr_dict",
         &HeteroBatchEncodingView::edge_attr_dict,
         nb::sig(
            "def edge_attr_dict(self) -> collections.abc.Mapping[tuple[str, str, str], "
            "torch.Tensor]"
         )
      )
      .def(
         "to",
         [](HeteroBatchEncodingView& view, nb::handle device) -> HeteroBatchEncodingView& {
            view.set_device(device);
            return view;
         },
         "device"_a,
         nb::rv_policy::reference_internal
      );

   nb::class_< HomoBatchEncodingView >(m, "HomoBatchEncodingView")
      .def_prop_ro("num_graphs", &HomoBatchEncodingView::num_graphs)
      .def_prop_ro("num_nodes", &HomoBatchEncodingView::num_nodes)
      .def_prop_ro("num_edges", &HomoBatchEncodingView::num_edges)
      .def_prop_ro("graph_kind", &HomoBatchEncodingView::graph_kind)
      .def_prop_ro("node_types", &HomoBatchEncodingView::node_types)
      .def_prop_ro("edge_types", &HomoBatchEncodingView::edge_types)
      .def_prop_ro("object_names", &HomoBatchEncodingView::object_names)
      .def_prop_ro("x", &HomoBatchEncodingView::x, nb::sig("def x(self) -> torch.Tensor | None"))
      .def_prop_ro(
         "edge_index",
         &HomoBatchEncodingView::edge_index,
         nb::sig("def edge_index(self) -> torch.Tensor | None")
      )
      .def_prop_ro(
         "batch", &HomoBatchEncodingView::batch, nb::sig("def batch(self) -> torch.Tensor | None")
      )
      .def_prop_ro(
         "ptr", &HomoBatchEncodingView::ptr, nb::sig("def ptr(self) -> torch.Tensor | None")
      )
      .def_prop_ro(
         "edge_attr",
         &HomoBatchEncodingView::edge_attr,
         nb::sig("def edge_attr(self) -> torch.Tensor | None")
      )
      .def(
         "to",
         [](HomoBatchEncodingView& view, nb::handle device) -> HomoBatchEncodingView& {
            view.set_device(device);
            return view;
         },
         "device"_a,
         nb::rv_policy::reference_internal
      );

   auto batch_encoding_cls =
      nb::class_< BatchBuilder::BatchEncoding >(m, "BatchEncoding", nb::dynamic_attr())
         .def(nb::init<>())
         .def_ro("num_graphs", &BatchBuilder::BatchEncoding::num_graphs)
         .def_prop_ro("num_nodes", &batch_encoding_num_nodes)
         .def_prop_ro("num_edges", &batch_encoding_num_edges)
         .def_prop_ro("node_types", &batch_encoding_node_types)
         .def_prop_ro("edge_types", &batch_encoding_edge_types)
         .def_ro("graph_kind", &BatchBuilder::BatchEncoding::graph_kind)
         .def_ro("schema", &BatchBuilder::BatchEncoding::schema)
         .def_prop_ro(
            "schema_flags",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.schema_flags called with invalid instance"
               );
               return make_map_view(encoding->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def_ro("node_feature_dims", &BatchBuilder::BatchEncoding::node_feature_dims)
         .def_ro("graph_attrs", &BatchBuilder::BatchEncoding::graph_attrs)
         .def(
            "schema_flags_view",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.schema_flags_view called with invalid instance"
               );
               return make_map_view(encoding->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def(
            "node_feature_dims_view",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.node_feature_dims_view called with invalid instance"
               );
               return make_map_view(encoding->node_feature_dims, self);
            },
            nb::rv_policy::move
         )
         .def(
            "as_dict",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.as_dict called with invalid instance"
               );
               return batch_encoding_as_dict(*encoding, self);
            }
         )
         .def(
            "to",
            [](nb::handle self, nb::handle device) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.to called with invalid instance"
               );
               (void) encoding;
               if(device.is_none()) {
                  return nb::borrow< nb::object >(self);
               }
               set_owner_target_device(self, device);
               nb::object normalized = owner_target_device(self);
               nb::dict attrs = batch_encoding_python_attrs(self);
               const auto native_keys = batch_encoding_native_graph_field_keys(*encoding);
               for(auto [key_obj, value_obj] : attrs) {
                  const std::string key = py_string(key_obj);
                  if(is_reserved_python_attr_key(key) or native_keys.contains(key)) {
                     continue;
                  }
                  attrs[key_obj] = move_object_to_device(
                     nb::borrow< nb::object >(value_obj), normalized
                  );
               }
               materialize_owner_tensor_cache(self, *encoding);
               return nb::borrow< nb::object >(self);
            },
            "device"_a
         )
         .def(
            "register_field_specs",
            [](nb::handle self, const nb::dict& specs) {
               register_batch_encoding_field_specs(self, specs);
            },
            "specs"_a
         )
         .def(
            "set_field",
            [](nb::handle self, const std::string& key, nb::handle value) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.set_field called with invalid instance"
               );
               set_batch_encoding_graph_field(*encoding, key, value);
               clear_owner_tensor_cache(self);
            },
            "key"_a,
            "value"_a
         )
         .def(
            "set_fields",
            [](nb::handle self, const nb::dict& values) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.set_fields called with invalid instance"
               );
               set_batch_encoding_graph_fields(*encoding, values);
               clear_owner_tensor_cache(self);
            },
            "values"_a
         )
         .def("field_specs", [](nb::handle self) { return batch_encoding_field_specs(self); })
         .def(
            "has_field",
            [](nb::handle self, const std::string& key) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.has_field called with invalid instance"
               );
               return batch_encoding_has_graph_field(*encoding, key);
            },
            "key"_a
         )
         .def(
            "get_field",
            [](nb::handle self, const std::string& key) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.get_field called with invalid instance"
               );
               return batch_encoding_get_graph_field(*encoding, key, self);
            },
            "key"_a
         )
         .def(
            "__getattr__",
            [](nb::handle self, const std::string& key) -> nb::object {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.__getattr__ called with invalid instance"
               );
               if(batch_encoding_has_graph_field(*encoding, key)) {
                  return batch_encoding_get_graph_field(*encoding, key, self);
               }
               const std::string message = "'BatchEncoding' object has no attribute '" + key + "'";
               PyErr_SetString(PyExc_AttributeError, message.c_str());
               throw nb::python_error();
            }
         )
         .def(
            "__setattr__",
            [](nb::handle self, const std::string& key, nb::handle value) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.__setattr__ called with invalid instance"
               );
               if(batch_encoding_has_graph_field(*encoding, key)) {
                  if(is_native_graph_field_ptr_key(*encoding, key)) {
                     throw std::invalid_argument(
                        "Direct assignment to ragged ptr key '" + key
                        + "' is not supported; assign the base field as (values, ptr)"
                     );
                  }
                  set_batch_encoding_graph_field(*encoding, key, value);
                  clear_owner_tensor_cache(self);
                  return;
               }
               set_python_attribute(self, key, value);
            }
         )
         .def(
            "__repr__",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.__repr__ called with invalid instance"
               );
               return batch_encoding_repr(self, *encoding);
            }
         )
         .def(
            "__str__",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.__str__ called with invalid instance"
               );
               return batch_encoding_str(self, *encoding);
            }
         )
         .def(
            "keys",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.keys called with invalid instance"
               );
               auto key_set = batch_encoding_native_graph_field_keys(*encoding);
               nb::dict attrs = batch_encoding_python_attrs(self);
               for(auto [key_obj, value_obj] : attrs) {
                  (void) value_obj;
                  const std::string key = py_string(key_obj);
                  if(is_reserved_python_attr_key(key) or key_set.contains(key)) {
                     continue;
                  }
                  key_set.insert(key);
               }
               nb::list out;
               for(const auto& key : key_set) {
                  out.append(key);
               }
               return out;
            }
         )
         .def(
            "items",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.items called with invalid instance"
               );
               auto key_set = batch_encoding_native_graph_field_keys(*encoding);
               nb::dict attrs = batch_encoding_python_attrs(self);
               for(auto [key_obj, value_obj] : attrs) {
                  (void) value_obj;
                  const std::string key = py_string(key_obj);
                  if(is_reserved_python_attr_key(key) or key_set.contains(key)) {
                     continue;
                  }
                  key_set.insert(key);
               }

               nb::list out;
               for(const auto& key : key_set) {
                  nb::object value;
                  if(batch_encoding_has_graph_field(*encoding, key)) {
                     value = batch_encoding_get_graph_field(*encoding, key, self);
                  } else {
                     value = nb::borrow< nb::object >(attrs[key.c_str()]);
                  }
                  out.append(nb::make_tuple(key, std::move(value)));
               }
               return out;
            }
         )
         .def(
            "as_pyg",
            [](nb::handle self, std::optional< bool > as_batch, bool include_python_attrs) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.as_pyg called with invalid instance"
               );
               nb::object out = batch_encoding_as_pyg(*encoding, as_batch);
               if(include_python_attrs) {
                  copy_python_attrs_to_object(self, out, as_batch, *encoding);
               }
               return out;
            },
            "as_batch"_a = nb::none(),
            "include_python_attrs"_a = true
         )
         .def(
            "as_hetero",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.as_hetero called with invalid instance"
               );
               if(encoding->graph_kind != "hetero") {
                  throw std::invalid_argument(
                     "BatchEncoding graph_kind mismatch: expected 'hetero'"
                  );
               }
               return HeteroBatchEncodingView(nb::borrow< nb::object >(self));
            }
         )
         .def(
            "as_homo",
            [](nb::handle self) {
               auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
                  self, "BatchEncoding.as_homo called with invalid instance"
               );
               if(encoding->graph_kind != "homo") {
                  throw std::invalid_argument("BatchEncoding graph_kind mismatch: expected 'homo'");
               }
               if(encoding->schema.node_types.size() > 1
                  or encoding->schema.edge_types.size() > 1) {
                  throw std::invalid_argument(
                     "BatchEncoding.as_homo() expects schema with at most one node type and one "
                     "edge type"
                  );
               }
               return HomoBatchEncodingView(nb::borrow< nb::object >(self));
            }
         )
         .def("schema_fingerprint", &schema_fingerprint)
         .def(
            "save",
            [](nb::handle self, const std::string& path, bool include_metadata) {
               nb::object file = builtins_open_handle()(path, "wb");
               nb::dict state = batch_encoding_state_from_instance(self, include_metadata);
               auto payload = pickle_dumps_handle()(state, 5);
               file.attr("write")(payload);
               file.attr("close")();
            },
            "path"_a,
            "include_metadata"_a = false
         )
         .def_static(
            "load",
            [](const std::string& path) {
               nb::object file = builtins_open_handle()(path, "rb");
               nb::bytes payload = nb::cast< nb::bytes >(file.attr("read")());
               nb::dict state = nb::cast< nb::dict >(pickle_loads_handle()(payload));
               file.attr("close")();
               return batch_encoding_object_from_state(state);
            }
         )
         .def(
            "dumps",
            [](nb::handle self, bool include_metadata) {
               nb::dict state = batch_encoding_state_from_instance(self, include_metadata);
               return nb::cast< nb::bytes >(pickle_dumps_handle()(state, 5));
            },
            "include_metadata"_a = true
         )
         .def_static(
            "loads",
            [](nb::bytes payload) {
               nb::dict state = nb::cast< nb::dict >(pickle_loads_handle()(payload));
               return batch_encoding_object_from_state(state);
            },
            "payload"_a
         )
         .def(
            "__getstate__",
            [](nb::handle self) { return batch_encoding_state_from_instance(self, true); }
         )
         .def(
            "__reduce__",
            [](nb::handle self) {
               nb::bytes payload = nb::cast< nb::bytes >(self.attr("dumps")(true));
               return nb::make_tuple(
                  mifrost_batch_encoding_loader_handle(), nb::make_tuple(std::move(payload))
               );
            }
         )
         .def(
            "__reduce_ex__",
            [](nb::handle self, int) {
               nb::bytes payload = nb::cast< nb::bytes >(self.attr("dumps")(true));
               return nb::make_tuple(
                  mifrost_batch_encoding_loader_handle(), nb::make_tuple(std::move(payload))
               );
            }
         )
         .def("__setstate__", [](nb::handle self, const nb::dict& state) {
            auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
               self, "BatchEncoding.__setstate__ called with invalid instance"
            );
            *encoding = batch_encoding_from_state_dict(state);
            batch_encoding_clear_python_attrs(self);
            batch_encoding_apply_python_attrs_from_state(self, state);
            clear_owner_tensor_cache(self);
         });

   batch_builder_cls.attr("__mifrost_map_view_methods__") = nb::make_tuple(
      "schema_flags_view", "node_feature_dims_view"
   );
   batch_encoding_cls.attr("__mifrost_map_view_methods__") = nb::make_tuple(
      "schema_flags_view", "node_feature_dims_view"
   );

   m.def(
      "batch_encodings",
      [](nb::iterable encodings_obj, nb::object field_specs_obj) {
         std::vector< const BatchBuilder::BatchEncoding* > encodings;
         std::vector< nb::object > source_objects;
         for(nb::handle item : encodings_obj) {
            nb::object source = nb::borrow< nb::object >(item);
            auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
               source, "batch_encodings expects BatchEncoding inputs"
            );
            encodings.push_back(encoding);
            source_objects.push_back(std::move(source));
         }

         if(encodings.empty()) {
            return nb::cast(BatchBuilder::BatchEncoding{});
         }

         const auto expected_fp = schema_fingerprint(*encodings.front());
         BatchBuilder builder;
         builder.set_graph_kind(encodings.front()->graph_kind);
         for(const auto* encoding : encodings) {
            if(encoding->num_graphs != 1) {
               throw std::invalid_argument("batch_encodings expects inputs with num_graphs == 1");
            }
            validate_batch_encoding_graph_fields(*encoding, "batch_encodings input validation");
            if(schema_fingerprint(*encoding) != expected_fp) {
               throw std::invalid_argument("batch_encodings schema_fingerprint mismatch");
            }
            builder.append_batch_encoding(*encoding);
         }

         nb::object out = nb::cast(builder.build());
         PythonCollationInputs collation_inputs;
         try {
            collation_inputs = build_python_collation_inputs(
               source_objects, encodings, field_specs_obj
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "batch_encodings field_specs preparation failed: " + std::string(ex.what())
            );
         }
         if(collation_inputs.field_specs.empty()) {
            return out;
         }

         auto* out_encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
            out, "batch_encodings failed to materialize BatchEncoding output instance"
         );
         const auto reserved_native_keys = batch_encoding_native_graph_field_keys(*out_encoding);
         for(const auto& [key, spec] : collation_inputs.field_specs) {
            if(reserved_native_keys.contains(key)) {
               throw std::invalid_argument(
                  "Python field spec key '" + key
                  + "' collides with native field key during batch_encodings"
               );
            }
            if(spec.mode == GraphFieldMode::RAGGED_CAT) {
               const std::string ptr_key = key + "_ptr";
               if(reserved_native_keys.contains(ptr_key)) {
                  throw std::invalid_argument(
                     "Python field spec key '" + key + "' collides with native field ptr key '"
                     + ptr_key + "' during batch_encodings"
                  );
               }
            }
         }
         auto filtered_specs = filter_python_field_specs_for_native_collisions(
            collation_inputs.field_specs, reserved_native_keys
         );
         if(filtered_specs.empty()) {
            return out;
         }

         try {
            apply_python_collation_to_output(
               out, filtered_specs, collation_inputs.source_attrs, collation_inputs.source_encodings
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "batch_encodings python collation failed: " + std::string(ex.what())
            );
         }

         try {
            register_batch_encoding_field_specs(out, python_field_specs_to_dict(filtered_specs));
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "batch_encodings field_specs registration failed: " + std::string(ex.what())
            );
         }
         return out;
      },
      "encodings"_a,
      "field_specs"_a = nb::none()
   );
}

}  // namespace mifrost
