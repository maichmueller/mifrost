#include <absl/container/btree_map.h>
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
#include <string_view>
#include <type_traits>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/default_relations.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/horizon_hgraph_encoder.hpp"
#include "mifrost/core/map_view.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/core/successor_hgraph_encoder.hpp"
#include "mifrost/core/transition_dag.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

namespace {

constexpr std::string_view kPythonGraphFieldSpecsAttr = "__mifrost_graph_field_specs__";

void apply_hgraph_config_kwargs(HGraphEncoderEngine::Config& config, const nb::kwargs& kwargs)
{
   apply_config_kwargs(config, kwargs, "HGraphEncoderConfig");
}

GraphFieldSpec graph_field_spec_from_dict(const nb::dict& spec_dict)
{
   GraphFieldSpec spec;
   if(not spec_dict.contains("dtype")) {
      throw std::invalid_argument("graph field spec requires 'dtype'");
   }
   if(not spec_dict.contains("mode")) {
      throw std::invalid_argument("graph field spec requires 'mode'");
   }
   const auto dtype = nb::cast< nb::str >(spec_dict["dtype"]);
   const auto mode = nb::cast< nb::str >(spec_dict["mode"]);
   spec.dtype = graph_field_dtype_from_name(dtype.c_str());
   spec.mode = graph_field_mode_from_name(mode.c_str());
   if(spec_dict.contains("dim")) {
      spec.dim = nb::cast< int >(spec_dict["dim"]);
   }
   if(spec_dict.contains("cat_dim")) {
      spec.cat_dim = normalize_graph_field_cat_dim(nb::cast< int >(spec_dict["cat_dim"]));
   }
   if(spec_dict.contains("inc")) {
      const auto inc_dict = nb::cast< nb::dict >(spec_dict["inc"]);
      if(inc_dict.contains("kind")) {
         const auto kind = nb::cast< nb::str >(inc_dict["kind"]);
         spec.inc.kind = graph_field_inc_kind_from_name(kind.c_str());
      }
      if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
         if(not inc_dict.contains("node_type")) {
            throw std::invalid_argument("graph field spec inc NODE_OFFSET requires node_type");
         }
         spec.inc.node_type = nb::cast< std::string >(inc_dict["node_type"]);
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
      const std::string key = nb::cast< std::string >(key_obj);
      const auto entry = nb::cast< nb::dict >(field_obj);
      GraphField field;
      field.spec = graph_field_spec_from_dict(nb::cast< nb::dict >(entry["spec"]));
      field.ptr = nb::cast< std::vector< int64_t > >(entry["ptr"]);

      const std::string dtype = nb::cast< std::string >(entry["dtype"]);
      const size_t length = static_cast< size_t >(nb::cast< int64_t >(entry["length"]));
      const nb::bytes raw_bytes = nb::cast< nb::bytes >(entry["raw"]);
      const std::string raw(raw_bytes.c_str(), raw_bytes.size());
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
      return nb::isinstance< nb::str >(handle) || nb::isinstance< nb::bytes >(handle);
   };

   if(nb::isinstance< nb::bool_ >(value) || nb::isinstance< nb::int_ >(value)
      || nb::isinstance< nb::float_ >(value)) {
      out.values.push_back(nb::cast< T >(value));
      out.ndim = 0;
      out.rows = 1;
      out.cols = 1;
      return out;
   }

   if(not nb::isinstance< nb::iterable >(value) || is_string_like(value)) {
      throw std::invalid_argument("Graph field value must be a scalar or iterable");
   }

   bool has_nested = false;
   bool has_scalar = false;
   bool nested_cols_set = false;
   size_t nested_cols = 0;
   size_t nested_rows = 0;

   nb::object iterable_obj = nb::borrow< nb::object >(value);
   for(nb::handle item : iterable_obj) {
      if(nb::isinstance< nb::iterable >(item) && ! is_string_like(item)) {
         has_nested = true;
         size_t row_size = 0;
         nb::object nested_obj = nb::borrow< nb::object >(item);
         for(nb::handle nested : nested_obj) {
            out.values.push_back(nb::cast< T >(nested));
            row_size++;
         }
         if(! nested_cols_set) {
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
   if(has_nested && has_scalar) {
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
                               || spec.mode == GraphFieldMode::RAGGED_CAT;
   if(is_concat_mode && spec.dim > 1) {
      if(cat_dim == 0) {
         if(input.ndim == 2 && input.cols != static_cast< size_t >(spec.dim)) {
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
void vector_owner_deleter(void* p) noexcept
{
   delete static_cast< std::vector< T >* >(p);
}

template < typename T >
auto vector_to_1d_ndarray_view(std::vector< T >& vec, nb::handle owner)
{
   size_t shape[1] = {vec.size()};
   return nb::ndarray< nb::numpy, T, nb::shape< -1 > >(vec.data(), 1, shape, owner);
}

template < typename T >
auto vector_to_2d_ndarray_view(std::vector< T >& vec, size_t rows, size_t cols, nb::handle owner)
{
   size_t shape[2] = {rows, cols};
   return nb::ndarray< nb::numpy, T, nb::shape< -1, -1 > >(vec.data(), 2, shape, owner);
}

template < typename T >
auto vector_to_1d_ndarray_owned(std::vector< T >&& vec)
{
   auto* heap_vec = new std::vector< T >(std::move(vec));
   heap_vec->shrink_to_fit();
   size_t shape[1] = {heap_vec->size()};
   nb::capsule owner(heap_vec, vector_owner_deleter< T >);
   return nb::ndarray< nb::numpy, T, nb::shape< -1 > >(heap_vec->data(), 1, shape, owner);
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
   if(! nb::isinstance< nb::list >(value)) {
      return nb::borrow< nb::object >(value);
   }
   nb::list outer = nb::borrow< nb::list >(value);
   if(nb::len(outer) == 1 && nb::isinstance< nb::list >(outer[0])) {
      return nb::borrow< nb::object >(outer[0]);
   }
   return nb::borrow< nb::object >(value);
}

void copy_store_attrs_without_batch(nb::object& dst_store, nb::object& src_store)
{
   for(auto key_obj : src_store.attr("keys")()) {
      const std::string key = nb::cast< std::string >(key_obj);
      if(key == "ptr" || key == "batch") {
         continue;
      }
      dst_store.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
   }
}

void copy_global_attrs_for_single(nb::object& dst, nb::object& src)
{
   nb::object global_store = src.attr("_global_store");
   for(auto key_obj : global_store.attr("keys")()) {
      const std::string key = nb::cast< std::string >(key_obj);
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
   nb::object tg_data = nb::module_::import_("torch_geometric.data");
   nb::object out = tg_data.attr("HeteroData")();

   for(auto node_type_obj : pyg_batch.attr("node_types")) {
      std::string node_type = nb::cast< std::string >(node_type_obj);
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
   nb::object tg_data = nb::module_::import_("torch_geometric.data");
   nb::object out = tg_data.attr("Data")();

   nb::list node_types = nb::cast< nb::list >(pyg_batch.attr("node_types"));
   if(nb::len(node_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=False) for homo expects a single node type"
      );
   }

   if(nb::len(node_types) == 1) {
      std::string node_type = nb::cast< std::string >(node_types[0]);
      nb::object src_store = pyg_batch.attr("__getitem__")(node_type);
      for(auto key_obj : src_store.attr("keys")()) {
         const std::string key = nb::cast< std::string >(key_obj);
         if(key == "ptr" || key == "batch") {
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
      out.emplace(nb::cast< std::string >(key_obj), nb::cast< value_type >(value_obj));
   }
   return out;
};

BatchBuilder::BatchEncoding batch_encoding_from_state_dict(const nb::dict& state)
{
   if(const int version = nb::cast< int >(state["format_version"]); version != 1) {
      throw std::invalid_argument("Unsupported BatchEncoding format version");
   }

   BatchBuilder::BatchEncoding encoding;
   encoding.graph_kind = nb::cast< std::string >(state["graph_kind"]);
   encoding.num_graphs = nb::cast< int64_t >(state["num_graphs"]);
   encoding.schema_flags = map_from_dict< bool >(nb::cast< nb::dict >(state["schema_flags"]));
   encoding.node_feature_dims = nb::cast< hash_map< std::string, int > >(
      state["node_feature_dims"]
   );
   encoding.graph_attrs = nb::cast< hash_map< std::string, BatchBuilder::GraphAttrValue > >(
      state["graph_attrs"]
   );
   if(state.contains("graph_fields")) {
      encoding.graph_fields = graph_field_map_from_dict(
         nb::cast< nb::dict >(state["graph_fields"])
      );
   }
   encoding.ptrs = nb::cast< hash_map< std::string, std::vector< int64_t > > >(state["ptrs"]);
   encoding.node_counts = map_from_dict< int64_t >(nb::cast< nb::dict >(state["node_counts"]));
   encoding.schema = Schema::from_dict(nb::cast< nb::dict >(state["schema"]));
   encoding.node_names = nb::cast< hash_map< std::string, std::vector< std::string > > >(
      state["node_names"]
   );
   encoding.object_names = nb::cast< std::vector< std::string > >(state["object_names"]);

   nb::dict columns = nb::cast< nb::dict >(state["columns"]);
   for(auto [key_obj, col_obj] : columns) {
      auto col = nb::cast< nb::dict >(col_obj);
      const auto key = nb::cast< std::string >(key_obj);
      const auto dim = nb::cast< int >(col["dim"]);
      const auto dtype = nb::cast< std::string >(col["dtype"]);
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

nb::dict batch_encoding_python_attrs(nb::handle self)
{
   return nb::cast< nb::dict >(self.attr("__dict__"));
}

nb::dict batch_encoding_python_attrs_copy(nb::handle self)
{
   nb::dict out;
   nb::dict attrs = batch_encoding_python_attrs(self);
   for(auto [key_obj, value_obj] : attrs) {
      out[key_obj] = nb::borrow< nb::object >(value_obj);
   }
   return out;
}

bool is_reserved_python_attr_key(std::string_view key)
{
   return key == kPythonGraphFieldSpecsAttr;
}

void batch_encoding_clear_python_attrs(nb::handle self) {}

nb::dict batch_encoding_graph_field_specs(nb::handle self)
{
   nb::dict attrs = nb::cast< nb::dict >(self.attr("__dict__"));
   if(! attrs.contains(kPythonGraphFieldSpecsAttr.data())) {
      return nb::dict();
   }
   nb::object raw_specs = nb::borrow< nb::object >(attrs[kPythonGraphFieldSpecsAttr.data()]);
   if(! nb::isinstance< nb::dict >(raw_specs)) {
      throw std::invalid_argument("BatchEncoding internal graph field specs must be a dict");
   }
   return nb::cast< nb::dict >(raw_specs);
}

void batch_encoding_apply_python_attrs_from_state(
   nb::handle self,
   const nb::dict& state,
   nb::dict& dst
)
{
   if(! state.contains("python_attrs")) {
      return;
   }
   auto src = nb::cast< nb::dict >(state["python_attrs"]);
   for(auto [key_obj, value_obj] : src) {
      dst[key_obj] = nb::borrow< nb::object >(value_obj);
   }
}
void batch_encoding_apply_python_attrs_from_state(nb::handle self, const nb::dict& state)
{
   if(! state.contains("python_attrs")) {
      return;
   }
   auto dst = nb::cast< nb::dict >(self.attr("__dict__"));
   auto src = nb::cast< nb::dict >(state["python_attrs"]);
   for(auto [key_obj, value_obj] : src) {
      dst[key_obj] = nb::borrow< nb::object >(value_obj);
   }
}

nb::dict batch_encoding_state_from_instance(nb::handle self, bool include_metadata)
{
   auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
   if(encoding == nullptr) {
      throw std::invalid_argument("BatchEncoding state extraction called with invalid instance");
   }
   nb::dict state = batch_encoding_to_state_dict(*encoding, include_metadata);
   nb::dict py_attrs = batch_encoding_python_attrs_copy(self);
   if(nb::len(py_attrs) > 0) {
      state["python_attrs"] = std::move(py_attrs);
   }
   return state;
}

nb::object batch_encoding_object_from_state(const nb::dict& state)
{
   nb::object cls = nb::module_::import_("mifrost._core").attr("BatchEncoding");
   nb::object obj = cls();
   auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(obj);
   if(encoding == nullptr) {
      throw std::invalid_argument("Failed to instantiate BatchEncoding during state load");
   }
   *encoding = batch_encoding_from_state_dict(state);
   auto attrs = nb::cast< nb::dict >(obj.attr("__dict__"));
   attrs.clear();
   batch_encoding_apply_python_attrs_from_state(obj, state, attrs);
   return obj;
}

std::string_view canonical_python_field_mode(std::string_view mode)
{
   if(ascii_iequals(mode, "stack")) {
      return "stack";
   }
   if(ascii_iequals(mode, "ragged_cat")) {
      return "ragged_cat";
   }
   if(ascii_iequals(mode, "const")) {
      return "const";
   }
   throw std::invalid_argument("Unsupported Python field mode: " + std::string(mode));
}

bool is_python_graph_field_dtype(std::string_view dtype)
{
   return ascii_iequals(dtype, "pyobj") || ascii_iequals(dtype, "python")
          || ascii_iequals(dtype, "object");
}

enum class PythonFieldMode { STACK, RAGGED_CAT, CONST };

PythonFieldMode python_field_mode_from_name(std::string_view mode)
{
   const auto canonical = canonical_python_field_mode(mode);
   if(canonical == "stack") {
      return PythonFieldMode::STACK;
   }
   if(canonical == "ragged_cat") {
      return PythonFieldMode::RAGGED_CAT;
   }
   return PythonFieldMode::CONST;
}

std::string_view python_field_mode_name(PythonFieldMode mode)
{
   switch(mode) {
      case PythonFieldMode::STACK: return "stack";
      case PythonFieldMode::RAGGED_CAT: return "ragged_cat";
      case PythonFieldMode::CONST: return "const";
   }
   throw std::logic_error("Unknown PythonFieldMode");
}

PythonFieldMode python_field_mode_from_spec(nb::handle spec_obj)
{
   std::string_view mode = "stack";
   std::optional< nb::str > mode_storage;
   if(spec_obj.is_none()) {
      mode = "stack";
   } else if(nb::isinstance< nb::str >(spec_obj)) {
      mode_storage = nb::cast< nb::str >(spec_obj);
      mode = mode_storage->c_str();
   } else if(nb::isinstance< nb::dict >(spec_obj)) {
      auto spec = nb::cast< nb::dict >(spec_obj);
      if(spec.contains("dtype")) {
         const auto dtype_storage = nb::cast< nb::str >(spec["dtype"]);
         const std::string_view dtype = dtype_storage.c_str();
         if(! is_python_graph_field_dtype(dtype)) {
            throw std::invalid_argument(
               "Python graph field spec dtype must be one of {pyobj, python, object}"
            );
         }
      }
      if(spec.contains("mode")) {
         mode_storage = nb::cast< nb::str >(spec["mode"]);
         mode = mode_storage->c_str();
      }
   } else {
      throw std::invalid_argument("Python field spec must be a string or dict containing 'mode'");
   }

   return python_field_mode_from_name(mode);
}

using PythonFieldSpecMap = absl::btree_map< std::string, PythonFieldMode >;

PythonFieldSpecMap canonicalize_python_graph_field_specs(const nb::dict& specs)
{
   PythonFieldSpecMap out;
   for(auto [key_obj, spec_obj] : specs) {
      const auto key = nb::cast< std::string >(key_obj);
      if(key.empty()) {
         throw std::invalid_argument("Python graph field spec keys must be non-empty");
      }
      out[key] = python_field_mode_from_spec(spec_obj);
   }
   return out;
}

void merge_python_graph_field_specs(PythonFieldSpecMap& dst, const PythonFieldSpecMap& src)
{
   for(const auto& [key, incoming_mode] : src) {
      if(const auto it = dst.find(key); it != dst.end()) {
         if(it->second != incoming_mode) {
            throw std::invalid_argument(
               "Conflicting Python graph field mode for key '" + key + "'"
            );
         }
      } else {
         dst.emplace(key, incoming_mode);
      }
   }
}

nb::dict python_graph_field_specs_to_dict(const PythonFieldSpecMap& specs)
{
   nb::dict out;
   for(const auto& [key, mode] : specs) {
      nb::dict normalized;
      normalized["dtype"] = "pyobj";
      normalized["mode"] = std::string(python_field_mode_name(mode));
      out[key.c_str()] = std::move(normalized);
   }
   return out;
}

bool python_objects_equal_for_const(const nb::object& lhs, const nb::object& rhs);

bool try_get_python_attr(const nb::dict& attrs, nb::handle key_obj, nb::object& out)
{
   if(! attrs.contains(key_obj)) {
      return false;
   }
   out = nb::borrow< nb::object >(attrs[key_obj]);
   return true;
}

nb::list
collate_python_stack_values(const std::vector< nb::dict >& source_attrs, nb::handle key_obj)
{
   nb::list values;
   for(const auto& attrs : source_attrs) {
      nb::object value;
      if(try_get_python_attr(attrs, key_obj, value)) {
         values.append(value);
      } else {
         values.append(nb::none());
      }
   }
   return values;
}

struct PythonRaggedCollation {
   nb::list values;
   nb::list ptr;
};

PythonRaggedCollation
collate_python_ragged_values(const std::vector< nb::dict >& source_attrs, nb::handle key_obj)
{
   PythonRaggedCollation out;
   int64_t offset = 0;
   out.ptr.append(offset);
   for(const auto& attrs : source_attrs) {
      nb::object value;
      if(try_get_python_attr(attrs, key_obj, value)) {
         if(nb::isinstance< nb::list >(value) || nb::isinstance< nb::tuple >(value)) {
            for(nb::handle entry : value) {
               out.values.append(nb::borrow< nb::object >(entry));
               offset += 1;
            }
         } else if(! value.is_none()) {
            out.values.append(value);
            offset += 1;
         }
      }
      out.ptr.append(offset);
   }
   return out;
}

nb::object collate_python_const_value(
   const std::string& key,
   const std::vector< nb::dict >& source_attrs,
   nb::handle key_obj
)
{
   nb::object first = nb::none();
   bool found = false;
   for(size_t source_idx = 0; source_idx < source_attrs.size(); ++source_idx) {
      const auto& attrs = source_attrs[source_idx];
      nb::object value;
      if(! try_get_python_attr(attrs, key_obj, value)) {
         throw std::invalid_argument(
            "Python const field '" + key + "' missing value for encoding index "
            + std::to_string(source_idx)
         );
      }
      if(! found) {
         first = value;
         found = true;
         continue;
      }
      if(! python_objects_equal_for_const(value, first)) {
         throw std::invalid_argument(
            "Python const field '" + key + "' has non-constant values across encodings"
         );
      }
   }
   if(! found) {
      throw std::invalid_argument(
         "Python const field '" + key + "' requires at least one encoding value"
      );
   }
   return first;
}

void seed_default_python_field_specs_from_attrs(
   PythonFieldSpecMap& field_specs,
   const std::vector< nb::dict >& source_attrs
)
{
   for(const auto& attrs : source_attrs) {
      for(auto [key_obj, value_obj] : attrs) {
         (void) value_obj;
         const std::string key = nb::cast< std::string >(key_obj);
         if(is_reserved_python_attr_key(key)) {
            continue;
         }
         field_specs.try_emplace(key, PythonFieldMode::STACK);
      }
   }
}

bool has_non_reserved_python_attrs(const nb::dict& attrs)
{
   for(auto [key_obj, value_obj] : attrs) {
      (void) value_obj;
      const std::string key = nb::cast< std::string >(key_obj);
      if(! is_reserved_python_attr_key(key)) {
         return true;
      }
   }
   return false;
}

void register_batch_encoding_graph_field_specs(nb::handle self, const nb::dict& specs)
{
   auto normalized = canonicalize_python_graph_field_specs(specs);
   auto existing = canonicalize_python_graph_field_specs(batch_encoding_graph_field_specs(self));
   merge_python_graph_field_specs(existing, normalized);
   nb::dict attrs = nb::cast< nb::dict >(self.attr("__dict__"));
   attrs[kPythonGraphFieldSpecsAttr.data()] = python_graph_field_specs_to_dict(existing);
}

void copy_python_attrs_to_object(
   nb::handle src,
   nb::handle dst,
   std::optional< bool > as_batch,
   int64_t num_graphs
)
{
   const bool want_batch = as_batch.value_or(num_graphs != 1);
   nb::dict attrs = batch_encoding_python_attrs(src);
   for(auto [key_obj, value_obj] : attrs) {
      const std::string key = nb::cast< std::string >(key_obj);
      if(is_reserved_python_attr_key(key)) {
         continue;
      }
      nb::object value = nb::borrow< nb::object >(value_obj);
      if(! want_batch && num_graphs == 1 && nb::isinstance< nb::list >(value)
         && nb::len(value) == 1) {
         dst.attr("__setattr__")(key.c_str(), value.attr("__getitem__")(0));
      } else {
         dst.attr("__setattr__")(key.c_str(), value);
      }
   }
}

bool try_cast_python_bool(nb::handle value, bool& out)
{
   try {
      out = nb::cast< bool >(value);
      return true;
   } catch(...) {
      return false;
   }
}

nb::object try_import_module(const char* module_name)
{
   try {
      return nb::module_::import_(module_name);
   } catch(...) {
      return nb::none();
   }
}

const nb::object& torch_module()
{
   static const nb::object module = try_import_module("torch");
   return module;
}

const nb::object& numpy_module()
{
   static const nb::object module = try_import_module("numpy");
   return module;
}

const nb::object& torch_tensor_type()
{
   static const nb::object type = []() -> nb::object {
      const nb::object& torch = torch_module();
      if(torch.is_none()) {
         return nb::none();
      }
      return torch.attr("Tensor");
   }();
   return type;
}

const nb::object& numpy_array_type()
{
   static const nb::object type = []() -> nb::object {
      const nb::object& np = numpy_module();
      if(np.is_none()) {
         return nb::none();
      }
      return np.attr("ndarray");
   }();
   return type;
}

const nb::object& operator_module()
{
   static const nb::object module = nb::module_::import_("operator");
   return module;
}

const nb::object& operator_eq_fn()
{
   static const nb::object eq_fn = operator_module().attr("eq");
   return eq_fn;
}

const nb::object& torch_equal_fn()
{
   static const nb::object fn = []() -> nb::object {
      const nb::object& torch = torch_module();
      if(torch.is_none()) {
         return nb::none();
      }
      return torch.attr("equal");
   }();
   return fn;
}

const nb::object& numpy_array_equal_fn()
{
   static const nb::object fn = []() -> nb::object {
      const nb::object& np = numpy_module();
      if(np.is_none()) {
         return nb::none();
      }
      return np.attr("array_equal");
   }();
   return fn;
}

bool is_torch_tensor(nb::handle value)
{
   const nb::object& type = torch_tensor_type();
   if(type.is_none()) {
      return false;
   }
   return nb::isinstance(value, type);
}

bool is_numpy_array(nb::handle value)
{
   const nb::object& type = numpy_array_type();
   if(type.is_none()) {
      return false;
   }
   return nb::isinstance(value, type);
}

bool python_eq_returns_true(nb::handle lhs, nb::handle rhs)
{
   bool result = false;
   if(! try_cast_python_bool(operator_eq_fn()(lhs, rhs), result)) {
      throw std::invalid_argument("Python const field comparison produced a non-boolean result");
   }
   return result;
}

bool torch_tensors_equal_exact(const nb::object& lhs, const nb::object& rhs)
{
   if(! python_eq_returns_true(lhs.attr("dtype"), rhs.attr("dtype"))) {
      return false;
   }
   if(! python_eq_returns_true(lhs.attr("shape"), rhs.attr("shape"))) {
      return false;
   }
   if(! python_eq_returns_true(lhs.attr("stride")(), rhs.attr("stride")())) {
      return false;
   }
   if(! python_eq_returns_true(lhs.attr("device"), rhs.attr("device"))) {
      return false;
   }
   if(! python_eq_returns_true(lhs.attr("layout"), rhs.attr("layout"))) {
      return false;
   }

   const nb::object& equal = torch_equal_fn();
   if(equal.is_none()) {
      throw std::invalid_argument("Torch tensor comparison requested but torch is unavailable");
   }
   return nb::cast< bool >(equal(lhs, rhs));
}

bool numpy_arrays_equal_exact(const nb::object& lhs, const nb::object& rhs)
{
   if(! python_eq_returns_true(lhs.attr("dtype"), rhs.attr("dtype"))) {
      return false;
   }
   if(! python_eq_returns_true(lhs.attr("shape"), rhs.attr("shape"))) {
      return false;
   }
   if(! python_eq_returns_true(lhs.attr("strides"), rhs.attr("strides"))) {
      return false;
   }

   const nb::object& array_equal = numpy_array_equal_fn();
   if(array_equal.is_none()) {
      throw std::invalid_argument("NumPy array comparison requested but numpy is unavailable");
   }
   return nb::cast< bool >(array_equal(lhs, rhs));
}

bool python_objects_equal_for_const(const nb::object& lhs, const nb::object& rhs)
{
   if(lhs.ptr() == rhs.ptr()) {
      return true;
   }
   if(lhs.is_none() || rhs.is_none()) {
      return lhs.is_none() && rhs.is_none();
   }

   const bool lhs_is_torch = is_torch_tensor(lhs);
   const bool rhs_is_torch = is_torch_tensor(rhs);
   if(lhs_is_torch || rhs_is_torch) {
      if(! (lhs_is_torch && rhs_is_torch)) {
         return false;
      }
      return torch_tensors_equal_exact(lhs, rhs);
   }

   const bool lhs_is_numpy = is_numpy_array(lhs);
   const bool rhs_is_numpy = is_numpy_array(rhs);
   if(lhs_is_numpy || rhs_is_numpy) {
      if(! (lhs_is_numpy && rhs_is_numpy)) {
         return false;
      }
      return numpy_arrays_equal_exact(lhs, rhs);
   }

   return python_eq_returns_true(lhs, rhs);
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
               tensors[key.c_str()] = vector_to_1d_ndarray_view(data, owner);
               return;
            }
            const size_t rows = col.dim > 0 ? data.size() / static_cast< size_t >(col.dim) : 0;
            tensors[key.c_str()] = vector_to_2d_ndarray_view(
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
      tensors[(node_type + "/ptr").c_str()] = vector_to_1d_ndarray_view(ptr, owner);
      tensors[(node_type + "/batch").c_str()] = vector_to_1d_ndarray_owned(ptr_to_batch(ptr));
   }
   if(! exported_ptr) {
      for(const auto& [node_type, count] : encoding.node_counts) {
         if(count <= 0) {
            continue;
         }
         std::vector< int64_t > ptr{0, count};
         tensors[(node_type + "/ptr").c_str()] = vector_to_1d_ndarray_owned(std::move(ptr));
         tensors[(node_type + "/batch").c_str()] = vector_to_1d_ndarray_owned(
            std::vector< int64_t >(count, 0)
         );
      }
   }

   for(auto& [attr, field] : encoding.graph_fields) {
      const std::string key = "__graph__/" + attr;
      std::visit(
         [&]< typename T >(std::vector< T >& data) {
            if(field.spec.dim == 1) {
               tensors[key.c_str()] = vector_to_1d_ndarray_view(data, owner);
               return;
            }
            const bool cat_dim_one = (field.spec.mode == GraphFieldMode::CAT
                                      || field.spec.mode == GraphFieldMode::RAGGED_CAT)
                                     && graph_field_cat_dim_is_one(field.spec.cat_dim);
            const size_t rows = cat_dim_one ? static_cast< size_t >(field.spec.dim)
                                            : data.size() / static_cast< size_t >(field.spec.dim);
            const size_t cols = cat_dim_one ? data.size() / static_cast< size_t >(field.spec.dim)
                                            : static_cast< size_t >(field.spec.dim);
            tensors[key.c_str()] = vector_to_2d_ndarray_view(data, rows, cols, owner);
         },
         field.values
      );
      if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
         tensors[(key + "/ptr").c_str()] = vector_to_1d_ndarray_view(field.ptr, owner);
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

   if(! encoding.graph_attrs.empty()) {
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
   const bool want_batch = as_batch.value_or(encoding.num_graphs != 1);
   BatchBuilder builder;
   builder.set_graph_kind(encoding.graph_kind);
   builder.load_from_batch_encoding(encoding);
   nb::object pyg_batch = builder.build_pyg();

   if(! want_batch && encoding.num_graphs != 1) {
      throw std::invalid_argument("BatchEncoding.as_pyg(as_batch=False) requires num_graphs == 1");
   }

   if(! want_batch) {
      if(encoding.graph_kind == "homo") {
         return batch_to_single_homo_data(pyg_batch);
      }
      return batch_to_single_hetero_data(pyg_batch);
   }

   return pyg_batch;
}

}  // namespace

void init_hgraph_encoder(nb::module_& m)
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
               if(src.ndim() != 1 || dst.ndim() != 1) {
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
               auto* builder = nb::inst_ptr< BatchBuilder >(self);
               if(builder == nullptr) {
                  throw std::invalid_argument(
                     "BatchBuilder.schema_flags_view called with invalid instance"
                  );
               }
               return make_map_view(builder->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def(
            "node_feature_dims_view",
            [](nb::handle self) {
               auto* builder = nb::inst_ptr< BatchBuilder >(self);
               if(builder == nullptr) {
                  throw std::invalid_argument(
                     "BatchBuilder.node_feature_dims_view called with invalid instance"
                  );
               }
               return make_map_view(builder->node_feature_dims, self);
            },
            nb::rv_policy::move
         )
         .def("graph_field_keys", &BatchBuilder::graph_field_keys)
         .def(
            "graph_field_specs",
            [](const BatchBuilder& builder) {
               nb::dict out;
               for(const auto& [key, spec] : builder.graph_field_specs()) {
                  out[key.c_str()] = graph_field_spec_to_dict(spec);
               }
               return out;
            }
         )
         .def(
            "register_graph_field",
            [](BatchBuilder& builder, const std::string& key, const nb::dict& spec) {
               builder.register_graph_field(key, graph_field_spec_from_dict(spec));
            },
            "key"_a,
            "spec"_a
         )
         .def(
            "set_graph_field",
            [](BatchBuilder& builder, const std::string& key, nb::handle value) {
               const auto spec = builder.get_graph_field_spec(key);
               if(spec.dtype == GraphFieldDType::F32) {
                  auto input = coerce_numeric_values< float >(value);
                  auto values = normalize_graph_field_input(key, spec, std::move(input));
                  builder.set_graph_field(
                     key, std::span< const float >(values.data(), values.size())
                  );
               } else {
                  auto input = coerce_numeric_values< int64_t >(value);
                  auto values = normalize_graph_field_input(key, spec, std::move(input));
                  builder.set_graph_field(
                     key, std::span< const int64_t >(values.data(), values.size())
                  );
               }
            },
            "key"_a,
            "value"_a
         )
         .def(
            "set_graph_fields",
            [](BatchBuilder& builder, const nb::dict& values) {
               for(auto [key_obj, value_obj] : values) {
                  const std::string key = nb::cast< std::string >(key_obj);
                  const auto spec = builder.get_graph_field_spec(key);
                  if(spec.dtype == GraphFieldDType::F32) {
                     auto input = coerce_numeric_values< float >(value_obj);
                     auto data = normalize_graph_field_input(key, spec, std::move(input));
                     builder.set_graph_field(
                        key, std::span< const float >(data.data(), data.size())
                     );
                  } else {
                     auto input = coerce_numeric_values< int64_t >(value_obj);
                     auto data = normalize_graph_field_input(key, spec, std::move(input));
                     builder.set_graph_field(
                        key, std::span< const int64_t >(data.data(), data.size())
                     );
                  }
               }
            },
            "values"_a
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
               auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
               if(encoding == nullptr) {
                  throw std::invalid_argument(
                     "BatchEncoding.schema_flags called with invalid instance"
                  );
               }
               return make_map_view(encoding->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def_ro("node_feature_dims", &BatchBuilder::BatchEncoding::node_feature_dims)
         .def_ro("graph_attrs", &BatchBuilder::BatchEncoding::graph_attrs)
         .def(
            "schema_flags_view",
            [](nb::handle self) {
               auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
               if(encoding == nullptr) {
                  throw std::invalid_argument(
                     "BatchEncoding.schema_flags_view called with invalid instance"
                  );
               }
               return make_map_view(encoding->schema_flags, self);
            },
            nb::rv_policy::move
         )
         .def(
            "node_feature_dims_view",
            [](nb::handle self) {
               auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
               if(encoding == nullptr) {
                  throw std::invalid_argument(
                     "BatchEncoding.node_feature_dims_view called with invalid instance"
                  );
               }
               return make_map_view(encoding->node_feature_dims, self);
            },
            nb::rv_policy::move
         )
         .def(
            "as_dict",
            [](nb::handle self) {
               auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
               if(encoding == nullptr) {
                  throw std::invalid_argument("BatchEncoding.as_dict called with invalid instance");
               }
               return batch_encoding_as_dict(*encoding, self);
            }
         )
         .def(
            "register_graph_field_specs",
            [](nb::handle self, const nb::dict& specs) {
               register_batch_encoding_graph_field_specs(self, specs);
            },
            "specs"_a
         )
         .def(
            "graph_field_specs",
            [](nb::handle self) { return batch_encoding_graph_field_specs(self); }
         )
         .def(
            "as_pyg",
            [](nb::handle self, std::optional< bool > as_batch, bool include_python_attrs) {
               auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
               if(encoding == nullptr) {
                  throw std::invalid_argument("BatchEncoding.as_pyg called with invalid instance");
               }
               nb::object out = batch_encoding_as_pyg(*encoding, as_batch);
               if(include_python_attrs) {
                  copy_python_attrs_to_object(self, out, as_batch, encoding->num_graphs);
               }
               return out;
            },
            "as_batch"_a = nb::none(),
            "include_python_attrs"_a = true
         )
         .def("schema_fingerprint", &schema_fingerprint)
         .def(
            "save",
            [](nb::handle self, const std::string& path, bool include_metadata) {
               nb::object pickle = nb::module_::import_("pickle");
               nb::object builtins = nb::module_::import_("builtins");
               nb::object file = builtins.attr("open")(path, "wb");
               nb::dict state = batch_encoding_state_from_instance(self, include_metadata);
               auto payload = pickle.attr("dumps")(state, 5);
               file.attr("write")(payload);
               file.attr("close")();
            },
            "path"_a,
            "include_metadata"_a = false
         )
         .def_static(
            "load",
            [](const std::string& path) {
               nb::object builtins = nb::module_::import_("builtins");
               nb::object pickle = nb::module_::import_("pickle");
               nb::object file = builtins.attr("open")(path, "rb");
               nb::bytes payload = nb::cast< nb::bytes >(file.attr("read")());
               nb::dict state = nb::cast< nb::dict >(pickle.attr("loads")(payload));
               file.attr("close")();
               return batch_encoding_object_from_state(state);
            }
         )
         .def(
            "dumps",
            [](nb::handle self, bool include_metadata) {
               nb::object pickle = nb::module_::import_("pickle");
               nb::dict state = batch_encoding_state_from_instance(self, include_metadata);
               return nb::cast< nb::bytes >(pickle.attr("dumps")(state, 5));
            },
            "include_metadata"_a = false
         )
         .def_static(
            "loads",
            [](nb::bytes payload) {
               nb::object pickle = nb::module_::import_("pickle");
               nb::dict state = nb::cast< nb::dict >(pickle.attr("loads")(payload));
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
               nb::object loader = nb::module_::import_("mifrost").attr(
                  "_batch_encoding_from_payload"
               );
               nb::bytes payload = nb::cast< nb::bytes >(self.attr("dumps")(true));
               return nb::make_tuple(std::move(loader), nb::make_tuple(std::move(payload)));
            }
         )
         .def(
            "__reduce_ex__",
            [](nb::handle self, int) {
               nb::object loader = nb::module_::import_("mifrost").attr(
                  "_batch_encoding_from_payload"
               );
               nb::bytes payload = nb::cast< nb::bytes >(self.attr("dumps")(true));
               return nb::make_tuple(std::move(loader), nb::make_tuple(std::move(payload)));
            }
         )
         .def("__setstate__", [](nb::handle self, const nb::dict& state) {
            auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
            if(encoding == nullptr) {
               throw std::invalid_argument(
                  "BatchEncoding.__setstate__ called with invalid instance"
               );
            }
            *encoding = batch_encoding_from_state_dict(state);
            batch_encoding_clear_python_attrs(self);
            batch_encoding_apply_python_attrs_from_state(self, state);
         });

   batch_builder_cls.attr("__mifrost_map_view_methods__") = nb::make_tuple(
      "schema_flags_view", "node_feature_dims_view"
   );
   batch_encoding_cls.attr("__mifrost_map_view_methods__") = nb::make_tuple(
      "schema_flags_view", "node_feature_dims_view"
   );

   m.def(
      "batch_encodings",
      [](nb::iterable encodings_obj,
         nb::object graph_field_specs_obj,
         nb::object py_field_specs_obj) {
         if(! graph_field_specs_obj.is_none() && ! py_field_specs_obj.is_none()) {
            throw std::invalid_argument(
               "batch_encodings received both graph_field_specs and py_field_specs"
            );
         }
         if(graph_field_specs_obj.is_none()) {
            graph_field_specs_obj = std::move(py_field_specs_obj);
         }

         std::vector< const BatchBuilder::BatchEncoding* > encodings;
         std::vector< nb::object > source_objects;
         for(nb::handle item : encodings_obj) {
            nb::object source = nb::borrow< nb::object >(item);
            auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(source);
            if(encoding == nullptr) {
               throw std::invalid_argument("batch_encodings expects BatchEncoding inputs");
            }
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
            if(schema_fingerprint(*encoding) != expected_fp) {
               throw std::invalid_argument("batch_encodings schema_fingerprint mismatch");
            }
            builder.append_batch_encoding(*encoding);
         }

         nb::object out = nb::cast(builder.build());
         PythonFieldSpecMap field_specs;
         std::vector< nb::dict > source_attrs;
         source_attrs.reserve(source_objects.size());
         if(graph_field_specs_obj.is_none()) {
            bool has_any_python_attrs = false;
            for(const auto& source : source_objects) {
               nb::dict attrs = batch_encoding_python_attrs(source);
               if(attrs.contains(kPythonGraphFieldSpecsAttr.data())) {
                  auto registered_specs = canonicalize_python_graph_field_specs(
                     batch_encoding_graph_field_specs(source)
                  );
                  merge_python_graph_field_specs(field_specs, registered_specs);
               }
               if(! has_any_python_attrs && has_non_reserved_python_attrs(attrs)) {
                  has_any_python_attrs = true;
               }
               source_attrs.push_back(std::move(attrs));
            }
            // True zero-overhead path after native collation when no Python fields exist.
            if(field_specs.empty() && ! has_any_python_attrs) {
               return out;
            }
         } else {
            if(! nb::isinstance< nb::dict >(graph_field_specs_obj)) {
               throw std::invalid_argument("graph_field_specs must be a dict when provided");
            }
            field_specs = canonicalize_python_graph_field_specs(
               nb::cast< nb::dict >(graph_field_specs_obj)
            );
            for(const auto& source : source_objects) {
               source_attrs.push_back(batch_encoding_python_attrs(source));
            }
         }

         if(field_specs.empty()) {
            seed_default_python_field_specs_from_attrs(field_specs, source_attrs);
         }

         if(field_specs.empty()) {
            return out;
         }

         for(const auto& [key, mode] : field_specs) {
            const nb::str key_obj(key.c_str());
            const std::string key_with_ptr = key + "_ptr";

            if(mode == PythonFieldMode::STACK) {
               nb::list values = collate_python_stack_values(source_attrs, key_obj);
               out.attr("__setattr__")(key.c_str(), std::move(values));
               continue;
            }

            if(mode == PythonFieldMode::RAGGED_CAT) {
               auto ragged = collate_python_ragged_values(source_attrs, key_obj);
               out.attr("__setattr__")(key.c_str(), std::move(ragged.values));
               out.attr("__setattr__")(key_with_ptr.c_str(), std::move(ragged.ptr));
               continue;
            }

            nb::object first = collate_python_const_value(key, source_attrs, key_obj);
            out.attr("__setattr__")(key.c_str(), std::move(first));
         }

         register_batch_encoding_graph_field_specs(
            out, python_graph_field_specs_to_dict(field_specs)
         );
         return out;
      },
      "encodings"_a,
      "graph_field_specs"_a = nb::none(),
      "py_field_specs"_a = nb::none()
   );

   nb::class_< HGraphEncoderEngine::Config >(m, "HGraphEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](HGraphEncoderEngine::Config* self, const nb::kwargs& kwargs) {
            new(self) HGraphEncoderEngine::Config();
            apply_hgraph_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("symbol_type_id", &HGraphEncoderEngine::Config::symbol_type_id)
      .def_rw("nullary_object_name", &HGraphEncoderEngine::Config::nullary_object_name)
      .def_rw("max_goal_level", &HGraphEncoderEngine::Config::max_goal_level)
      .def_rw("support_literals", &HGraphEncoderEngine::Config::support_literals)
      .def_rw(
         "goal_satisfaction_derivations",
         &HGraphEncoderEngine::Config::goal_satisfaction_derivations
      )
      .def_rw("add_nullary_predicates", &HGraphEncoderEngine::Config::add_nullary_predicates)
      .def_rw("ignore_actions", &HGraphEncoderEngine::Config::ignore_actions)
      .def_rw("include_static", &HGraphEncoderEngine::Config::include_static)
      .def_rw("include_lgan_edges", &HGraphEncoderEngine::Config::include_lgan_edges)
      .def_rw("include_empty_edge_types", &HGraphEncoderEngine::Config::include_empty_edge_types)
      .def_rw("export_node_names", &HGraphEncoderEngine::Config::export_node_names)
      .def_rw("history_link_relation", &HGraphEncoderEngine::Config::history_link_relation)
      .def_rw(
         "lgan_nn_edge_pos",
         &HGraphEncoderEngine::Config::lgan_nn_edge_pos,
         "lgan_nn_edge_pos"_a = defaults::lgan_nn_edge_pos
      );

   nb::class_< HGraphEncoderEngine >(m, "HGraphEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, HGraphEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, HGraphEncoderEngine::Config >())
      .def_prop_ro("config", &HGraphEncoderEngine::get_config, nb::rv_policy::reference_internal)
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, builder);
            return builder.build();
         },
         "state"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, goals, actions, builder);
            return builder.build();
         },
         "state"_a,
         "goals"_a,
         "actions"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >& history_subgoals,
            std::optional< int > history_max_steps) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, goals, actions, history_subgoals, history_max_steps, builder);
            return builder.build();
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            BatchBuilder& builder) { encoder.encode(state, builder); },
         "state"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            BatchBuilder& builder) { encoder.encode(state, goals, actions, builder); },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >& history_subgoals,
            std::optional< int > history_max_steps,
            BatchBuilder& builder) {
            encoder.encode(state, goals, actions, history_subgoals, history_max_steps, builder);
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         "builder"_a
      );

   nb::class_< HGraphMutableStreamEncoder >(m, "HGraphMutableStreamEncoder")
      .def(nb::init< HGraphEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(&HGraphMutableStreamEncoder::append),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &HGraphMutableStreamEncoder::append
         ),
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphMutableStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State& >(
            &HGraphMutableStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &HGraphMutableStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphMutableStreamEncoder::update),
         "id"_a,
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &HGraphMutableStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &HGraphMutableStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush", &HGraphMutableStreamEncoder::flush)
      .def("flush_pyg", &HGraphMutableStreamEncoder::flush_pyg)
      .def("reset", &HGraphMutableStreamEncoder::reset);

   nb::class_< HGraphStreamEncoder >(m, "HGraphStreamEncoder")
      .def(nb::init< HGraphEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(&HGraphStreamEncoder::append),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(&HGraphStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("flush", &HGraphStreamEncoder::flush)
      .def("flush_pyg", &HGraphStreamEncoder::flush_pyg)
      .def("reset", &HGraphStreamEncoder::reset);
}

}  // namespace mifrost
