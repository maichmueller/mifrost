#include "mifrost/batch_encoding_python_collation.hpp"

#include <fmt/format.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/graph_fields.hpp"
#include "mifrost/core/nb_instance.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

constexpr std::string_view kPythonCollateSpecAttr = "__mifrost_collate_spec__";
constexpr std::string_view kPythonTensorDeviceAttr = "__mifrost_tensor_device__";
constexpr std::string_view kPythonTensorCacheAttr = "__mifrost_tensor_cache__";

bool try_cast_python_bool(nb::handle value, bool& out)
{
   try {
      out = nb::cast< bool >(value);
      return true;
   } catch(...) {
      return false;
   }
}

bool is_numpy_array(nb::handle value)
{
   const nb::handle type = py::numpy_array_type();
   if(type.is_none()) {
      return false;
   }
   return nb::isinstance(value, type);
}

bool python_eq_returns_true(nb::handle lhs, nb::handle rhs)
{
   bool result = false;
   nb::handle eq = py::operator_eq_fn();
   if(not try_cast_python_bool(eq(lhs, rhs), result)) {
      throw std::invalid_argument("Python const field comparison produced a non-boolean result");
   }
   return result;
}

bool torch_tensors_equal_exact(const nb::object& lhs, const nb::object& rhs)
{
   if(not python_eq_returns_true(lhs.attr("dtype"), rhs.attr("dtype"))) {
      return false;
   }
   if(not python_eq_returns_true(lhs.attr("shape"), rhs.attr("shape"))) {
      return false;
   }
   if(not python_eq_returns_true(lhs.attr("stride")(), rhs.attr("stride")())) {
      return false;
   }
   if(not python_eq_returns_true(lhs.attr("device"), rhs.attr("device"))) {
      return false;
   }
   if(not python_eq_returns_true(lhs.attr("layout"), rhs.attr("layout"))) {
      return false;
   }

   const nb::handle equal = py::torch_equal_fn();
   if(equal.is_none()) {
      throw std::invalid_argument("Torch tensor comparison requested but torch is unavailable");
   }
   return nb::cast< bool >(nb::borrow< nb::object >(equal)(lhs, rhs));
}

bool numpy_arrays_equal_exact(const nb::object& lhs, const nb::object& rhs)
{
   if(not python_eq_returns_true(lhs.attr("dtype"), rhs.attr("dtype"))) {
      return false;
   }
   if(not python_eq_returns_true(lhs.attr("shape"), rhs.attr("shape"))) {
      return false;
   }
   if(not python_eq_returns_true(lhs.attr("strides"), rhs.attr("strides"))) {
      return false;
   }

   const nb::handle array_equal = py::numpy_array_equal_fn();
   if(array_equal.is_none()) {
      throw std::invalid_argument("NumPy array comparison requested but numpy is unavailable");
   }
   return nb::cast< bool >(nb::borrow< nb::object >(array_equal)(lhs, rhs));
}

bool python_objects_equal_for_const(const nb::object& lhs, const nb::object& rhs)
{
   if(lhs.ptr() == rhs.ptr()) {
      return true;
   }
   if(lhs.is_none() or rhs.is_none()) {
      return lhs.is_none() and rhs.is_none();
   }

   const bool lhs_is_torch = is_torch_tensor(lhs);
   const bool rhs_is_torch = is_torch_tensor(rhs);
   if(lhs_is_torch or rhs_is_torch) {
      if(not(lhs_is_torch and rhs_is_torch)) {
         return false;
      }
      return torch_tensors_equal_exact(lhs, rhs);
   }

   const bool lhs_is_numpy = is_numpy_array(lhs);
   const bool rhs_is_numpy = is_numpy_array(rhs);
   if(lhs_is_numpy or rhs_is_numpy) {
      if(not(lhs_is_numpy and rhs_is_numpy)) {
         return false;
      }
      return numpy_arrays_equal_exact(lhs, rhs);
   }

   return python_eq_returns_true(lhs, rhs);
}

std::string_view canonical_python_field_mode(std::string_view mode)
{
   if(ascii_iequals(mode, "stack")) {
      return "stack";
   }
   if(ascii_iequals(mode, "cat")) {
      return "cat";
   }
   if(ascii_iequals(mode, "ragged_cat")) {
      return "ragged_cat";
   }
   if(ascii_iequals(mode, "const")) {
      return "const";
   }
   throw std::invalid_argument("Unsupported Python field mode: " + std::string(mode));
}

GraphFieldMode python_field_mode_from_name(std::string_view mode)
{
   const auto canonical = canonical_python_field_mode(mode);
   if(canonical == "stack") {
      return GraphFieldMode::STACK;
   }
   if(canonical == "cat") {
      return GraphFieldMode::CAT;
   }
   if(canonical == "ragged_cat") {
      return GraphFieldMode::RAGGED_CAT;
   }
   return GraphFieldMode::CONST;
}

std::string_view python_field_mode_name(GraphFieldMode mode)
{
   switch(mode) {
      case GraphFieldMode::STACK: return "stack";
      case GraphFieldMode::CAT: return "cat";
      case GraphFieldMode::RAGGED_CAT: return "ragged_cat";
      case GraphFieldMode::CONST: return "const";
   }
   throw std::logic_error("Unknown GraphFieldMode");
}

PythonFieldDType python_field_dtype_from_name(std::string_view dtype)
{
   if(ascii_iequals(dtype, "<class 'str'>")) {
      return PythonFieldDType::STR;
   }
   if(ascii_iequals(dtype, "<class 'float'>")) {
      return PythonFieldDType::F32;
   }
   if(ascii_iequals(dtype, "<class 'int'>")) {
      return PythonFieldDType::I64;
   }
   if(ascii_iequals(dtype, "pyobj") or ascii_iequals(dtype, "python")
      or ascii_iequals(dtype, "object")) {
      return PythonFieldDType::PYOBJ;
   }
   if(ascii_iequals(dtype, "str") or ascii_iequals(dtype, "string")) {
      return PythonFieldDType::STR;
   }
   if(ascii_iequals(dtype, "f32") or ascii_iequals(dtype, "float32")
      or ascii_iequals(dtype, "float")) {
      return PythonFieldDType::F32;
   }
   if(ascii_iequals(dtype, "i64") or ascii_iequals(dtype, "int64") or ascii_iequals(dtype, "int")) {
      return PythonFieldDType::I64;
   }
   throw std::invalid_argument("Python collate_spec dtype must be one of {pyobj, str, f32, i64}");
}

const char* python_field_dtype_name(PythonFieldDType dtype)
{
   switch(dtype) {
      case PythonFieldDType::PYOBJ: return "pyobj";
      case PythonFieldDType::STR: return "str";
      case PythonFieldDType::F32: return "f32";
      case PythonFieldDType::I64: return "i64";
   }
   throw std::logic_error("Unknown PythonFieldDType");
}

bool is_string_like(nb::handle value)
{
   return nb::isinstance< nb::str >(value) or nb::isinstance< nb::bytes >(value);
}

bool is_numeric_scalar(nb::handle value)
{
   return nb::isinstance< nb::bool_ >(value) or nb::isinstance< nb::int_ >(value)
          or nb::isinstance< nb::float_ >(value);
}

nb::object torch_dtype_for_python_field_dtype(PythonFieldDType dtype)
{
   nb::handle torch = py::torch_module();
   switch(dtype) {
      case PythonFieldDType::F32: return torch.attr("float32");
      case PythonFieldDType::I64: return torch.attr("int64");
      default: throw std::invalid_argument("Expected numeric dtype for tensor collation");
   }
}

PythonFieldDType infer_numeric_dtype_from_tensor(const nb::object& tensor)
{
   const auto dtype_str = nb::cast< std::string >(tensor.attr("dtype").attr("__str__")());
   if(dtype_str.find("float") != std::string::npos or dtype_str.find("double") != std::string::npos
      or dtype_str.find("bfloat") != std::string::npos
      or dtype_str.find("half") != std::string::npos) {
      return PythonFieldDType::F32;
   }
   if(dtype_str.find("int") != std::string::npos or dtype_str.find("long") != std::string::npos
      or dtype_str.find("bool") != std::string::npos
      or dtype_str.find("uint") != std::string::npos) {
      return PythonFieldDType::I64;
   }
   throw std::invalid_argument("Cannot infer numeric dtype from tensor dtype '" + dtype_str + "'");
}

nb::object
normalize_numeric_tensor(const std::string& key, const PythonFieldSpec& spec, nb::handle value)
{
   nb::handle torch = py::torch_module();
   nb::object tensor = torch.attr("as_tensor")(
      value, "dtype"_a = torch_dtype_for_python_field_dtype(spec.dtype)
   );
   int64_t ndim = nb::cast< int64_t >(tensor.attr("ndim"));
   if(ndim == 0) {
      tensor = tensor.attr("reshape")(nb::make_tuple(1));
      ndim = 1;
   }

   if(spec.mode == GraphFieldMode::STACK or spec.mode == GraphFieldMode::CONST) {
      const int64_t total = nb::cast< int64_t >(tensor.attr("numel")());
      if(total != spec.dim) {
         throw std::invalid_argument(
            "Field '" + key + "' STACK/CONST expects exactly dim elements"
         );
      }
      if(spec.dim == 1) {
         return tensor.attr("reshape")(nb::make_tuple(1));
      }
      return tensor.attr("reshape")(nb::make_tuple(spec.dim));
   }

   if(spec.dim == 1) {
      if(ndim == 1) {
         return tensor;
      }
      if(ndim == 2) {
         const int64_t rows = nb::cast< int64_t >(tensor.attr("shape").attr("__getitem__")(0));
         const int64_t cols = nb::cast< int64_t >(tensor.attr("shape").attr("__getitem__")(1));
         if(rows == 1 or cols == 1) {
            return tensor.attr("reshape")(nb::make_tuple(rows * cols));
         }
      }
      throw std::invalid_argument("Field '" + key + "' with dim=1 expects a scalar or 1D value");
   }

   if(ndim != 2) {
      throw std::invalid_argument("Field '" + key + "' with dim>1 expects a 2D value");
   }
   const int64_t rows = nb::cast< int64_t >(tensor.attr("shape").attr("__getitem__")(0));
   const int64_t cols = nb::cast< int64_t >(tensor.attr("shape").attr("__getitem__")(1));
   if(spec.cat_dim == 0) {
      if(cols != spec.dim) {
         throw std::invalid_argument("Field '" + key + "' with cat_dim=0 expects shape [N, dim]");
      }
      return tensor;
   }
   if(rows != spec.dim) {
      throw std::invalid_argument("Field '" + key + "' with cat_dim=1 expects shape [dim, N]");
   }
   return tensor;
}

int64_t rows_for_tensor_piece(const PythonFieldSpec& spec, const nb::object& tensor)
{
   if(spec.dim == 1) {
      return nb::cast< int64_t >(tensor.attr("numel")());
   }
   if(spec.cat_dim == 1) {
      return nb::cast< int64_t >(tensor.attr("shape").attr("__getitem__")(1));
   }
   return nb::cast< int64_t >(tensor.attr("shape").attr("__getitem__")(0));
}

void validate_field_spec_shape_rules(const std::string_view key, const PythonFieldSpec& spec)
{
   if(spec.dtype == PythonFieldDType::F32 or spec.dtype == PythonFieldDType::I64) {
      GraphFieldSpec native{
         .dtype = spec.dtype == PythonFieldDType::F32 ? GraphFieldDType::F32 : GraphFieldDType::I64,
         .mode = spec.mode,
         .dim = spec.dim,
         .cat_dim = spec.cat_dim,
         .inc = spec.inc,
      };
      validate_graph_field_spec(key, native);
      return;
   }

   if(spec.mode == GraphFieldMode::CAT) {
      throw std::invalid_argument(
         fmt::format("Python field '{}' CAT mode requires numeric dtype", key)
      );
   }
   if(spec.cat_dim != 0 or spec.dim != 1 or spec.inc.kind != GraphFieldInc::Kind::NONE) {
      throw std::invalid_argument(
         fmt::format("Python field '{}' non-numeric dtype does not support dim/cat_dim/inc", key)
      );
   }
}

PythonFieldSpec parse_python_field_spec(const std::string_view key, nb::handle spec_obj)
{
   PythonFieldSpec spec;
   if(spec_obj.is_none()) {
      spec.inferred = true;
      return spec;
   }
   if(nb::isinstance< nb::str >(spec_obj)) {
      const auto mode_raw = nb::str(spec_obj);
      spec.mode = python_field_mode_from_name(mode_raw.c_str());
      spec.inferred = false;
      validate_field_spec_shape_rules(key, spec);
      return spec;
   }
   if(not nb::isinstance< nb::dict >(spec_obj)) {
      throw std::invalid_argument(
         fmt::format("Collate spec for '{}' must be a dict or mode string", key)
      );
   }

   nb::dict spec_dict;
   try {
      spec_dict = nb::cast< nb::dict >(spec_obj);
   } catch(const std::exception& ex) {
      throw std::invalid_argument(
         fmt::format("Field '{}' spec is not a dict-like object: {}", key, ex.what())
      );
   }
   static const std::unordered_set< std::string > kAllowed{
      "dtype", "mode", "dim", "cat_dim", "inc"
   };
   for(auto [name_obj, value_obj] : spec_dict) {
      (void) value_obj;
      const auto name = nb::str(name_obj);
      if(not kAllowed.contains(name.c_str())) {
         throw std::invalid_argument(
            fmt::format("Unsupported collate spec key '{}' for field '{}'", name.c_str(), key)
         );
      }
   }

   if(spec_dict.contains("dtype")) {
      nb::handle dtype_obj = spec_dict["dtype"];
      if(nb::isinstance(dtype_obj, py::builtins_type_type())) {
         if(dtype_obj.ptr() == py::builtins_str_type().ptr()) {
            spec.dtype = PythonFieldDType::STR;
         } else if(dtype_obj.ptr() == py::builtins_float_type().ptr()) {
            spec.dtype = PythonFieldDType::F32;
         } else if(dtype_obj.ptr() == py::builtins_int_type().ptr()) {
            spec.dtype = PythonFieldDType::I64;
         } else {
            auto dtype_s = nb::str(dtype_obj);
            spec.dtype = python_field_dtype_from_name(dtype_s.c_str());
         }
      } else {
         auto dtype_s = nb::str(dtype_obj);
         spec.dtype = python_field_dtype_from_name(dtype_s.c_str());
      }
   }
   if(spec_dict.contains("mode")) {
      const auto mode_raw = nb::str(spec_dict["mode"]);
      spec.mode = python_field_mode_from_name(mode_raw.c_str());
   }
   if(spec_dict.contains("dim")) {
      spec.dim = nb::cast< int >(spec_dict["dim"]);
   }
   if(spec_dict.contains("cat_dim")) {
      spec.cat_dim = normalize_graph_field_cat_dim(nb::cast< int >(spec_dict["cat_dim"]));
   }
   if(spec_dict.contains("inc") and not spec_dict["inc"].is_none()) {
      if(not nb::isinstance< nb::dict >(spec_dict["inc"])) {
         throw std::invalid_argument(fmt::format("Field '{}' inc must be a dict", key));
      }
      const auto inc_dict = nb::cast< nb::dict >(spec_dict["inc"]);
      if(inc_dict.contains("kind")) {
         const auto kind = nb::str(inc_dict["kind"]);
         spec.inc.kind = graph_field_inc_kind_from_name(kind.c_str());
      }
      if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
         if(not inc_dict.contains("node_type")) {
            throw std::invalid_argument(
               fmt::format("Field '{}' inc NODE_OFFSET requires node_type", key)
            );
         }
         spec.inc.node_type = nb::str(inc_dict["node_type"]).c_str();
      } else if(spec.inc.kind == GraphFieldInc::Kind::FIELD_OFFSET) {
         if(not inc_dict.contains("field_key")) {
            throw std::invalid_argument(
               fmt::format("Field '{}' inc FIELD_OFFSET requires field_key", key)
            );
         }
         spec.inc.field_key = nb::str(inc_dict["field_key"]).c_str();
      }
   }
   spec.inferred = false;
   validate_field_spec_shape_rules(key, spec);
   return spec;
}

bool try_get_python_attr(const nb::dict& attrs, nb::handle key_obj, nb::object& out)
{
   if(not attrs.contains(key_obj)) {
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

PythonRaggedCollation
collate_python_ragged_values(const std::vector< nb::dict >& source_attrs, nb::handle key_obj)
{
   PythonRaggedCollation out;
   int64_t offset = 0;
   out.ptr.append(offset);
   for(const auto& attrs : source_attrs) {
      nb::object value;
      if(try_get_python_attr(attrs, key_obj, value)) {
         if(nb::isinstance< nb::list >(value) or nb::isinstance< nb::tuple >(value)) {
            for(nb::handle entry : value) {
               out.values.append(entry);
               offset += 1;
            }
         } else if(not value.is_none()) {
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
      if(not try_get_python_attr(attrs, key_obj, value)) {
         throw std::invalid_argument(
            "Python const field '" + key + "' missing value for encoding index "
            + std::to_string(source_idx)
         );
      }
      if(not found) {
         first = value;
         found = true;
         continue;
      }
      if(not python_objects_equal_for_const(value, first)) {
         throw std::invalid_argument(
            "Python const field '" + key + "' has non-constant values across encodings"
         );
      }
   }
   if(not found) {
      throw std::invalid_argument(
         "Python const field '" + key + "' requires at least one encoding value"
      );
   }
   return first;
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
   return key.starts_with("__mifrost_") or key == kPythonCollateSpecAttr
          or key == kPythonTensorDeviceAttr or key == kPythonTensorCacheAttr;
}

bool is_pyg_structural_attr_key(std::string_view key)
{
   static constexpr std::array< std::string_view, 10 > kReserved{
      "x",
      "edge_index",
      "edge_attr",
      "batch",
      "ptr",
      "x_dict",
      "edge_index_dict",
      "edge_attr_dict",
      "batch_dict",
      "ptr_dict",
   };
   return std::ranges::find(kReserved, key) != kReserved.end();
}

bool is_forbidden_dynamic_attr_key(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view key
)
{
   return is_reserved_python_attr_key(key) or is_pyg_structural_attr_key(key)
          or batch_encoding_has_native_tensor(encoding, key);
}

void batch_encoding_clear_python_attrs(nb::handle self)
{
   nb::dict attrs = nb::cast< nb::dict >(self.attr("__dict__"));
   attrs.clear();
}

nb::dict batch_encoding_collate_spec(nb::handle self)
{
   auto attrs = nb::cast< nb::dict >(self.attr("__dict__"));
   if(not attrs.contains(kPythonCollateSpecAttr.data())) {
      return {};
   }
   auto raw_specs = nb::borrow< nb::object >(attrs[kPythonCollateSpecAttr.data()]);
   if(not nb::isinstance< nb::dict >(raw_specs)) {
      throw std::invalid_argument("BatchEncoding internal collate_spec must be a dict");
   }
   return nb::cast< nb::dict >(raw_specs);
}

void batch_encoding_apply_python_attrs_from_state(
   nb::handle self,
   const nb::dict& state,
   nb::dict& dst
)
{
   (void) self;
   if(not state.contains("python_attrs")) {
      return;
   }
   auto src = nb::cast< nb::dict >(state["python_attrs"]);
   for(auto [key_obj, value_obj] : src) {
      dst[key_obj] = nb::borrow< nb::object >(value_obj);
   }
}

void batch_encoding_apply_python_attrs_from_state(nb::handle self, const nb::dict& state)
{
   auto dst = nb::cast< nb::dict >(self.attr("__dict__"));
   batch_encoding_apply_python_attrs_from_state(self, state, dst);
}

PythonFieldSpecMap canonicalize_python_collate_spec(const nb::dict& specs)
{
   PythonFieldSpecMap out;
   for(auto [key_obj, spec_obj] : specs) {
      const auto key = nb::str(key_obj);
      auto key_view = std::string_view{key.c_str()};
      if(key_view.empty()) {
         throw std::invalid_argument("Python collate_spec keys must be non-empty");
      }
      try {
         out[key_view] = parse_python_field_spec(key_view, nb::borrow< nb::object >(spec_obj));
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            fmt::format("Failed to parse collate_spec entry for key '{}': {}", key_view, ex.what())
         );
      }
   }
   return out;
}

void merge_python_collate_spec(PythonFieldSpecMap& dst, const PythonFieldSpecMap& src)
{
   for(const auto& [key, incoming_spec] : src) {
      if(const auto it = dst.find(key); it != dst.end()) {
         if(it->second != incoming_spec) {
            throw std::invalid_argument("Conflicting Python collate_spec for key '" + key + "'");
         }
      } else {
         dst.emplace(key, incoming_spec);
      }
   }
}

nb::dict python_collate_spec_to_dict(const PythonFieldSpecMap& specs)
{
   nb::dict out;
   for(const auto& [key, spec] : specs) {
      nb::dict normalized;
      normalized["dtype"] = python_field_dtype_name(spec.dtype);
      normalized["mode"] = std::string(python_field_mode_name(spec.mode));
      if(spec.dtype == PythonFieldDType::F32 or spec.dtype == PythonFieldDType::I64) {
         normalized["dim"] = spec.dim;
         normalized["cat_dim"] = spec.cat_dim;
         nb::dict inc;
         inc["kind"] = graph_field_inc_kind_name(spec.inc.kind);
         if(spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
            inc["node_type"] = spec.inc.node_type;
         } else if(spec.inc.kind == GraphFieldInc::Kind::FIELD_OFFSET) {
            inc["field_key"] = spec.inc.field_key;
         }
         normalized["inc"] = std::move(inc);
      }
      out[key.c_str()] = std::move(normalized);
   }
   return out;
}

std::tuple< PythonFieldSpecMap, std::vector< nanobind::dict > > build_python_collation_inputs(
   const nb::sequence& source_objects,
   const nb::object& collate_spec_obj
)
{
   std::vector< nanobind::dict > source_attrs;
   PythonFieldSpecMap collate_spec;
   source_attrs.reserve(nb::len(source_objects));
   if(collate_spec_obj.is_none()) {
      for(const auto& source : source_objects) {
         source_attrs.push_back(batch_encoding_python_attrs(source));
      }
   } else {
      if(not nb::isinstance< nb::dict >(collate_spec_obj)) {
         throw std::invalid_argument("collate_spec must be a dict when provided");
      }
      collate_spec = canonicalize_python_collate_spec(nb::cast< nb::dict >(collate_spec_obj));
      for(const auto& source : source_objects) {
         source_attrs.push_back(batch_encoding_python_attrs(source));
      }
   }
   return std::tuple{std::move(collate_spec), std::move(source_attrs)};
}

std::vector< std::string > collect_default_python_collation_keys(
   const std::vector< nb::dict >& source_attrs,
   const PythonFieldSpecMap& collate_spec
)
{
   std::set< std::string > key_set;
   for(const auto& attrs : source_attrs) {
      for(auto [key_obj, value_obj] : attrs) {
         (void) value_obj;
         const std::string key = py::to_std_string(key_obj);
         if(collate_spec.contains(key)) {
            continue;
         }
         if(is_reserved_python_attr_key(key)) {
            continue;
         }
         key_set.insert(key);
      }
   }

   std::vector< std::string > out;
   out.reserve(key_set.size());
   for(const auto& key : key_set) {
      if(is_pyg_structural_attr_key(key)) {
         throw std::invalid_argument(
            "Default collation key '" + key + "' collides with a reserved key"
         );
      }
      const nb::str key_obj(key.c_str());
      for(size_t source_idx = 0; source_idx < source_attrs.size(); ++source_idx) {
         if(not source_attrs[source_idx].contains(key_obj)) {
            throw std::invalid_argument(
               "Default collation key '" + key + "' missing value for encoding index "
               + std::to_string(source_idx)
            );
         }
      }
      out.push_back(key);
   }
   return out;
}

nb::dict apply_default_python_collation(
   const std::vector< std::string >& keys,
   const std::vector< nb::dict >& source_attrs
)
{
   nb::dict out;
   for(const auto& key : keys) {
      const nb::str key_obj(key.c_str());
      std::vector< nb::object > values;
      values.reserve(source_attrs.size());
      bool any_dict = false;
      bool all_dict = true;
      for(size_t source_idx = 0; source_idx < source_attrs.size(); ++source_idx) {
         const auto& attrs = source_attrs[source_idx];
         if(not attrs.contains(key_obj)) {
            throw std::invalid_argument(
               "Default collation key '" + key + "' missing value for encoding index "
               + std::to_string(source_idx)
            );
         }
         nb::object value = nb::borrow< nb::object >(attrs[key_obj]);
         const bool is_dict = nb::isinstance< nb::dict >(value);
         any_dict = any_dict or is_dict;
         all_dict = all_dict and is_dict;
         values.push_back(std::move(value));
      }

      if(any_dict and not all_dict) {
         throw std::invalid_argument(
            "Default collation key '" + key + "' mixes dict and non-dict values across encodings"
         );
      }

      if(not any_dict) {
         nb::list collated;
         for(auto& value : values) {
            collated.append(value);
         }
         out[key_obj] = std::move(collated);
         continue;
      }

      const nb::dict first = nb::cast< nb::dict >(values.front());
      std::vector< std::string > nested_keys;
      nested_keys.reserve(nb::len(first));
      for(auto [nested_key_obj, nested_value_obj] : first) {
         (void) nested_value_obj;
         nested_keys.push_back(py::to_std_string(nested_key_obj));
      }

      for(size_t idx = 1; idx < values.size(); ++idx) {
         const nb::dict current = nb::cast< nb::dict >(values[idx]);
         if(nb::len(current) != nb::len(first)) {
            throw std::invalid_argument(
               "Default dict collation key '" + key
               + "' requires identical key sets across encodings"
            );
         }
         for(const auto& nested_key : nested_keys) {
            if(not current.contains(nested_key.c_str())) {
               throw std::invalid_argument(
                  "Default dict collation key '" + key
                  + "' requires identical key sets across encodings"
               );
            }
         }
      }

      nb::dict collated_map;
      for(const auto& nested_key : nested_keys) {
         nb::list collated_values;
         for(auto& value : values) {
            const nb::dict current = nb::cast< nb::dict >(value);
            collated_values.append(nb::borrow< nb::object >(current[nested_key.c_str()]));
         }
         collated_map[nested_key.c_str()] = std::move(collated_values);
      }
      out[key_obj] = std::move(collated_map);
   }
   return out;
}

PythonFieldSpecMap filter_python_collate_spec_for_native_collisions(
   const PythonFieldSpecMap& collate_spec,
   const std::set< std::string >& reserved_native_keys
)
{
   PythonFieldSpecMap out;
   for(const auto& [key, spec] : collate_spec) {
      if(reserved_native_keys.contains(key) or is_reserved_python_attr_key(key)
         or is_pyg_structural_attr_key(key)) {
         throw std::invalid_argument(
            "Python collate_spec key '" + key + "' collides with a reserved/native key"
         );
      }
      if(spec.mode == GraphFieldMode::RAGGED_CAT) {
         const std::string ptr_key = key + "_ptr";
         if(reserved_native_keys.contains(ptr_key) or is_reserved_python_attr_key(ptr_key)
            or is_pyg_structural_attr_key(ptr_key)) {
            throw std::invalid_argument(
               "Python collate_spec key '" + key + "' collides with reserved ptr key '" + ptr_key
               + "'"
            );
         }
      }
      out.emplace(key, spec);
   }
   return out;
}

void validate_string_value(const std::string& key, GraphFieldMode mode, nb::handle value)
{
   if(mode == GraphFieldMode::RAGGED_CAT) {
      if(nb::isinstance< nb::str >(value)) {
         return;
      }
      if(nb::isinstance< nb::list >(value) or nb::isinstance< nb::tuple >(value)) {
         for(nb::handle entry : nb::borrow< nb::object >(value)) {
            if(not nb::isinstance< nb::str >(entry)) {
               throw std::invalid_argument(
                  "Field '" + key + "' ragged string values must contain only strings"
               );
            }
         }
         return;
      }
      throw std::invalid_argument(
         "Field '" + key + "' ragged string value must be a string or sequence of strings"
      );
   }
   if(not nb::isinstance< nb::str >(value)) {
      throw std::invalid_argument("Field '" + key + "' expects string values");
   }
}

void register_batch_encoding_collate_spec(nb::handle self, const nb::dict& specs)
{
   auto normalized = canonicalize_python_collate_spec(specs);
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      self, "internal collate_spec registration called with invalid BatchEncoding instance"
   );
   const auto native_keys = batch_encoding_native_tensor_keys(*encoding);
   for(const auto& [key, spec] : normalized) {
      if(is_forbidden_dynamic_attr_key(*encoding, key)) {
         throw std::invalid_argument(
            "Python collate_spec key '" + key + "' collides with a reserved/native key"
         );
      }
      if(spec.mode == GraphFieldMode::RAGGED_CAT) {
         const std::string ptr_key = key + "_ptr";
         if(native_keys.contains(ptr_key) or is_reserved_python_attr_key(ptr_key)
            or is_pyg_structural_attr_key(ptr_key)) {
            throw std::invalid_argument(
               "Python collate_spec key '" + key + "' collides with reserved ptr key '" + ptr_key
               + "'"
            );
         }
      }
   }
   auto existing = canonicalize_python_collate_spec(batch_encoding_collate_spec(self));
   merge_python_collate_spec(existing, normalized);
   nb::dict attrs = nb::cast< nb::dict >(self.attr("__dict__"));
   attrs[kPythonCollateSpecAttr.data()] = python_collate_spec_to_dict(existing);
}

void copy_python_attrs_to_object(
   nb::handle src,
   nb::handle dst,
   std::optional< bool > as_batch,
   const BatchBuilder::BatchEncoding& encoding
)
{
   const int64_t num_graphs = encoding.num_graphs;
   const bool want_batch = as_batch.value_or(num_graphs != 1);
   nb::dict attrs = batch_encoding_python_attrs(src);
   for(auto [key_obj, value_obj] : attrs) {
      const auto key = nb::str(key_obj);
      if(is_forbidden_dynamic_attr_key(encoding, key.c_str())) {
         continue;
      }
      nb::object value = nb::borrow< nb::object >(value_obj);
      if(not want_batch and num_graphs == 1 and nb::isinstance< nb::list >(value)
         and nb::len(value) == 1) {
         dst.attr("__setattr__")(key.c_str(), value.attr("__getitem__")(0));
      } else {
         dst.attr("__setattr__")(key.c_str(), value);
      }
   }
}

}  // namespace mifrost
