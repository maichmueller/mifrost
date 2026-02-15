#include "mifrost/batch_encoding_python_collation.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <utility>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/core/graph_fields.hpp"
#include "mifrost/core/nb_instance.hpp"

namespace nb = nanobind;

namespace mifrost {

namespace {

constexpr std::string_view kPythonGraphFieldSpecsAttr = "__mifrost_graph_field_specs__";

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

}  // namespace

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

void batch_encoding_clear_python_attrs(nb::handle self)
{
   (void) self;
}

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
   (void) self;
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
   auto dst = nb::cast< nb::dict >(self.attr("__dict__"));
   batch_encoding_apply_python_attrs_from_state(self, state, dst);
}

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

PythonCollationInputs build_python_collation_inputs(
   const std::vector< nb::object >& source_objects,
   nb::object graph_field_specs_obj
)
{
   PythonCollationInputs out;
   out.source_attrs.reserve(source_objects.size());
   if(graph_field_specs_obj.is_none()) {
      bool has_any_python_attrs = false;
      for(const auto& source : source_objects) {
         nb::dict attrs = batch_encoding_python_attrs(source);
         if(attrs.contains(kPythonGraphFieldSpecsAttr.data())) {
            auto registered_specs = canonicalize_python_graph_field_specs(
               batch_encoding_graph_field_specs(source)
            );
            merge_python_graph_field_specs(out.field_specs, registered_specs);
         }
         if(! has_any_python_attrs && has_non_reserved_python_attrs(attrs)) {
            has_any_python_attrs = true;
         }
         out.source_attrs.push_back(std::move(attrs));
      }
      if(out.field_specs.empty() && ! has_any_python_attrs) {
         return out;
      }
   } else {
      if(! nb::isinstance< nb::dict >(graph_field_specs_obj)) {
         throw std::invalid_argument("graph_field_specs must be a dict when provided");
      }
      out.field_specs = canonicalize_python_graph_field_specs(
         nb::cast< nb::dict >(graph_field_specs_obj)
      );
      for(const auto& source : source_objects) {
         out.source_attrs.push_back(batch_encoding_python_attrs(source));
      }
   }
   if(out.field_specs.empty()) {
      seed_default_python_field_specs_from_attrs(out.field_specs, out.source_attrs);
   }
   return out;
}

PythonFieldSpecMap filter_python_field_specs_for_native_collisions(
   const PythonFieldSpecMap& field_specs,
   const std::set< std::string >& reserved_native_keys
)
{
   PythonFieldSpecMap out;
   for(const auto& [key, mode] : field_specs) {
      if(reserved_native_keys.contains(key)) {
         continue;
      }
      if(mode == PythonFieldMode::RAGGED_CAT) {
         const std::string ptr_key = key + "_ptr";
         if(reserved_native_keys.contains(ptr_key)) {
            continue;
         }
      }
      out.emplace(key, mode);
   }
   return out;
}

void apply_python_collation_to_output(
   nb::handle out,
   const PythonFieldSpecMap& field_specs,
   const std::vector< nb::dict >& source_attrs
)
{
   nb::dict out_attrs = batch_encoding_python_attrs(out);
   for(const auto& [key, mode] : field_specs) {
      const nb::str key_obj(key.c_str());
      if(mode == PythonFieldMode::STACK) {
         out_attrs[key_obj] = collate_python_stack_values(source_attrs, key_obj);
         continue;
      }

      if(mode == PythonFieldMode::RAGGED_CAT) {
         auto ragged = collate_python_ragged_values(source_attrs, key_obj);
         out_attrs[key_obj] = std::move(ragged.values);
         const std::string key_with_ptr = key + "_ptr";
         out_attrs[key_with_ptr.c_str()] = std::move(ragged.ptr);
         continue;
      }

      out_attrs[key_obj] = collate_python_const_value(key, source_attrs, key_obj);
   }
}

void register_batch_encoding_graph_field_specs(nb::handle self, const nb::dict& specs)
{
   auto normalized = canonicalize_python_graph_field_specs(specs);
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      self, "BatchEncoding.register_graph_field_specs called with invalid instance"
   );
   const auto native_keys = batch_encoding_native_graph_field_keys(*encoding);
   for(const auto& [key, mode] : normalized) {
      if(native_keys.contains(key)) {
         throw std::invalid_argument(
            "Python graph field spec key '" + key + "' collides with native graph field key"
         );
      }
      if(mode == PythonFieldMode::RAGGED_CAT) {
         const std::string ptr_key = key + "_ptr";
         if(native_keys.contains(ptr_key)) {
            throw std::invalid_argument(
               "Python graph field spec key '" + key
               + "' collides with native graph field ptr key '" + ptr_key + "'"
            );
         }
      }
   }
   auto existing = canonicalize_python_graph_field_specs(batch_encoding_graph_field_specs(self));
   merge_python_graph_field_specs(existing, normalized);
   nb::dict attrs = nb::cast< nb::dict >(self.attr("__dict__"));
   attrs[kPythonGraphFieldSpecsAttr.data()] = python_graph_field_specs_to_dict(existing);
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
   const auto reserved_native_keys = batch_encoding_native_graph_field_keys(encoding);
   nb::dict attrs = batch_encoding_python_attrs(src);
   for(auto [key_obj, value_obj] : attrs) {
      const std::string key = nb::cast< std::string >(key_obj);
      if(is_reserved_python_attr_key(key) || reserved_native_keys.contains(key)) {
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

}  // namespace mifrost
