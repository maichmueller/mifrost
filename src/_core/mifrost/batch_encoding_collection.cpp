#include "mifrost/batch_encoding_collection.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_python_collation.hpp"
#include "mifrost/batch_encoding_schema.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/nb_instance.hpp"

namespace nb = nanobind;

namespace mifrost {

nb::object
batch_encodings_from_sequence(nb::sequence encodings, nb::object collate_spec_obj, bool fast_path)
{
   using BatchEncoding = BatchBuilder::BatchEncoding;

   auto enc_cast = [](const nb::handle& source) -> BatchEncoding* {
      return require_instance_ptr< BatchBuilder::BatchEncoding >(
         source, "batch_encodings expects BatchEncoding inputs"
      );
   };

   if(nb::len(encodings) == 0) {
      return nb::cast(BatchBuilder::BatchEncoding{});
   }
   const BatchEncoding* zeroth_entry = enc_cast(encodings[0]);

   std::vector< const BatchEncoding* > entries;
   entries.reserve(nb::len(encodings));
   entries.push_back(zeroth_entry);
   for(size_t i = 1; i < static_cast< size_t >(nb::len(encodings)); ++i) {
      entries.push_back(enc_cast(encodings[i]));
   }

   bool use_fast_path = false;
   if(fast_path and not entries.empty()) {
      const auto expected_fp = schema_fingerprint(*entries.front());
      use_fast_path = true;
      for(size_t i = 1; i < entries.size(); ++i) {
         if(schema_fingerprint(*entries[i]) != expected_fp) {
            use_fast_path = false;
            break;
         }
      }
   }

   BatchBuilder builder;
   builder.set_graph_kind(zeroth_entry->graph_kind);
   for(size_t i = 0; i < entries.size(); ++i) {
      const BatchEncoding* encoding = entries[i];
      if(encoding->num_graphs != 1) {
         throw std::invalid_argument("batch_encodings expects inputs with num_graphs == 1");
      }
      if(not use_fast_path or i == 0) {
         validate_batch_encoding_graph_fields(*encoding, "batch_encodings input validation");
      }
      builder.append_batch_encoding(*encoding);
   }

   BatchEncoding out = builder.build();
   auto [collate_spec, source_attrs] = std::invoke([&] {
      try {
         return build_python_collation_inputs(encodings, std::move(collate_spec_obj));
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "batch_encodings collate_spec preparation failed: " + std::string(ex.what())
         );
      }
   });

   const auto reserved_native_keys = batch_encoding_native_tensor_keys(out);
   auto filtered_specs = filter_python_collate_spec_for_native_collisions(
      collate_spec, reserved_native_keys
   );
   const auto default_keys = collect_default_python_collation_keys(
      source_attrs, filtered_specs
   );
   for(const auto& key : default_keys) {
      if(reserved_native_keys.contains(key)) {
         throw std::invalid_argument(
            "Default collation key '" + key + "' collides with a native field key"
         );
      }
   }
   auto out_py = nb::cast(out);
   if(filtered_specs.empty() and default_keys.empty()) {
      return out_py;
   }

   try {
      nb::dict out_attrs = apply_python_collation(
         filtered_specs,
         source_attrs,
         std::views::iota(size_t{0}, nb::len(encodings))
            | std::views::transform([&](size_t i) { return enc_cast(encodings[i]); })
      );
      nb::dict default_attrs = apply_default_python_collation(default_keys, source_attrs);
      for(auto [k, v] : default_attrs) {
         out_attrs[k] = nb::borrow< nb::object >(v);
      }
      for(auto [k, v] : out_attrs) {
         py::set_python_attribute(out_py, nb::str(k), v);
      }
   } catch(const std::exception& ex) {
      throw std::invalid_argument(
         "batch_encodings python collation failed: " + std::string(ex.what())
      );
   }

   if(not filtered_specs.empty()) {
      try {
         register_batch_encoding_collate_spec(
            out_py, python_collate_spec_to_dict(filtered_specs)
         );
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "batch_encodings collate_spec registration failed: " + std::string(ex.what())
         );
      }
   }

   return out_py;
}

}  // namespace mifrost
