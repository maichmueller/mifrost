#pragma once

#include <absl/container/btree_map.h>
#include <nanobind/nanobind.h>

#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "mifrost/core/batch_builder.hpp"

namespace mifrost {

enum class PythonFieldMode { STACK, RAGGED_CAT, CONST };
using PythonFieldSpecMap = absl::btree_map< std::string, PythonFieldMode >;

struct PythonCollationInputs {
   PythonFieldSpecMap field_specs;
   std::vector< nanobind::dict > source_attrs;
};

nanobind::dict batch_encoding_python_attrs(nanobind::handle self);

nanobind::dict batch_encoding_python_attrs_copy(nanobind::handle self);

bool is_reserved_python_attr_key(std::string_view key);

void batch_encoding_clear_python_attrs(nanobind::handle self);

nanobind::dict batch_encoding_graph_field_specs(nanobind::handle self);

void batch_encoding_apply_python_attrs_from_state(
   nanobind::handle self,
   const nanobind::dict& state,
   nanobind::dict& dst
);

void batch_encoding_apply_python_attrs_from_state(
   nanobind::handle self,
   const nanobind::dict& state
);

PythonFieldSpecMap canonicalize_python_graph_field_specs(const nanobind::dict& specs);

void merge_python_graph_field_specs(PythonFieldSpecMap& dst, const PythonFieldSpecMap& src);

nanobind::dict python_graph_field_specs_to_dict(const PythonFieldSpecMap& specs);

PythonCollationInputs build_python_collation_inputs(
   const std::vector< nanobind::object >& source_objects,
   nanobind::object graph_field_specs_obj
);

PythonFieldSpecMap filter_python_field_specs_for_native_collisions(
   const PythonFieldSpecMap& field_specs,
   const std::set< std::string >& reserved_native_keys
);

void apply_python_collation_to_output(
   nanobind::handle out,
   const PythonFieldSpecMap& field_specs,
   const std::vector< nanobind::dict >& source_attrs
);

void register_batch_encoding_graph_field_specs(nanobind::handle self, const nanobind::dict& specs);

void copy_python_attrs_to_object(
   nanobind::handle src,
   nanobind::handle dst,
   std::optional< bool > as_batch,
   const BatchBuilder::BatchEncoding& encoding
);

}  // namespace mifrost
