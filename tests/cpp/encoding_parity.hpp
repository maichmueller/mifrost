/**
 * @file encoding_parity.hpp
 * @brief Exact batch-encoding comparison shared by the direct-View parity tests.
 *
 * Every direct/compatibility parity test asserts the same thing: two execution
 * paths produced byte-identical output. The comparison is deliberately exact --
 * no sorting, no tolerance, no normalization -- because the whole point of the
 * static-dispatch conversion is that choosing the storage mode changes nothing
 * an encoder emits, including lane ordering and target-name materialization.
 */
#pragma once

#include <gtest/gtest.h>

#include <string>
#include <variant>

#include "mifrost/core/batch_builder.hpp"

namespace mifrost_test {

template < typename Variant >
void expect_numeric_variant_equal(
   const Variant& expected,
   const Variant& actual,
   const std::string& label
)
{
   ASSERT_EQ(expected.index(), actual.index()) << label;
   std::visit(
      [&]< typename T >(const T& expected_values) {
         ASSERT_TRUE(std::holds_alternative< T >(actual)) << label;
         EXPECT_EQ(expected_values, std::get< T >(actual)) << label;
      },
      expected
   );
}

/**
 * @brief Assert that two batch encodings are identical in every exported field.
 *
 * @param label Free-form context (a configuration name, a domain) added to each
 *              failure message; parity tests run the same assertions many times
 *              over, and the field name alone does not say which run failed.
 */
inline void expect_encoding_equal(
   const mifrost::BatchBuilder::BatchEncoding& expected,
   const mifrost::BatchBuilder::BatchEncoding& actual,
   const std::string& label = {}
)
{
   EXPECT_EQ(expected.graph_kind, actual.graph_kind) << label;
   EXPECT_EQ(expected.schema_flags, actual.schema_flags) << label;
   EXPECT_EQ(expected.node_names, actual.node_names) << label;
   EXPECT_EQ(expected.object_names, actual.object_names) << label;
   EXPECT_EQ(expected.node_feature_dims, actual.node_feature_dims) << label;
   EXPECT_EQ(expected.graph_attrs, actual.graph_attrs) << label;
   EXPECT_EQ(expected.lazy_target_name_strings, actual.lazy_target_name_strings) << label;
   EXPECT_EQ(expected.ptrs, actual.ptrs) << label;
   EXPECT_EQ(expected.num_graphs, actual.num_graphs) << label;
   EXPECT_EQ(expected.node_counts, actual.node_counts) << label;

   ASSERT_EQ(expected.columns.size(), actual.columns.size()) << label;
   for(const auto& [key, expected_column] : expected.columns) {
      const auto actual_it = actual.columns.find(key);
      ASSERT_NE(actual_it, actual.columns.end()) << label << " " << key;
      EXPECT_EQ(expected_column.dim, actual_it->second.dim) << label << " " << key;
      expect_numeric_variant_equal(expected_column.data, actual_it->second.data, label + " " + key);
   }

   ASSERT_EQ(expected.graph_fields.size(), actual.graph_fields.size()) << label;
   for(const auto& [key, expected_field] : expected.graph_fields) {
      const auto actual_it = actual.graph_fields.find(key);
      ASSERT_NE(actual_it, actual.graph_fields.end()) << label << " " << key;
      EXPECT_EQ(expected_field.spec, actual_it->second.spec) << label << " " << key;
      EXPECT_EQ(expected_field.ptr, actual_it->second.ptr) << label << " " << key;
      expect_numeric_variant_equal(
         expected_field.values, actual_it->second.values, label + " " + key
      );
      ASSERT_EQ(expected_field.pending.has_value(), actual_it->second.pending.has_value())
         << label << " " << key;
      if(expected_field.pending.has_value()) {
         expect_numeric_variant_equal(
            *expected_field.pending, *actual_it->second.pending, label + " " + key
         );
      }
   }
}

}  // namespace mifrost_test
