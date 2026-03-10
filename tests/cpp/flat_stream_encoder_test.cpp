#include <gtest/gtest.h>

#include "mifrost/core/flat_horizon_encoder.hpp"
#include "mifrost/core/flat_relation_encoder.hpp"
#include "mifrost/core/transition_dag.hpp"
#include "test_utils.hpp"

using namespace mifrost;

namespace {

void expect_schema_equal(const Schema& actual, const Schema& expected)
{
   EXPECT_EQ(actual.version, expected.version);
   EXPECT_EQ(actual.graph_kind, expected.graph_kind);
   EXPECT_EQ(actual.node_types, expected.node_types);
   EXPECT_EQ(actual.flags, expected.flags);

   ASSERT_EQ(actual.edge_types.size(), expected.edge_types.size());
   for(size_t i = 0; i < actual.edge_types.size(); ++i) {
      EXPECT_EQ(actual.edge_types[i].src, expected.edge_types[i].src);
      EXPECT_EQ(actual.edge_types[i].rel, expected.edge_types[i].rel);
      EXPECT_EQ(actual.edge_types[i].dst, expected.edge_types[i].dst);
   }

   ASSERT_EQ(actual.node_tensors.size(), expected.node_tensors.size());
   for(size_t i = 0; i < actual.node_tensors.size(); ++i) {
      EXPECT_EQ(actual.node_tensors[i].node_type, expected.node_tensors[i].node_type);
      EXPECT_EQ(actual.node_tensors[i].attr, expected.node_tensors[i].attr);
      EXPECT_EQ(actual.node_tensors[i].key, expected.node_tensors[i].key);
   }

   ASSERT_EQ(actual.edge_tensors.size(), expected.edge_tensors.size());
   for(size_t i = 0; i < actual.edge_tensors.size(); ++i) {
      EXPECT_EQ(actual.edge_tensors[i].edge_type, expected.edge_tensors[i].edge_type);
      EXPECT_EQ(actual.edge_tensors[i].attr, expected.edge_tensors[i].attr);
      EXPECT_EQ(actual.edge_tensors[i].key, expected.edge_tensors[i].key);
      EXPECT_EQ(actual.edge_tensors[i].part, expected.edge_tensors[i].part);
   }
}

void expect_parts_equal(
   const BatchBuilder::BatchEncoding& actual,
   const BatchBuilder::BatchEncoding& expected
)
{
   EXPECT_EQ(actual.num_graphs, expected.num_graphs);
   EXPECT_EQ(actual.graph_kind, expected.graph_kind);
   EXPECT_EQ(actual.schema_flags, expected.schema_flags);
   EXPECT_EQ(actual.node_feature_dims, expected.node_feature_dims);
   EXPECT_EQ(actual.node_names, expected.node_names);
   EXPECT_EQ(actual.object_names, expected.object_names);
   EXPECT_EQ(actual.node_counts, expected.node_counts);
   EXPECT_EQ(actual.graph_attrs, expected.graph_attrs);

   expect_schema_equal(actual.schema, expected.schema);

   ASSERT_EQ(actual.columns.size(), expected.columns.size());
   for(const auto& [key, col] : actual.columns) {
      auto it = expected.columns.find(key);
      ASSERT_NE(it, expected.columns.end());
      EXPECT_EQ(col.dim, it->second.dim);

      const auto* actual_float = std::get_if< BatchBuilder::FloatCol >(&col.data);
      const auto* expected_float = std::get_if< BatchBuilder::FloatCol >(&it->second.data);
      if(actual_float != nullptr or expected_float != nullptr) {
         ASSERT_NE(actual_float, nullptr);
         ASSERT_NE(expected_float, nullptr);
         EXPECT_EQ(*actual_float, *expected_float);
         continue;
      }

      const auto* actual_long = std::get_if< BatchBuilder::LongCol >(&col.data);
      const auto* expected_long = std::get_if< BatchBuilder::LongCol >(&it->second.data);
      ASSERT_NE(actual_long, nullptr);
      ASSERT_NE(expected_long, nullptr);
      EXPECT_EQ(*actual_long, *expected_long);
   }

   ASSERT_EQ(actual.graph_fields.size(), expected.graph_fields.size());
   for(const auto& [key, field] : actual.graph_fields) {
      auto it = expected.graph_fields.find(key);
      ASSERT_NE(it, expected.graph_fields.end());
      EXPECT_EQ(field.spec.dtype, it->second.spec.dtype);
      EXPECT_EQ(field.spec.inc, it->second.spec.inc);
      EXPECT_EQ(field.spec.mode, it->second.spec.mode);
      EXPECT_EQ(field.spec.dim, it->second.spec.dim);
      EXPECT_EQ(field.spec.cat_dim, it->second.spec.cat_dim);
      EXPECT_EQ(field.values, it->second.values);
      EXPECT_EQ(field.ptr, it->second.ptr);
   }
}

TransitionDAG make_successor_dag(
   const mimir::search::State& root,
   const mimir::search::State& successor,
   const mimir::formalism::GroundAction& action
)
{
   TransitionDAG dag(root);
   dag.register_transition(root, successor, action);
   return dag;
}

}  // namespace

class FlatStreamEncoderTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(FlatStreamEncoderTest, FlatRelationAppendOnlyMatchesDirectBatch)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);
   (void) succ_action;

   FlatRelationEncoderEngine engine(ctx.problem->get_domain());
   FlatRelationStreamEncoder stream(engine);
   stream.append(ctx.root);
   stream.append(succ_state);

   const auto actual = stream.flush();

   BatchBuilder builder;
   builder.set_graph_kind("homo");
   engine.encode(ctx.root, builder);
   builder.next_graph();
   engine.encode(succ_state, builder);
   builder.next_graph();
   const auto expected = builder.build();

   expect_parts_equal(actual, expected);
}

TEST_P(FlatStreamEncoderTest, FlatRelationMutableRemoveUpdateAndReuseMatchDirectBatch)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);
   (void) succ_action;

   FlatRelationEncoderEngine engine(ctx.problem->get_domain());
   FlatRelationMutableStreamEncoder stream(engine);

   const auto root_id = stream.append(ctx.root);
   const auto succ_id = stream.append(succ_state);
   stream.update(root_id, succ_state);
   stream.remove(succ_id);

   auto actual = stream.flush();

   BatchBuilder builder;
   builder.set_graph_kind("homo");
   engine.encode(succ_state, builder);
   builder.next_graph();
   auto expected = builder.build();
   expect_parts_equal(actual, expected);

   FlatRelationMutableStreamEncoder reuse_stream(engine);
   reuse_stream.set_reuse_removed(true);
   const auto first_id = reuse_stream.append(ctx.root);
   const auto second_id = reuse_stream.append(succ_state);
   (void) second_id;
   reuse_stream.remove(first_id);
   const auto reused_id = reuse_stream.append(ctx.root);

   EXPECT_EQ(reused_id, first_id);

   actual = reuse_stream.flush();

   BatchBuilder reuse_builder;
   reuse_builder.set_graph_kind("homo");
   engine.encode(ctx.root, reuse_builder);
   reuse_builder.next_graph();
   engine.encode(succ_state, reuse_builder);
   reuse_builder.next_graph();
   expected = reuse_builder.build();
   expect_parts_equal(actual, expected);
}

TEST_P(FlatStreamEncoderTest, FlatHorizonMutableRemoveUpdateAndReuseMatchDirectBatch)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);
   const auto goals = mifrost_test::make_goal_inputs(ctx.problem);

   TransitionDAG empty_dag(ctx.root);
   const auto successor_dag = make_successor_dag(ctx.root, succ_state, succ_action);

   FlatHorizonEncoderEngine::Config config;
   config.ignore_actions = false;
   FlatHorizonEncoderEngine engine(ctx.problem->get_domain(), config);
   FlatHorizonStreamEncoder stream(engine);

   const auto empty_id = stream.append(ctx.root, empty_dag, goals);
   const auto full_id = stream.append(ctx.root, successor_dag, goals);
   stream.update(empty_id, ctx.root, successor_dag, goals);
   stream.remove(full_id);

   auto actual = stream.flush();

   BatchBuilder builder;
   builder.set_graph_kind("homo");
   engine.encode(ctx.root, successor_dag, goals, builder);
   builder.next_graph();
   auto expected = builder.build();
   expect_parts_equal(actual, expected);

   FlatHorizonStreamEncoder reuse_stream(engine);
   reuse_stream.set_reuse_removed(true);
   const auto first_id = reuse_stream.append(ctx.root, empty_dag, goals);
   const auto second_id = reuse_stream.append(ctx.root, successor_dag, goals);
   (void) second_id;
   reuse_stream.remove(first_id);
   const auto reused_id = reuse_stream.append(ctx.root, empty_dag, goals);

   EXPECT_EQ(reused_id, first_id);

   actual = reuse_stream.flush();

   BatchBuilder reuse_builder;
   reuse_builder.set_graph_kind("homo");
   engine.encode(ctx.root, empty_dag, goals, reuse_builder);
   reuse_builder.next_graph();
   engine.encode(ctx.root, successor_dag, goals, reuse_builder);
   reuse_builder.next_graph();
   expected = reuse_builder.build();
   expect_parts_equal(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   FlatStreamEncoderTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);
