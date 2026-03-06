#include "mifrost/core/flat_relation_encoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "test_utils.hpp"

namespace {

using mifrost::BatchBuilder;

const std::vector< int64_t >&
i64_field(const BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   return std::get< std::vector< int64_t > >(encoding.graph_fields.at(std::string(key)).values);
}

const std::vector< int64_t >&
i64_attr(const BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   return std::get< std::vector< int64_t > >(encoding.graph_attrs.at(std::string(key)));
}

const std::vector< std::string >&
str_attr(const BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   return std::get< std::vector< std::string > >(encoding.graph_attrs.at(std::string(key)));
}

const std::vector< std::string >&
str_vec_attr(const BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   return std::get< std::vector< std::string > >(encoding.graph_attrs.at(std::string(key)));
}

std::vector< int64_t >
row_major_slice(const std::vector< int64_t >& values, size_t row, size_t width)
{
   const auto begin = values.begin() + static_cast< ptrdiff_t >(row * width);
   const auto end = begin + static_cast< ptrdiff_t >(width);
   return std::vector< int64_t >(begin, end);
}

size_t
total_slots(std::span< const int64_t > relation_counts, std::span< const int64_t > relation_arities)
{
   size_t total = 0;
   for(size_t idx = 0; idx < relation_counts.size(); ++idx) {
      total += static_cast< size_t >(relation_counts[idx] * relation_arities[idx]);
   }
   return total;
}

BatchBuilder::BatchEncoding
encode_single(mifrost::FlatRelationEncoderEngine& engine, const mimir::search::State& state)
{
   BatchBuilder builder;
   engine.encode(state, builder);
   builder.next_graph();
   return builder.build();
}

BatchBuilder::BatchEncoding encode_single(
   mifrost::FlatRelationEncoderEngine& engine,
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions
)
{
   BatchBuilder builder;
   engine.encode(state, actions, builder);
   builder.next_graph();
   return builder.build();
}

}  // namespace

class FlatRelationEncoderTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(FlatRelationEncoderTest, RelationNamesFollowRelationDictOrder)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);

   mifrost::FlatRelationEncoderEngine engine(ctx.problem->get_domain());

   const auto& actual_names = engine.get_relation_names();
   const auto& actual_arities = engine.get_relation_arities();
   ASSERT_EQ(actual_names.size(), actual_arities.size());

   std::unordered_set< std::string > actual_set(actual_names.begin(), actual_names.end());
   std::vector< std::string > expected_names;
   std::vector< int64_t > expected_arities;
   for(const auto& [name, arity] : engine.get_relation_dict().arity) {
      if(actual_set.contains(name)) {
         expected_names.push_back(name);
         expected_arities.push_back(arity);
      }
   }

   EXPECT_EQ(actual_names, expected_names);
   EXPECT_EQ(actual_arities, expected_arities);
}

TEST_P(FlatRelationEncoderTest, SingleGraphFieldsAreConsistent)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);

   mifrost::FlatRelationEncoderEngine engine(ctx.problem->get_domain());
   const auto encoding = encode_single(engine, ctx.root);

   ASSERT_EQ(encoding.graph_kind, "homo");
   ASSERT_EQ(encoding.num_graphs, 1);
   ASSERT_TRUE(encoding.schema_flags.contains("flat_relations"));
   EXPECT_TRUE(encoding.schema_flags.at("flat_relations"));

   const auto relation_names = str_attr(encoding, "relation_names");
   const auto relation_arities = i64_attr(encoding, "relation_arities");
   const auto relation_sources = str_attr(encoding, "relation_sources");
   ASSERT_EQ(relation_names.size(), relation_arities.size());
   ASSERT_EQ(relation_names.size(), relation_sources.size());

   const auto& entity_names = encoding.node_names.at("entity");
   const auto& object_names = encoding.object_names;
   EXPECT_EQ(object_names, entity_names);

   const auto node_sizes = i64_field(encoding, "node_sizes");
   const auto object_sizes = i64_field(encoding, "object_sizes");
   const auto object_indices = i64_field(encoding, "object_indices");
   const auto target_entity_sizes = i64_field(encoding, "target_entity_sizes");
   const auto target_entity_indices = i64_field(encoding, "target_entity_indices");
   const auto relation_counts = i64_field(encoding, "relation_counts");
   const auto relation_args = i64_field(encoding, "relation_args");

   ASSERT_EQ(node_sizes.size(), 1u);
   ASSERT_EQ(object_sizes.size(), 1u);
   ASSERT_EQ(target_entity_sizes.size(), 1u);
   ASSERT_EQ(node_sizes.front(), static_cast< int64_t >(entity_names.size()));
   ASSERT_EQ(object_sizes.front(), static_cast< int64_t >(object_names.size()));
   ASSERT_EQ(target_entity_sizes.front(), 0);
   ASSERT_EQ(object_indices.size(), object_names.size());
   ASSERT_TRUE(target_entity_indices.empty());
   for(size_t idx = 0; idx < object_indices.size(); ++idx) {
      EXPECT_EQ(object_indices[idx], static_cast< int64_t >(idx));
   }

   ASSERT_EQ(relation_counts.size(), relation_names.size());
   const size_t expected_slots = total_slots(
      std::span{relation_counts}, std::span{relation_arities}
   );
   EXPECT_EQ(relation_args.size(), expected_slots);
   for(const auto arg : relation_args) {
      EXPECT_GE(arg, 0);
      EXPECT_LT(arg, node_sizes.front());
   }
}

TEST_P(FlatRelationEncoderTest, BatchEncodingMatchesSingleGraphSlices)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [succ_state, _succ_action] = mifrost_test::find_successor(ctx);

   mifrost::FlatRelationEncoderEngine engine(ctx.problem->get_domain());

   const auto single_root = encode_single(engine, ctx.root);
   const auto single_succ = encode_single(engine, succ_state);

   BatchBuilder builder;
   engine.encode(ctx.root, builder);
   builder.next_graph();
   engine.encode(succ_state, builder);
   builder.next_graph();
   const auto batch = builder.build();

   ASSERT_EQ(batch.num_graphs, 2);
   ASSERT_TRUE(batch.schema_flags.contains("flat_relations"));
   EXPECT_TRUE(batch.schema_flags.at("flat_relations"));

   const auto relation_arities = i64_attr(batch, "relation_arities");
   const size_t relation_count = relation_arities.size();

   const auto batch_node_sizes = i64_field(batch, "node_sizes");
   const auto batch_object_sizes = i64_field(batch, "object_sizes");
   const auto batch_object_indices = i64_field(batch, "object_indices");
   const auto batch_target_entity_sizes = i64_field(batch, "target_entity_sizes");
   const auto batch_target_entity_indices = i64_field(batch, "target_entity_indices");
   const auto batch_relation_counts = i64_field(batch, "relation_counts");
   const auto batch_relation_args = i64_field(batch, "relation_args");

   ASSERT_EQ(batch_node_sizes.size(), 2u);
   ASSERT_EQ(batch_object_sizes.size(), 2u);
   ASSERT_EQ(batch_target_entity_sizes.size(), 2u);
   ASSERT_TRUE(batch_target_entity_indices.empty());
   ASSERT_EQ(batch_relation_counts.size(), 2u * relation_count);

   const auto root_node_sizes = i64_field(single_root, "node_sizes");
   const auto succ_node_sizes = i64_field(single_succ, "node_sizes");
   const auto root_object_sizes = i64_field(single_root, "object_sizes");
   const auto succ_object_sizes = i64_field(single_succ, "object_sizes");
   const auto root_target_entity_sizes = i64_field(single_root, "target_entity_sizes");
   const auto succ_target_entity_sizes = i64_field(single_succ, "target_entity_sizes");
   EXPECT_EQ(batch_node_sizes[0], root_node_sizes.front());
   EXPECT_EQ(batch_node_sizes[1], succ_node_sizes.front());
   EXPECT_EQ(batch_object_sizes[0], root_object_sizes.front());
   EXPECT_EQ(batch_object_sizes[1], succ_object_sizes.front());
   EXPECT_EQ(batch_target_entity_sizes[0], root_target_entity_sizes.front());
   EXPECT_EQ(batch_target_entity_sizes[1], succ_target_entity_sizes.front());

   const auto root_counts = i64_field(single_root, "relation_counts");
   const auto succ_counts = i64_field(single_succ, "relation_counts");
   EXPECT_EQ(row_major_slice(batch_relation_counts, 0, relation_count), root_counts);
   EXPECT_EQ(row_major_slice(batch_relation_counts, 1, relation_count), succ_counts);

   const size_t root_slots = total_slots(std::span{root_counts}, std::span{relation_arities});
   const size_t succ_slots = total_slots(std::span{succ_counts}, std::span{relation_arities});
   ASSERT_EQ(batch_relation_args.size(), root_slots + succ_slots);

   const auto root_args = i64_field(single_root, "relation_args");
   const auto succ_args = i64_field(single_succ, "relation_args");
   EXPECT_EQ(
      std::vector< int64_t >(batch_relation_args.begin(), batch_relation_args.begin() + root_slots),
      root_args
   );

   std::vector< int64_t > expected_succ_args = succ_args;
   const int64_t succ_offset = batch_node_sizes[0];
   for(auto& value : expected_succ_args) {
      value += succ_offset;
   }
   EXPECT_EQ(
      std::vector< int64_t >(
         batch_relation_args.begin() + static_cast< ptrdiff_t >(root_slots),
         batch_relation_args.end()
      ),
      expected_succ_args
   );

   const auto root_object_indices = i64_field(single_root, "object_indices");
   const auto succ_object_indices = i64_field(single_succ, "object_indices");
   EXPECT_EQ(
      std::vector< int64_t >(
         batch_object_indices.begin(),
         batch_object_indices.begin() + static_cast< ptrdiff_t >(root_object_indices.size())
      ),
      root_object_indices
   );

   std::vector< int64_t > expected_succ_object_indices = succ_object_indices;
   for(auto& value : expected_succ_object_indices) {
      value += succ_offset;
   }
   EXPECT_EQ(
      std::vector< int64_t >(
         batch_object_indices.begin() + static_cast< ptrdiff_t >(root_object_indices.size()),
         batch_object_indices.end()
      ),
      expected_succ_object_indices
   );
}

TEST_P(FlatRelationEncoderTest, ExplicitActionsPopulateTargetEntityFields)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   mifrost::FlatRelationEncoderEngine engine(ctx.problem->get_domain());

   const std::array< mimir::formalism::GroundAction, 1 > actions = {succ_action};
   const auto single_root = encode_single(engine, ctx.root, std::span{actions});
   const auto single_succ = encode_single(engine, succ_state, std::span{actions});

   const auto& root_entity_names = single_root.node_names.at("entity");
   const auto& root_object_names = single_root.object_names;
   ASSERT_EQ(root_entity_names.size(), root_object_names.size() + 1);
   EXPECT_EQ(root_entity_names.back(), mifrost::RelationFormatter::format_action(succ_action));

   const auto root_object_sizes = i64_field(single_root, "object_sizes");
   const auto root_target_entity_sizes = i64_field(single_root, "target_entity_sizes");
   const auto root_target_entity_indices = i64_field(single_root, "target_entity_indices");
   ASSERT_EQ(root_object_sizes.size(), 1u);
   ASSERT_EQ(root_target_entity_sizes.size(), 1u);
   ASSERT_EQ(root_target_entity_indices.size(), 1u);
   EXPECT_EQ(root_target_entity_sizes.front(), 1);
   EXPECT_EQ(root_target_entity_indices.front(), root_object_sizes.front());

   const auto relation_names = str_attr(single_root, "relation_names");
   const auto relation_arities = i64_attr(single_root, "relation_arities");
   const auto root_relation_counts = i64_field(single_root, "relation_counts");
   const auto root_relation_args = i64_field(single_root, "relation_args");
   const auto action_relation_name = mifrost::RelationFormatter::format_action_schema(
      *succ_action->get_action()
   );

   auto relation_it = std::find(relation_names.begin(), relation_names.end(), action_relation_name);
   ASSERT_NE(relation_it, relation_names.end());
   const size_t relation_idx = static_cast< size_t >(
      std::distance(relation_names.begin(), relation_it)
   );
   ASSERT_EQ(root_relation_counts[relation_idx], 1);
   ASSERT_EQ(
      relation_arities[relation_idx],
      static_cast< int64_t >(succ_action->get_action()->get_arity()) + 1
   );

   size_t slot_start = 0;
   for(size_t idx = 0; idx < relation_idx; ++idx) {
      slot_start += static_cast< size_t >(root_relation_counts[idx] * relation_arities[idx]);
   }
   const size_t slot_end = slot_start + static_cast< size_t >(relation_arities[relation_idx]);
   ASSERT_LE(slot_end, root_relation_args.size());

   std::vector< int64_t > action_args(
      root_relation_args.begin() + static_cast< ptrdiff_t >(slot_start),
      root_relation_args.begin() + static_cast< ptrdiff_t >(slot_end)
   );
   ASSERT_FALSE(action_args.empty());
   EXPECT_EQ(action_args.front(), root_target_entity_indices.front());

   BatchBuilder builder;
   engine.encode(ctx.root, std::span{actions}, builder);
   builder.next_graph();
   engine.encode(succ_state, std::span{actions}, builder);
   builder.next_graph();
   const auto batch = builder.build();

   const auto batch_node_sizes = i64_field(batch, "node_sizes");
   const auto batch_target_entity_sizes = i64_field(batch, "target_entity_sizes");
   const auto batch_target_entity_indices = i64_field(batch, "target_entity_indices");
   const auto batch_relation_counts = i64_field(batch, "relation_counts");
   const auto batch_relation_args = i64_field(batch, "relation_args");

   ASSERT_EQ(batch_node_sizes.size(), 2u);
   ASSERT_EQ(batch_target_entity_sizes.size(), 2u);
   ASSERT_EQ(batch_target_entity_sizes[0], 1);
   ASSERT_EQ(batch_target_entity_sizes[1], 1);

   const auto succ_target_entity_indices = i64_field(single_succ, "target_entity_indices");
   ASSERT_EQ(succ_target_entity_indices.size(), 1u);
   ASSERT_EQ(batch_target_entity_indices.size(), 2u);
   EXPECT_EQ(batch_target_entity_indices[0], root_target_entity_indices[0]);
   EXPECT_EQ(batch_target_entity_indices[1], succ_target_entity_indices[0] + batch_node_sizes[0]);

   const size_t relation_count = relation_names.size();
   ASSERT_EQ(batch_relation_counts.size(), 2u * relation_count);
   const auto succ_counts = i64_field(single_succ, "relation_counts");
   EXPECT_EQ(row_major_slice(batch_relation_counts, 0, relation_count), root_relation_counts);
   EXPECT_EQ(row_major_slice(batch_relation_counts, 1, relation_count), succ_counts);

   const size_t root_slots = total_slots(
      std::span{root_relation_counts}, std::span{relation_arities}
   );
   const auto succ_args = i64_field(single_succ, "relation_args");
   ASSERT_EQ(batch_relation_args.size(), root_slots + succ_args.size());

   std::vector< int64_t > expected_succ_args = succ_args;
   for(auto& value : expected_succ_args) {
      value += batch_node_sizes[0];
   }
   EXPECT_EQ(
      std::vector< int64_t >(
         batch_relation_args.begin() + static_cast< ptrdiff_t >(root_slots),
         batch_relation_args.end()
      ),
      expected_succ_args
   );
}

TEST_P(FlatRelationEncoderTest, ActionTargetsEmitSharedTargetMetadata)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   mifrost::FlatRelationEncoderEngine::Config config;
   config.target_sources = {mifrost::TargetSource::Actions};
   mifrost::FlatRelationEncoderEngine engine(ctx.problem->get_domain(), config);

   const std::array< mimir::formalism::GroundAction, 2 > actions = {succ_action, succ_action};
   const auto single = encode_single(engine, ctx.root, std::span{actions});

   const auto target_sizes = i64_field(single, "target_sizes");
   const auto target_positions = i64_field(single, "target_positions");
   const auto target_indices = i64_field(single, "target_indices");
   const auto target_candidate_ids = i64_field(single, "target_candidate_ids");
   const auto target_group_ids = i64_field(single, "target_group_ids");
   const auto target_entity_indices = i64_field(single, "target_entity_indices");
   const auto& target_names = str_vec_attr(single, "target_names");
   const auto& target_groups = str_vec_attr(single, "target_groups");

   ASSERT_EQ(target_sizes, (std::vector< int64_t >{2}));
   ASSERT_EQ(target_positions.size(), 2u);
   EXPECT_EQ(target_positions[0], target_entity_indices.front());
   EXPECT_EQ(target_positions[1], target_entity_indices.front());
   EXPECT_EQ(target_indices, (std::vector< int64_t >{0, 1}));
   EXPECT_EQ(target_candidate_ids, (std::vector< int64_t >{0, 1}));
   EXPECT_EQ(target_group_ids, (std::vector< int64_t >{0, 0}));
   EXPECT_EQ(target_groups, (std::vector< std::string >{"action"}));
   ASSERT_EQ(target_names.size(), 2u);
   EXPECT_EQ(target_names[0], mifrost::RelationFormatter::format_action(succ_action));
   EXPECT_EQ(target_names[1], mifrost::RelationFormatter::format_action(succ_action));

   BatchBuilder builder;
   engine.encode(ctx.root, std::span{actions}, builder);
   builder.next_graph();
   engine.encode(succ_state, std::span< const mimir::formalism::GroundAction >{}, builder);
   builder.next_graph();
   const auto batch = builder.build();

   const auto batch_target_sizes = i64_field(batch, "target_sizes");
   const auto batch_target_positions = i64_field(batch, "target_positions");
   const auto batch_target_indices = i64_field(batch, "target_indices");
   const auto batch_target_candidate_ids = i64_field(batch, "target_candidate_ids");
   const auto batch_target_group_ids = i64_field(batch, "target_group_ids");

   ASSERT_EQ(batch_target_sizes, (std::vector< int64_t >{2, 0}));
   ASSERT_EQ(batch_target_positions.size(), 2u);
   EXPECT_EQ(batch_target_positions[0], target_positions[0]);
   EXPECT_EQ(batch_target_positions[1], target_positions[1]);
   EXPECT_EQ(batch_target_indices, target_indices);
   EXPECT_EQ(batch_target_candidate_ids, target_candidate_ids);
   EXPECT_EQ(batch_target_group_ids, target_group_ids);
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   FlatRelationEncoderTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);
