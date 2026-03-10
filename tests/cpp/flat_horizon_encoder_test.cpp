#include "mifrost/core/flat_horizon_encoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <vector>

#include "mifrost/core/relation_formatter.hpp"
#include "mifrost/core/transition_dag.hpp"
#include "test_utils.hpp"

namespace {

using mifrost::BatchBuilder;
using mifrost::FlatHorizonEncoderEngine;
using mifrost::TransitionDAG;

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

std::optional< size_t >
relation_index_for(const std::vector< std::string >& relation_names, std::string_view relation_name)
{
   const auto it = std::find(relation_names.begin(), relation_names.end(), relation_name);
   if(it == relation_names.end()) {
      return std::nullopt;
   }
   return static_cast< size_t >(std::distance(relation_names.begin(), it));
}

size_t relation_slot_offset(
   std::span< const int64_t > relation_counts,
   std::span< const int64_t > relation_arities,
   size_t relation_idx
)
{
   size_t total = 0;
   for(size_t idx = 0; idx < relation_idx; ++idx) {
      total += static_cast< size_t >(relation_counts[idx] * relation_arities[idx]);
   }
   return total;
}

BatchBuilder::BatchEncoding encode_single(
   FlatHorizonEncoderEngine& engine,
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const mifrost::GoalInputs& goals
)
{
   BatchBuilder builder;
   engine.encode(root, dag, goals, builder);
   builder.next_graph();
   return builder.build();
}

}  // namespace

class FlatHorizonEncoderTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(FlatHorizonEncoderTest, EmitsStateTargetMetadataAndCarrierRows)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, succ_state, succ_action, int64_t{101});

   FlatHorizonEncoderEngine::Config config;
   config.ignore_actions = false;
   FlatHorizonEncoderEngine engine(ctx.problem->get_domain(), config);

   const auto encoding = encode_single(
      engine, ctx.root, dag, mifrost_test::make_goal_inputs(ctx.problem)
   );

   ASSERT_EQ(encoding.graph_kind, "homo");
   ASSERT_EQ(encoding.num_graphs, 1);
   ASSERT_TRUE(encoding.schema_flags.contains("flat_relations"));
   EXPECT_TRUE(encoding.schema_flags.at("flat_relations"));

   const auto& entity_names = encoding.node_names.at("entity");
   const auto target_entity_sizes = i64_field(encoding, "target_entity_sizes");
   const auto target_entity_indices = i64_field(encoding, "target_entity_indices");
   const auto target_entity_group_ids = i64_field(encoding, "target_entity_group_ids");
   const auto target_sizes = i64_field(encoding, "target_sizes");
   const auto target_positions = i64_field(encoding, "target_positions");
   const auto target_indices = i64_field(encoding, "target_indices");
   const auto target_candidate_ids = i64_field(encoding, "target_candidate_ids");
   const auto target_depths = i64_field(encoding, "target_depths");
   const auto relation_instance_sizes = i64_field(encoding, "relation_instance_sizes");

   ASSERT_EQ(target_entity_sizes, std::vector< int64_t >({2}));
   ASSERT_EQ(target_entity_indices.size(), 2u);
   EXPECT_EQ(target_entity_indices[0], static_cast< int64_t >(ctx.problem->get_objects().size()));
   EXPECT_EQ(target_entity_indices[1], target_entity_indices[0] + 1);
   EXPECT_EQ(target_entity_group_ids, std::vector< int64_t >({0, 0}));
   EXPECT_EQ(target_sizes, std::vector< int64_t >({1}));
   EXPECT_EQ(target_positions, std::vector< int64_t >({target_entity_indices[1]}));
   EXPECT_EQ(target_indices, std::vector< int64_t >({1}));
   EXPECT_EQ(target_candidate_ids, std::vector< int64_t >({101}));
   EXPECT_EQ(target_depths, std::vector< int64_t >({1}));
   ASSERT_EQ(relation_instance_sizes.size(), 1u);
   EXPECT_EQ(str_attr(encoding, "target_entity_groups"), std::vector< std::string >({"state"}));
   EXPECT_EQ(str_attr(encoding, "target_groups"), std::vector< std::string >({"state"}));
   EXPECT_EQ(entity_names[target_entity_indices[0]], "target:0");
   EXPECT_EQ(entity_names[target_entity_indices[1]], "target:1");
}

TEST_P(FlatHorizonEncoderTest, RelationsAnchorOnStateCarrierRows)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, succ_state, succ_action, int64_t{42});

   FlatHorizonEncoderEngine::Config config;
   config.ignore_actions = false;
   FlatHorizonEncoderEngine engine(ctx.problem->get_domain(), config);

   const auto encoding = encode_single(
      engine, ctx.root, dag, mifrost_test::make_goal_inputs(ctx.problem)
   );

   const auto relation_names = str_attr(encoding, "relation_names");
   const auto relation_arities = i64_attr(encoding, "relation_arities");
   const auto relation_counts = i64_field(encoding, "relation_counts");
   const auto relation_args = i64_field(encoding, "relation_args");
   const auto target_entity_indices = i64_field(encoding, "target_entity_indices");
   const auto target_positions = i64_field(encoding, "target_positions");
   std::unordered_set< int64_t > target_entities(
      target_entity_indices.begin(), target_entity_indices.end()
   );

   size_t cursor = 0;
   for(size_t relation_idx = 0; relation_idx < relation_names.size(); ++relation_idx) {
      const auto arity = static_cast< size_t >(relation_arities[relation_idx]);
      const auto instances = static_cast< size_t >(relation_counts[relation_idx]);
      for(size_t instance_idx = 0; instance_idx < instances; ++instance_idx) {
         ASSERT_LT(cursor, relation_args.size());
         EXPECT_TRUE(target_entities.contains(relation_args[cursor]))
            << "relation=" << relation_names[relation_idx];
         cursor += arity;
      }
   }
   EXPECT_EQ(cursor, relation_args.size());

   const auto action_relation_name = mifrost::RelationFormatter::format_action_schema(
      *succ_action->get_action()
   );
   const auto action_relation_idx = relation_index_for(relation_names, action_relation_name);
   ASSERT_TRUE(action_relation_idx.has_value());
   ASSERT_GT(relation_counts[*action_relation_idx], 0);
   const auto action_slot = relation_slot_offset(
      std::span{relation_counts}, std::span{relation_arities}, *action_relation_idx
   );
   ASSERT_LT(action_slot, relation_args.size());
   EXPECT_EQ(relation_args[action_slot], target_positions.front());
}

TEST_P(FlatHorizonEncoderTest, LGANCandidateRowsEmitPackedFields)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, succ_state, succ_action, int64_t{101});

   FlatHorizonEncoderEngine::Config config;
   config.ignore_actions = false;
   config.include_lgan_edges = true;
   FlatHorizonEncoderEngine engine(ctx.problem->get_domain(), config);

   const auto encoding = encode_single(
      engine, ctx.root, dag, mifrost_test::make_goal_inputs(ctx.problem)
   );

   const auto relation_counts = i64_field(encoding, "relation_counts");
   const auto relation_instance_sizes = i64_field(encoding, "relation_instance_sizes");
   const auto lgan_tn_sizes = i64_field(encoding, "lgan_tn_sizes");
   const auto lgan_tn_relation_indices = i64_field(encoding, "lgan_tn_relation_indices");
   const auto lgan_tn_entity_indices = i64_field(encoding, "lgan_tn_entity_indices");
   const auto lgan_nn_relation_indices = i64_field(encoding, "lgan_nn_relation_indices");
   const auto lgan_nn_entity_indices = i64_field(encoding, "lgan_nn_entity_indices");
   const auto lgan_rr_src_relation_indices = i64_field(encoding, "lgan_rr_src_relation_indices");
   const auto lgan_rr_dst_relation_indices = i64_field(encoding, "lgan_rr_dst_relation_indices");
   const auto target_positions = i64_field(encoding, "target_positions");

   ASSERT_EQ(relation_instance_sizes.size(), 1u);
   EXPECT_EQ(
      relation_instance_sizes.front(),
      std::accumulate(relation_counts.begin(), relation_counts.end(), int64_t{0})
   );
   ASSERT_EQ(
      lgan_tn_sizes,
      std::vector< int64_t >({static_cast< int64_t >(lgan_tn_relation_indices.size())})
   );
   ASSERT_EQ(lgan_tn_relation_indices.size(), lgan_tn_entity_indices.size());
   ASSERT_EQ(lgan_nn_relation_indices.size(), lgan_nn_entity_indices.size());
   ASSERT_EQ(lgan_rr_src_relation_indices.size(), lgan_rr_dst_relation_indices.size());
   ASSERT_FALSE(lgan_tn_relation_indices.empty());

   std::unordered_set< int64_t > candidate_rows(target_positions.begin(), target_positions.end());
   for(const auto entity_index : lgan_tn_entity_indices) {
      EXPECT_TRUE(candidate_rows.contains(entity_index));
   }
}

TEST_P(FlatHorizonEncoderTest, LGANRejectsMissingCandidateRows)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   TransitionDAG dag(ctx.root);

   FlatHorizonEncoderEngine::Config config;
   config.ignore_actions = false;
   config.include_lgan_edges = true;
   FlatHorizonEncoderEngine engine(ctx.problem->get_domain(), config);

   EXPECT_THROW(
      {
         try {
            BatchBuilder builder;
            engine.encode(ctx.root, dag, mifrost_test::make_goal_inputs(ctx.problem), builder);
            builder.next_graph();
            (void) builder.build();
         } catch(const std::invalid_argument& e) {
            EXPECT_NE(
               std::string(e.what()).find("requires surviving candidate state rows"),
               std::string::npos
            );
            throw;
         }
      },
      std::invalid_argument
   );
}

TEST_P(FlatHorizonEncoderTest, RejectsPartialExplicitCandidateIds)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   std::vector< std::pair< mimir::search::State, mimir::formalism::GroundAction > > successors;
   for(const auto& action : ctx.actions) {
      auto [succ_state, _metric] = ctx.repo->get_or_create_successor_state(
         ctx.root, action, ctx.root_metric
      );
      if(succ_state.get_index() == ctx.root.get_index()) {
         continue;
      }
      const bool seen = std::ranges::any_of(successors, [&](const auto& existing) {
         return existing.first.get_index() == succ_state.get_index();
      });
      if(not seen) {
         successors.emplace_back(succ_state, action);
      }
      if(successors.size() >= 2) {
         break;
      }
   }
   if(successors.size() < 2) {
      GTEST_SKIP() << "Need two distinct successors for partial candidate-id validation.";
   }

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, successors[0].first, successors[0].second, int64_t{7});
   dag.register_transition(ctx.root, successors[1].first, successors[1].second, std::nullopt);

   FlatHorizonEncoderEngine engine(ctx.problem->get_domain());

   EXPECT_THROW(
      {
         try {
            BatchBuilder builder;
            engine.encode(ctx.root, dag, mifrost_test::make_goal_inputs(ctx.problem), builder);
         } catch(const std::invalid_argument& e) {
            EXPECT_NE(
               std::string(e.what()).find("missing candidate_id for target node index"),
               std::string::npos
            );
            throw;
         }
      },
      std::invalid_argument
   );
}

TEST_P(FlatHorizonEncoderTest, RejectsDuplicateExplicitCandidateIds)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   std::vector< std::pair< mimir::search::State, mimir::formalism::GroundAction > > successors;
   for(const auto& action : ctx.actions) {
      auto [succ_state, _metric] = ctx.repo->get_or_create_successor_state(
         ctx.root, action, ctx.root_metric
      );
      if(succ_state.get_index() == ctx.root.get_index()) {
         continue;
      }
      const bool seen = std::ranges::any_of(successors, [&](const auto& existing) {
         return existing.first.get_index() == succ_state.get_index();
      });
      if(not seen) {
         successors.emplace_back(succ_state, action);
      }
      if(successors.size() >= 2) {
         break;
      }
   }
   if(successors.size() < 2) {
      GTEST_SKIP() << "Need two distinct successors for duplicate candidate-id validation.";
   }

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, successors[0].first, successors[0].second, int64_t{9});
   dag.register_transition(ctx.root, successors[1].first, successors[1].second, int64_t{9});

   FlatHorizonEncoderEngine engine(ctx.problem->get_domain());

   EXPECT_THROW(
      {
         try {
            BatchBuilder builder;
            engine.encode(ctx.root, dag, mifrost_test::make_goal_inputs(ctx.problem), builder);
         } catch(const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("duplicate candidate_id"), std::string::npos);
            throw;
         }
      },
      std::invalid_argument
   );
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   FlatHorizonEncoderTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);
