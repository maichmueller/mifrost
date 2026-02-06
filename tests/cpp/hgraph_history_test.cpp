#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include "mifrost/core/default_relations.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/relation_formatter.hpp"
#include "test_utils.hpp"

namespace {

using EdgePairs = mifrost_test::EdgePairs;

std::string edge_key(std::string_view src, std::string_view rel, std::string_view dst)
{
   std::string key;
   key.reserve(src.size() + rel.size() + dst.size() + 2);
   key.append(src);
   key.push_back('|');
   key.append(rel);
   key.push_back('|');
   key.append(dst);
   return key;
}

std::vector< mifrost::GoalInputs::AnyGoalLiteral > collect_goals(
   const mimir::formalism::Problem& problem
)
{
   std::vector< mifrost::GoalInputs::AnyGoalLiteral > goals;
   for(const auto& goal : problem->get_goal_literals< mimir::formalism::StaticTag >()) {
      goals.emplace_back(goal);
   }
   for(const auto& goal : problem->get_goal_literals< mimir::formalism::FluentTag >()) {
      goals.emplace_back(goal);
   }
   for(const auto& goal : problem->get_goal_literals< mimir::formalism::DerivedTag >()) {
      goals.emplace_back(goal);
   }
   return goals;
}

}  // namespace

class HGraphHistoryTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(HGraphHistoryTest, HistoryNodesAndEdgesPresent)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto goals = collect_goals(ctx.problem);
   if(goals.empty()) {
      GTEST_SKIP() << "No goal literals available for history encoding.";
   }

   std::vector< mifrost::HGraphEncoderEngine::HistorySubgoal > history;
   history.push_back({-1, {goals.front()}});
   if(goals.size() > 1) {
      history.push_back({-2, {goals[1]}});
   }

   mifrost::GoalInputs inputs = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain());
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("hetero");
   engine.encode(ctx.root, inputs, {}, history, std::nullopt, builder);

   const auto history_it = builder.node_names.find("history");
   ASSERT_NE(history_it, builder.node_names.end());
   const auto& history_names = history_it->second;
   ASSERT_EQ(history_names.size(), history.size());

   const auto dt_it = builder.columns.find("history/history_dt");
   ASSERT_NE(dt_it, builder.columns.end()) << "history/history_dt column missing.";
   const auto& dt_col = std::get< mifrost::BatchBuilder::FloatCol >(dt_it->second.data);
   ASSERT_EQ(dt_col.size(), history.size());
   if(history.size() == 1) {
      EXPECT_EQ(dt_col[0], -1.0f);
   } else {
      std::vector< float > expected = {-2.0f, -1.0f};
      std::vector< float > actual = {dt_col[0], dt_col[1]};
      EXPECT_EQ(actual, expected);
   }

   auto expected_history = history;
   std::ranges::stable_sort(expected_history, [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
   });

   const auto index_map = mifrost_test::build_index_map(builder);
   const auto history_index = index_map.at("history");

   for(size_t entry_idx = 0; entry_idx < expected_history.size(); ++entry_idx) {
      const auto& [time_delta, subgoals] = expected_history[entry_idx];
      const auto history_name = history_names[entry_idx];
      const auto history_idx_it = history_index.find(history_name);
      ASSERT_NE(history_idx_it, history_index.end());
      const int64_t history_idx = history_idx_it->second;

      for(const auto& goal : subgoals) {
         std::visit(
            [&]< typename LiteralT >(const LiteralT& literal) {
               using Tag = std::remove_pointer_t< LiteralT >::Type;
               const auto predicate = literal->get_atom()->get_predicate();
               const auto node_type = mifrost::RelationFormatter::format_predicate(
                  predicate, std::nullopt, std::nullopt, literal->get_polarity()
               );
               const auto node_key = mifrost::RelationFormatter::format_literal< Tag >(
                  literal, std::nullopt
               );

               auto type_it = index_map.find(node_type);
               ASSERT_NE(type_it, index_map.end());
               auto node_it = type_it->second.find(node_key);
               ASSERT_NE(node_it, type_it->second.end());

               const auto rel = std::string(mifrost::defaults::history_link_relation);
               const auto forward_key = edge_key(node_type, rel, "history");
               const auto reverse_key = edge_key("history", rel, node_type);

               EdgePairs forward_pairs = mifrost_test::edge_pairs_for(builder, forward_key);
               EdgePairs reverse_pairs = mifrost_test::edge_pairs_for(builder, reverse_key);

               const int64_t literal_idx = node_it->second;
               bool has_forward = std::ranges::any_of(forward_pairs, [&](const auto& pair) {
                  return pair.first == literal_idx and pair.second == history_idx;
               });
               bool has_reverse = std::ranges::any_of(reverse_pairs, [&](const auto& pair) {
                  return pair.first == history_idx and pair.second == literal_idx;
               });

               EXPECT_TRUE(has_forward);
               EXPECT_TRUE(has_reverse);
            },
            goal
         );
      }
   }
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   HGraphHistoryTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

TEST(HGraphHistoryOverrideTest, CustomHistoryRelation)
{
   auto ctx = mifrost_test::make_context("blocks", "probBLOCKS-4-0");
   auto goals = collect_goals(ctx.problem);
   ASSERT_FALSE(goals.empty());

   std::vector< mifrost::HGraphEncoderEngine::HistorySubgoal > history;
   history.push_back({-1, {goals.front()}});

   mifrost::GoalInputs inputs = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::HGraphEncoderEngine::Config config;
   config.history_link_relation = "_custom_history_";
   mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain(), config);
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("hetero");
   engine.encode(ctx.root, inputs, {}, history, std::nullopt, builder);

   const auto index_map = mifrost_test::build_index_map(builder);
   const auto history_index = index_map.at("history");
   const auto history_name = builder.node_names.at("history").front();
   const auto history_idx = history_index.at(history_name);

   const auto& goal_variant = history.front().second.front();
   std::visit(
      [&]< typename LiteralT >(const LiteralT& literal) {
         using Tag = std::remove_pointer_t< LiteralT >::Type;
         const auto predicate = literal->get_atom()->get_predicate();
         const auto node_type = mifrost::RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, literal->get_polarity()
         );
         const auto node_key = mifrost::RelationFormatter::format_literal< Tag >(
            literal, std::nullopt
         );
         const auto literal_idx = index_map.at(node_type).at(node_key);

         const auto forward_key = edge_key(node_type, config.history_link_relation, "history");
         const auto reverse_key = edge_key("history", config.history_link_relation, node_type);

         EdgePairs forward_pairs = mifrost_test::edge_pairs_for(builder, forward_key);
         EdgePairs reverse_pairs = mifrost_test::edge_pairs_for(builder, reverse_key);

         EXPECT_TRUE(std::any_of(forward_pairs.begin(), forward_pairs.end(), [&](const auto& pair) {
            return pair.first == literal_idx and pair.second == history_idx;
         }));
         EXPECT_TRUE(std::any_of(reverse_pairs.begin(), reverse_pairs.end(), [&](const auto& pair) {
            return pair.first == history_idx and pair.second == literal_idx;
         }));
      },
      goal_variant
   );
}
