#include "mifrost/core/encoders/flat/flat_horizon_encoder.hpp"

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

#include "mifrost/core/encoders/common/relation_formatter.hpp"
#include "mifrost/core/encoders/common/transition_dag.hpp"
#include "test_utils.hpp"

namespace {

using mifrost::BatchBuilder;
using mifrost::FlatHorizonEncoderEngine;
using mifrost::TransitionDAG;

const std::vector< int64_t >&
i64_field(const BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   return std::get< std::vector< int64_t > >(encoding.graph_fields.at(key).values);
}

const std::vector< int64_t >&
i64_attr(const BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   return std::get< std::vector< int64_t > >(encoding.graph_attrs.at(key));
}

const std::vector< std::string >&
str_attr(const BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   return std::get< std::vector< std::string > >(encoding.graph_attrs.at(key));
}

std::optional< size_t >
relation_index_for(const std::vector< std::string >& relation_names, std::string_view relation_name)
{
   const auto it = std::ranges::find(relation_names, relation_name);
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

   ASSERT_EQ(encoding.graph_kind, "flat");
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

   ASSERT_EQ(target_entity_sizes, std::vector< int64_t >({1}));
   ASSERT_EQ(target_entity_indices.size(), 1u);
   EXPECT_EQ(
      target_entity_indices[0], static_cast< int64_t >(ctx.problem->get_objects().size() + 1)
   );
   EXPECT_EQ(target_entity_group_ids, std::vector< int64_t >({0}));
   EXPECT_EQ(target_sizes, std::vector< int64_t >({1}));
   EXPECT_EQ(target_positions, std::vector< int64_t >({target_entity_indices[0]}));
   EXPECT_EQ(target_indices, std::vector< int64_t >({1}));
   EXPECT_EQ(target_candidate_ids, std::vector< int64_t >({101}));
   EXPECT_EQ(target_depths, std::vector< int64_t >({1}));
   ASSERT_EQ(relation_instance_sizes.size(), 1u);
   EXPECT_EQ(str_attr(encoding, "target_entity_groups"), std::vector< std::string >({"state"}));
   EXPECT_EQ(str_attr(encoding, "target_groups"), std::vector< std::string >({"state"}));
   EXPECT_EQ(entity_names[ctx.problem->get_objects().size()], "_root_state_");
   EXPECT_EQ(entity_names[target_entity_indices[0]], "target:1");
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
   std::unordered_set< int64_t > object_rows;
   const auto object_indices = i64_field(encoding, "object_indices");
   object_rows.insert(object_indices.begin(), object_indices.end());
   bool saw_base_relation = false;
   bool saw_state_relation = false;

   size_t cursor = 0;
   for(size_t relation_idx = 0; relation_idx < relation_names.size(); ++relation_idx) {
      const auto arity = static_cast< size_t >(relation_arities[relation_idx]);
      const auto instances = static_cast< size_t >(relation_counts[relation_idx]);
      for(size_t instance_idx = 0; instance_idx < instances; ++instance_idx) {
         ASSERT_LT(cursor, relation_args.size());
         const auto anchor_row = relation_args[cursor];
         if(relation_names[relation_idx].find("[state]") != std::string::npos
            || relation_names[relation_idx]
                  == mifrost::RelationFormatter::format_action_schema(*succ_action->get_action())) {
            saw_state_relation = true;
            EXPECT_TRUE(target_entities.contains(anchor_row))
               << "relation=" << relation_names[relation_idx];
         } else {
            saw_base_relation = true;
            EXPECT_TRUE(object_rows.contains(anchor_row))
               << "relation=" << relation_names[relation_idx];
         }
         cursor += arity;
      }
   }
   EXPECT_EQ(cursor, relation_args.size());
   EXPECT_TRUE(saw_base_relation);
   EXPECT_TRUE(saw_state_relation);

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

TEST_P(FlatHorizonEncoderTest, PredicateVirtualNodesFollowStateSlotMetadata)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, succ_state, succ_action, int64_t{42});

   FlatHorizonEncoderEngine::Config config;
   config.ignore_actions = false;
   config.use_predicate_virtual_nodes = true;
   FlatHorizonEncoderEngine engine(ctx.problem->get_domain(), config);
   FlatHorizonEncoderEngine base_engine(
      ctx.problem->get_domain(),
      FlatHorizonEncoderEngine::Config{
         .ignore_actions = false,
      }
   );

   const auto encoding = encode_single(
      engine, ctx.root, dag, mifrost_test::make_goal_inputs(ctx.problem)
   );
   const auto base = encode_single(
      base_engine, ctx.root, dag, mifrost_test::make_goal_inputs(ctx.problem)
   );

   const auto relation_names = str_attr(encoding, "relation_names");
   const auto relation_arities = i64_attr(encoding, "relation_arities");
   const auto relation_logical_arities = i64_attr(encoding, "relation_logical_arities");
   const auto relation_encoded_arities = i64_attr(encoding, "relation_encoded_arities");
   const auto relation_slot_roles = i64_attr(encoding, "relation_slot_roles");
   const auto relation_slot_role_offsets = i64_attr(encoding, "relation_slot_role_offsets");
   const auto slot_role_names = str_attr(encoding, "slot_role_names");
   const auto entity_role_names = str_attr(encoding, "entity_role_names");
   const auto entity_role_ids = i64_field(encoding, "entity_role_ids");
   const auto relation_counts = i64_field(encoding, "relation_counts");
   const auto relation_args = i64_field(encoding, "relation_args");
   const auto base_relation_arities = i64_attr(base, "relation_arities");
   const auto base_relation_counts = i64_field(base, "relation_counts");
   const auto base_relation_args = i64_field(base, "relation_args");

   ASSERT_EQ(slot_role_names[1], "predicate_slot");
   ASSERT_EQ(entity_role_names[1], "predicate_virtual");

   size_t checked_state_predicate_relations = 0;
   for(size_t relation_idx = 0; relation_idx < relation_names.size(); ++relation_idx) {
      const auto roles_begin = relation_slot_roles.begin()
                               + static_cast< ptrdiff_t >(relation_slot_role_offsets[relation_idx]);
      const auto roles_end = relation_slot_roles.begin()
                             + static_cast< ptrdiff_t >(
                                relation_slot_role_offsets[relation_idx + 1]
                             );
      const std::vector< int64_t > roles(roles_begin, roles_end);

      EXPECT_EQ(relation_arities[relation_idx], relation_encoded_arities[relation_idx]);
      EXPECT_EQ(
         relation_logical_arities[relation_idx],
         static_cast< int64_t >(std::count(roles.begin(), roles.end(), int64_t{0}))
      );

      const auto state_it = std::find(roles.begin(), roles.end(), int64_t{2});
      const auto predicate_it = std::find(roles.begin(), roles.end(), int64_t{1});
      if(state_it == roles.end() || predicate_it == roles.end()) {
         continue;
      }
      EXPECT_EQ(roles.front(), int64_t{2});
      EXPECT_EQ(std::distance(roles.begin(), predicate_it), 1);

      if(relation_counts[relation_idx] <= 0 || base_relation_counts[relation_idx] <= 0) {
         continue;
      }

      const size_t predicate_slot = static_cast< size_t >(
         std::distance(roles.begin(), predicate_it)
      );
      const size_t slot = relation_slot_offset(
         std::span{relation_counts}, std::span{relation_arities}, relation_idx
      );
      const size_t base_slot = relation_slot_offset(
         std::span{base_relation_counts}, std::span{base_relation_arities}, relation_idx
      );
      const size_t width = static_cast< size_t >(relation_arities[relation_idx]);
      const size_t base_width = static_cast< size_t >(base_relation_arities[relation_idx]);

      ASSERT_EQ(width, base_width + 1);
      ASSERT_LT(slot + width - 1, relation_args.size());
      ASSERT_LT(base_slot + base_width - 1, base_relation_args.size());
      EXPECT_EQ(entity_role_ids[relation_args[slot + predicate_slot]], 1);

      for(size_t col = 0; col < predicate_slot; ++col) {
         EXPECT_EQ(relation_args[slot + col], base_relation_args[base_slot + col]);
      }
      for(size_t col = predicate_slot + 1; col < width; ++col) {
         EXPECT_EQ(relation_args[slot + col], base_relation_args[base_slot + (col - 1)]);
      }
      ++checked_state_predicate_relations;
   }

   EXPECT_GT(checked_state_predicate_relations, 0);
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
