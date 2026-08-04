#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "mifrost/core/encoders/flat/flat_composition.hpp"

namespace mifrost {
namespace {

using Node = SemanticTransitionDAG::Node;
using Edge = SemanticTransitionDAG::Edge;

std::vector< SemanticPredicateSpec > predicates()
{
   return {
      {SemanticPredicateCategory::fluent, "at", 1},
      {SemanticPredicateCategory::static_predicate, "ready", 0},
   };
}

std::vector< SemanticActionSpec > actions()
{
   return {{"move", 2}, {"finish", 0}};
}

SemanticFlatRelationInput state(int64_t object)
{
   SemanticFlatRelationInput value;
   value.objects = {"a", "b"};
   value.state_facts = {{0, {object}}, {1, {}}};
   value.goals = {{{0, {1}}, true}};
   return value;
}

std::vector< Node > sample_nodes()
{
   return {
      {
         .state = state(0),
         .index = 0,
         .depth = 0,
         .candidate_id = 0,
         .display_name = std::string("root"),
      },
      {
         .state = state(1),
         .index = 1,
         .depth = 1,
         .incoming_action = SemanticGroundAction{0, {0, 1}},
         .candidate_id = 101,
         .delta_literals = std::vector< SemanticLiteral >{{{0, {1}}, true}, {{0, {0}}, false}},
         .display_name = std::string("left"),
      },
      {
         .state = state(0),
         .index = 2,
         .depth = 1,
         .incoming_action = SemanticGroundAction{1, {}},
         .candidate_id = 202,
         .display_name = std::string("right"),
      },
   };
}

std::vector< Edge > sample_edges()
{
   return {{0, 1}, {0, 2}};
}

SemanticTransitionDAG make_dag()
{
   return SemanticTransitionDAG(predicates(), actions(), sample_nodes(), sample_edges());
}

SemanticTransitionDAG make_root_only_dag()
{
   auto nodes = sample_nodes();
   nodes.resize(1);
   return SemanticTransitionDAG(predicates(), actions(), std::move(nodes), {});
}

bool contains(const std::vector< std::string >& names, const std::string& name)
{
   return std::ranges::find(names, name) != names.end();
}

TEST(SemanticFlatHorizonEncoderEngineTest, FullModeDefaultConfigEncodesWithoutThrowing)
{
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions());
   const auto& names = engine.get_relation_names();
   ASSERT_FALSE(names.empty());
   // Root-only variant and the split "[state]"-anchored candidate variant must both exist,
   // since the default root_policy (exclude) triggers split_full_state_relations() in full mode.
   EXPECT_TRUE(contains(names, "at"));
   EXPECT_TRUE(contains(names, "at[state]"));
   EXPECT_TRUE(contains(names, "[+]at[g]"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, RootOnlyGraphPreservesEmptyTargetNames)
{
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions());

   const auto encoding = engine.encode(make_root_only_dag());

   ASSERT_TRUE(encoding.graph_attrs.contains(std::string(kTargetNamesAttr)));
   EXPECT_TRUE(
      std::get< std::vector< std::string > >(encoding.graph_attrs.at(std::string(kTargetNamesAttr)))
         .empty()
   );
}

TEST(SemanticFlatHorizonEncoderEngineTest, DeltaModeRegistersLiteralCandidateRelations)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.transition_mode = SemanticHorizonMode::delta;
   config.support_literals = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto& names = engine.get_relation_names();
   EXPECT_TRUE(contains(names, "[+]at"));
   EXPECT_TRUE(contains(names, "[-]at"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, BatchEncodingUsesCompiledPlanForRelationMajorPacking)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.pack_relation_args_relation_major = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto encoding = engine.encode_batch({make_dag(), make_dag()});

   EXPECT_EQ(encoding.num_graphs, 2);
}

TEST(SemanticFlatHorizonEncoderEngineTest, ParityMatrixUsesCompiledPlanAcrossHorizonPolicies)
{
   std::vector< SemanticFlatHorizonEncoderEngine::Config > configs;
   configs.emplace_back();
   configs.back().root_policy = RootPolicy::include;
   configs.back().support_literals = true;
   configs.back().include_lgan_edges = true;
   configs.back().enable_parent_relation = true;
   configs.back().enable_sibling_relation = true;
   configs.back().enable_cousin_relation = true;
   configs.back().use_predicate_virtual_nodes = true;
   configs.back().goal_derivations = {
      GoalDerivation::plain,
      GoalDerivation::satisfied,
      GoalDerivation::unsatisfied,
      GoalDerivation::added_satisfied,
      GoalDerivation::added_unsatisfied,
   };

   configs.emplace_back();
   configs.back().root_policy = RootPolicy::encode_only;
   configs.back().pack_relation_args_relation_major = true;

   configs.emplace_back();
   configs.back().root_policy = RootPolicy::exclude;
   configs.back().export_node_names = false;

   configs.emplace_back();
   configs.back().transition_mode = SemanticHorizonMode::delta;
   configs.back().root_policy = RootPolicy::exclude;
   configs.back().pack_relation_args_relation_major = true;

   configs.emplace_back();
   configs.back().transition_mode = SemanticHorizonMode::action;
   configs.back().ignore_actions = false;
   configs.back().root_policy = RootPolicy::include;

   for(const auto& config : configs) {
      SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);
      const auto encoding = engine.encode_batch({make_dag(), make_dag()});

      EXPECT_EQ(encoding.num_graphs, 2);
   }
}

TEST(SemanticFlatHorizonEncoderEngineTest, TopologyRelationsUseConfiguredNamesVerbatim)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.enable_parent_relation = true;
   config.enable_sibling_relation = true;
   config.enable_cousin_relation = true;
   config.parent_relation = "_custom_parent_";
   config.sibling_relation = "_custom_sibling_";
   config.cousin_relation = "_custom_cousin_";
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto& names = engine.get_relation_names();
   EXPECT_TRUE(contains(names, "_custom_parent_"));
   EXPECT_TRUE(contains(names, "_custom_sibling_"));
   EXPECT_TRUE(contains(names, "_custom_cousin_"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, RootPolicyIncludeSkipsSplitStateAnchoring)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.root_policy = RootPolicy::include;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto& names = engine.get_relation_names();
   EXPECT_TRUE(contains(names, "at"));
   // No split candidate relation should exist when the root isn't excluded.
   EXPECT_FALSE(contains(names, "at[state]"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, GoalSatisfactionDerivationsRegisterRootAndAnchoredForms)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.goal_derivations = {
      GoalDerivation::plain, GoalDerivation::satisfied, GoalDerivation::unsatisfied
   };
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto& names = engine.get_relation_names();
   EXPECT_TRUE(contains(names, "[+]at[g][sat]"));
   EXPECT_TRUE(contains(names, "[+]at[g][sat][state]"));
   EXPECT_TRUE(contains(names, "[+]at[g][unsat]"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, BuilderPathMatchesOneShotComposition)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.support_literals = true;
   config.include_lgan_edges = true;
   config.enable_parent_relation = true;
   config.enable_sibling_relation = true;
   config.enable_cousin_relation = true;
   config.use_predicate_virtual_nodes = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);
   const auto dag = make_dag();

   const auto expected = engine.encode(dag);
   BatchBuilder builder;
   engine.encode(dag, builder);
   builder.next_graph();
   auto actual = builder.build();
   engine.finalize_batch_encoding(actual);

   const auto parity = compare_flat_batch_encodings(expected, actual);
   ASSERT_TRUE(parity.equal) << parity.mismatch;
}

TEST(SemanticFlatHorizonEncoderEngineTest, CompiledPlanIsSafeForConcurrentEncodes)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.transition_mode = SemanticHorizonMode::delta;
   config.include_lgan_edges = true;
   config.enable_parent_relation = true;
   config.use_predicate_virtual_nodes = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);
   const auto dag = make_dag();
   const auto expected = engine.encode(dag);

   std::vector< std::future< BatchBuilder::BatchEncoding > > jobs;
   for(size_t index = 0; index < 8; ++index) {
      jobs.push_back(std::async(std::launch::async, [&engine, &dag] {
         return engine.encode(dag);
      }));
   }
   for(auto& job : jobs) {
      const auto parity = compare_flat_batch_encodings(expected, job.get());
      ASSERT_TRUE(parity.equal) << parity.mismatch;
   }
}

}  // namespace
}  // namespace mifrost
