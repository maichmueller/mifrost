#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

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
   EXPECT_TRUE(engine.last_encoding_used_composed_plan()) << engine.last_composition_diagnostic();
}

TEST(SemanticFlatHorizonEncoderEngineTest, RootOnlyGraphPreservesEmptyTargetNames)
{
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions());

   const auto encoding = engine.encode(make_root_only_dag());

   EXPECT_TRUE(engine.last_encoding_used_composed_plan()) << engine.last_composition_diagnostic();
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
   EXPECT_TRUE(engine.last_encoding_used_composed_plan()) << engine.last_composition_diagnostic();
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

}  // namespace
}  // namespace mifrost
