#include "mifrost/core/semantic/semantic_transition_dag.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <string>
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
   value.actions = {{1, {}}};
   value.subgoal_layers = {{{{0, {0}}, false}}};
   value.history = {{-1, {{{0, {object}}, true}}}};
   value.history_max_steps = 2;
   return value;
}

std::vector< Node > valid_nodes()
{
   return {
      {
         .state = state(0),
         .index = 0,
         .depth = 0,
         .display_name = std::string("root"),
      },
      {
         .state = state(1),
         .index = 1,
         .depth = 1,
         .incoming_action = SemanticGroundAction{0, {0, 1}},
         .candidate_id = 101,
         .delta_literals = std::vector< SemanticLiteral >{{{0, {1}}, true}},
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
      {
         .state = state(1),
         .index = 3,
         .depth = 2,
         .incoming_action = SemanticGroundAction{0, {1, 0}},
         .candidate_id = 303,
      },
   };
}

std::vector< Edge > valid_edges()
{
   return {{2, 3}, {0, 2}, {1, 3}, {0, 1}};
}

void expect_invalid(const std::function< void() >& call, const std::string& message)
{
   try {
      call();
      FAIL() << "Expected std::invalid_argument";
   } catch(const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
   } catch(...) {
      FAIL() << "Expected std::invalid_argument";
   }
}

SemanticTransitionDAG make_dag(std::vector< Node > nodes, std::vector< Edge > edges)
{
   return SemanticTransitionDAG(predicates(), actions(), std::move(nodes), std::move(edges));
}

TEST(SemanticTransitionDAGTest, OwnsCanonicalDeterministicGraph)
{
   const auto dag = make_dag(valid_nodes(), valid_edges());

   ASSERT_EQ(dag.size(), 4);
   EXPECT_EQ(dag.root().index, 0);
   EXPECT_EQ(dag.root().state.objects, (std::vector< std::string >{"a", "b"}));
   EXPECT_EQ(dag.edges(), (std::vector< Edge >{{0, 1}, {0, 2}, {1, 3}, {2, 3}}));
   EXPECT_EQ(dag.children(0), (std::vector< int64_t >{1, 2}));
   EXPECT_EQ(dag.children(3), (std::vector< int64_t >{}));
   EXPECT_EQ(dag.parents(3), (std::vector< int64_t >{1, 2}));
   EXPECT_EQ(dag.nodes()[3].depth, 2);
   EXPECT_EQ(dag.nodes()[1].incoming_action, (SemanticGroundAction{0, {0, 1}}));
   EXPECT_EQ(dag.nodes()[1].display_name, "left");
   EXPECT_EQ(dag.candidate_id_coverage(), SemanticCandidateIdCoverage::complete);
   EXPECT_EQ(dag.candidate_ids(), (std::vector< int64_t >{101, 202, 303}));
   EXPECT_NO_THROW(dag.validate_candidate_ids());
   EXPECT_THROW((void) dag.children(-1), std::out_of_range);
   EXPECT_THROW((void) dag.parents(4), std::out_of_range);
}

TEST(SemanticTransitionDAGTest, AllowsRootOnlyDagWithEmptySchemas)
{
   const SemanticTransitionDAG dag(
      {}, {}, {{.state = SemanticFlatRelationInput{}, .index = 0, .depth = 0}}, {}
   );

   EXPECT_EQ(dag.size(), 1);
   EXPECT_TRUE(dag.predicates().empty());
   EXPECT_TRUE(dag.actions().empty());
   EXPECT_TRUE(dag.edges().empty());
   EXPECT_EQ(dag.candidate_id_coverage(), SemanticCandidateIdCoverage::none);
   EXPECT_NO_THROW(dag.validate_candidate_ids());
}

TEST(SemanticTransitionDAGTest, CandidateValidationSupportsImplicitPartialAndDuplicateModes)
{
   auto nodes = valid_nodes();
   for(size_t index = 1; index < nodes.size(); ++index) {
      nodes[index].candidate_id.reset();
   }
   const auto implicit = make_dag(nodes, valid_edges());
   EXPECT_EQ(implicit.candidate_id_coverage(), SemanticCandidateIdCoverage::none);
   EXPECT_TRUE(implicit.candidate_ids().empty());

   nodes[1].candidate_id = 7;
   const auto partial = make_dag(nodes, valid_edges());
   EXPECT_EQ(partial.candidate_id_coverage(), SemanticCandidateIdCoverage::partial);
   expect_invalid(
      [&] { partial.validate_candidate_ids(); }, "missing candidate_id for node index 2"
   );

   nodes[2].candidate_id = 7;
   nodes[3].candidate_id = 8;
   const auto duplicate = make_dag(nodes, valid_edges());
   expect_invalid([&] { (void) duplicate.candidate_ids(); }, "duplicate candidate_id 7");

   nodes[2].candidate_id = 9;
   nodes[0].candidate_id = 42;
   const auto root_candidate = make_dag(nodes, valid_edges());
   EXPECT_EQ(root_candidate.candidate_ids(), (std::vector< int64_t >{7, 9, 8}));
   EXPECT_EQ(root_candidate.candidate_ids(true), (std::vector< int64_t >{42, 7, 9, 8}));
}

TEST(SemanticTransitionDAGTest, RejectsNodeAndRootInvariantViolations)
{
   expect_invalid([&] { make_dag({}, {}); }, "requires a nonempty node list");

   auto nodes = valid_nodes();
   nodes[2].index = 7;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "indices must be stable and contiguous");

   nodes = valid_nodes();
   nodes[0].depth = 1;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "root must have index and depth 0");

   nodes = valid_nodes();
   nodes[1].depth = -1;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "depth must be non-negative");

   nodes = valid_nodes();
   nodes[2].state.objects = {"b", "a"};
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "ordered object tables");

   nodes = valid_nodes();
   nodes[1].display_name = "";
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "display name must not be empty");

   nodes = valid_nodes();
   nodes[0].incoming_action = SemanticGroundAction{1, {}};
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "root cannot have an incoming action");

   nodes = valid_nodes();
   nodes[0].delta_literals = std::vector< SemanticLiteral >{};
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "root cannot have delta literals");
}

TEST(SemanticTransitionDAGTest, RejectsInvalidSchemaAndSemanticPayloads)
{
   auto predicate_specs = predicates();
   predicate_specs[1].name = predicate_specs[0].name;
   expect_invalid(
      [&] { SemanticTransitionDAG(predicate_specs, actions(), valid_nodes(), valid_edges()); },
      "unique predicate names"
   );

   predicate_specs = predicates();
   predicate_specs[0].category = static_cast< SemanticPredicateCategory >(99);
   expect_invalid(
      [&] { SemanticTransitionDAG(predicate_specs, actions(), valid_nodes(), valid_edges()); },
      "predicate category is invalid"
   );

   auto action_specs = actions();
   action_specs[0].arity = -1;
   expect_invalid(
      [&] { SemanticTransitionDAG(predicates(), action_specs, valid_nodes(), valid_edges()); },
      "action arity must be non-negative"
   );

   auto nodes = valid_nodes();
   nodes[0].state.objects[1] = nodes[0].state.objects[0];
   for(size_t index = 1; index < nodes.size(); ++index) {
      nodes[index].state.objects = nodes[0].state.objects;
   }
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "object names must be unique");

   nodes = valid_nodes();
   nodes[1].state.state_facts[0].predicate = 99;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "predicate index out of range");

   nodes = valid_nodes();
   nodes[1].state.goals[0].atom.arguments.clear();
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "schema arity");

   nodes = valid_nodes();
   nodes[1].state.subgoal_layers[0][0].atom.arguments = {2};
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "object index out of range");

   nodes = valid_nodes();
   nodes[1].state.history[0].dt = 0;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "history requires negative dt");

   nodes = valid_nodes();
   nodes[1].state.history_max_steps = -1;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "history_max_steps");

   nodes = valid_nodes();
   nodes[1].state.actions[0].action = 99;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "action index out of range");

   nodes = valid_nodes();
   nodes[1].incoming_action->arguments.clear();
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "incoming action argument count");

   nodes = valid_nodes();
   nodes[1].delta_literals->front().atom.arguments = {3};
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "delta literal object index");
}

TEST(SemanticTransitionDAGTest, RejectsMalformedTopologyAndIncorrectDepths)
{
   expect_invalid([&] { make_dag(valid_nodes(), {{0, 4}}); }, "edge endpoint out of range");
   expect_invalid([&] { make_dag(valid_nodes(), {{0, 0}}); }, "self edges");
   expect_invalid([&] { make_dag(valid_nodes(), {{0, 1}, {0, 1}}); }, "duplicate edge");
   expect_invalid([&] { make_dag(valid_nodes(), {{0, 1}, {1, 2}, {2, 1}, {2, 3}}); }, "acyclic");
   expect_invalid([&] { make_dag(valid_nodes(), {{0, 1}, {0, 2}}); }, "reachable from root");

   auto nodes = valid_nodes();
   nodes[3].depth = 3;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "depth must equal shortest-path depth");

   nodes = valid_nodes();
   nodes[3].depth = 1;
   expect_invalid([&] { make_dag(nodes, valid_edges()); }, "depth must equal shortest-path depth");
}

}  // namespace
}  // namespace mifrost
