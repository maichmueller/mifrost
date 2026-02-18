#include "mifrost/core/horizon_hgraph_encoder.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <mimir/search/applicable_action_generators/interface.hpp>
#include <mimir/search/search_context.hpp>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mifrost/core/transition_dag.hpp"
#include "test_utils.hpp"

using namespace mifrost;

class HorizonHGraphEncoderTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(HorizonHGraphEncoderTest, EmitsTargetGraphAttributesAndSymbols)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, succ_state, succ_action);

   const std::array< HorizonHGraphEncoderEngine::Mode, 2 > modes = {
      HorizonHGraphEncoderEngine::Mode::Full,
      HorizonHGraphEncoderEngine::Mode::Delta,
   };

   for(const auto mode : modes) {
      HorizonHGraphEncoderEngine::Config config;
      config.transition_mode = mode;
      if(mode == HorizonHGraphEncoderEngine::Mode::Delta) {
         config.support_literals = true;
      }

      HorizonHGraphEncoderEngine engine(ctx.problem->get_domain(), config);

      BatchBuilder builder;
      builder.set_graph_kind("hetero");
      auto goals = mifrost_test::make_goal_inputs(ctx.problem);
      engine.encode(ctx.root, dag, goals, builder);
      builder.next_graph();

      EXPECT_EQ(builder.graph_kind, "hetero");

      ASSERT_NE(builder.graph_fields, nullptr);
      EXPECT_TRUE(builder.graph_fields->contains("target_positions"));
      EXPECT_TRUE(builder.graph_fields->contains("target_indices"));
      EXPECT_TRUE(builder.graph_fields->contains("target_depths"));
      EXPECT_TRUE(builder.graph_attrs.contains("target_names"));
      EXPECT_TRUE(builder.graph_attrs.contains("target_symbol_prefix"));
      EXPECT_TRUE(builder.graph_attrs.contains("parent_relation"));

      const auto& positions = std::get< std::vector< int64_t > >(
         builder.graph_fields->at("target_positions").values
      );
      const auto& indices = std::get< std::vector< int64_t > >(
         builder.graph_fields->at("target_indices").values
      );
      const auto& depths = std::get< std::vector< int64_t > >(
         builder.graph_fields->at("target_depths").values
      );
      const auto& names = std::get< std::vector< std::string > >(
         builder.graph_attrs.at("target_names")
      );

      ASSERT_EQ(positions.size(), indices.size());
      ASSERT_EQ(positions.size(), depths.size());
      ASSERT_EQ(positions.size(), names.size());
      const size_t expected_candidates = config.exclude_root_candidate
                                            ? (dag.nodes().empty() ? 0u : dag.nodes().size() - 1u)
                                            : dag.nodes().size();
      ASSERT_EQ(positions.size(), expected_candidates);
      ASSERT_FALSE(positions.empty());

      const auto prefix = std::get< std::string >(builder.graph_attrs.at("target_symbol_prefix"));
      const auto parent_relation = std::get< std::string >(
         builder.graph_attrs.at("parent_relation")
      );
      EXPECT_EQ(prefix, config.target_symbol_prefix);
      EXPECT_EQ(parent_relation, config.parent_relation);

      const auto symbol_it = builder.node_names.find(engine.get_config().symbol_type_id);
      ASSERT_NE(symbol_it, builder.node_names.end());
      const auto& symbol_names = symbol_it->second;
      std::unordered_map< std::string, int64_t > symbol_indices;
      for(size_t i = 0; i < symbol_names.size(); ++i) {
         symbol_indices.emplace(symbol_names[i], static_cast< int64_t >(i));
      }

      size_t candidate_pos = 0;
      for(const auto& node : dag.nodes()) {
         if(config.exclude_root_candidate and node.index == dag.root_index()) {
            continue;
         }
         const std::string key = config.target_symbol_prefix + std::to_string(node.index);
         const auto sym_it = symbol_indices.find(key);
         ASSERT_NE(sym_it, symbol_indices.end()) << "Missing target symbol node: " << key;
         ASSERT_LT(candidate_pos, positions.size());
         EXPECT_EQ(indices[candidate_pos], node.index);
         EXPECT_EQ(positions[candidate_pos], sym_it->second);
         EXPECT_EQ(depths[candidate_pos], node.depth);
         ++candidate_pos;
      }

      if(not builder.object_names.empty()) {
         for(const auto& node : dag.nodes()) {
            const std::string key = config.target_symbol_prefix + std::to_string(node.index);
            EXPECT_EQ(
               std::find(builder.object_names.begin(), builder.object_names.end(), key),
               builder.object_names.end()
            );
         }
      }
   }
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   HorizonHGraphEncoderTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

TEST_P(HorizonHGraphEncoderTest, DeltaModeEncodesOnlyChangedLiteralsForSuccessor)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, succ_state, succ_action);

   HorizonHGraphEncoderEngine::Config config;
   config.transition_mode = HorizonHGraphEncoderEngine::Mode::Delta;
   config.support_literals = true;
   config.ignore_actions = true;
   config.include_lgan_edges = false;

   HorizonHGraphEncoderEngine engine(ctx.problem->get_domain(), config);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(ctx.root, dag, goals, builder);

   const std::string prefix = config.target_symbol_prefix + std::to_string(succ_state.get_index())
                              + "|";

   std::unordered_map< std::string, std::unordered_set< std::string > > expected;

   const auto& repos = ctx.problem->get_repositories();
   auto collect_indices = [&]< typename Tag >(Tag, const mimir::search::State& state) {
      std::unordered_set< int > indices;
      const auto atoms = repos.get_ground_atoms_from_indices< Tag >(state.get_atoms< Tag >());
      for(const auto& atom : atoms) {
         if(atom->get_predicate()->get_arity() == 0 && not config.add_nullary_predicates) {
            continue;
         }
         indices.insert(atom->get_index());
      }
      return indices;
   };

   auto root_fluents = collect_indices(mimir::formalism::FluentTag{}, ctx.root);
   auto root_derived = collect_indices(mimir::formalism::DerivedTag{}, ctx.root);
   auto succ_fluents = collect_indices(mimir::formalism::FluentTag{}, succ_state);
   auto succ_derived = collect_indices(mimir::formalism::DerivedTag{}, succ_state);

   auto add_literal = [&]< typename Tag >(Tag, int idx, bool polarity) {
      auto atom = repos.get_ground_atom< Tag >(idx);
      if(atom->get_predicate()->get_arity() == 0 && not config.add_nullary_predicates) {
         return;
      }
      const std::string node_type = RelationFormatter::format_predicate(
         atom->get_predicate(), std::nullopt, std::nullopt, polarity
      );
      const std::string atom_str = RelationFormatter::format_atom(atom);
      const std::string literal_str = fmt::format(
         "{}{}", RelationFormatter::polarity_prefix(polarity), atom_str
      );
      const std::string node_key = prefix + literal_str;
      expected[node_type].insert(node_key);
   };

   for(const auto idx : succ_fluents) {
      if(not root_fluents.contains(idx)) {
         add_literal(mimir::formalism::FluentTag{}, idx, true);
      }
   }
   for(const auto idx : root_fluents) {
      if(not succ_fluents.contains(idx)) {
         add_literal(mimir::formalism::FluentTag{}, idx, false);
      }
   }
   for(const auto idx : succ_derived) {
      if(not root_derived.contains(idx)) {
         add_literal(mimir::formalism::DerivedTag{}, idx, true);
      }
   }
   for(const auto idx : root_derived) {
      if(not succ_derived.contains(idx)) {
         add_literal(mimir::formalism::DerivedTag{}, idx, false);
      }
   }

   if(expected.empty()) {
      GTEST_SKIP() << "No delta literals found for successor state.";
   }

   for(const auto& [node_type, nodes] : expected) {
      const auto it = builder.node_names.find(node_type);
      ASSERT_NE(it, builder.node_names.end()) << "Missing delta node type: " << node_type;
      std::unordered_set< std::string > actual(it->second.begin(), it->second.end());
      for(const auto& node_key : nodes) {
         EXPECT_TRUE(actual.contains(node_key)) << "Missing delta node key: " << node_key;
      }
   }

   for(const auto& [node_type, nodes] : builder.node_names) {
      for(const auto& node_key : nodes) {
         if(node_key.rfind(prefix, 0) != 0) {
            continue;
         }
         const auto it = expected.find(node_type);
         ASSERT_NE(it, expected.end()) << "Unexpected delta node type for prefix: " << node_type;
         EXPECT_TRUE(it->second.contains(node_key)) << "Unexpected delta node key: " << node_key;
      }
   }
}

TEST_P(HorizonHGraphEncoderTest, ObjectSymbolsParticipateInSymbolToRelationEdges)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, succ_state, succ_action);

   HorizonHGraphEncoderEngine::Config config;
   config.transition_mode = HorizonHGraphEncoderEngine::Mode::Full;
   config.ignore_actions = false;
   config.include_lgan_edges = false;

   HorizonHGraphEncoderEngine engine(ctx.problem->get_domain(), config);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(ctx.root, dag, goals, builder);

   const auto index_map = mifrost_test::build_index_map(builder);
   const auto symbol_it = index_map.find(config.symbol_type_id);
   ASSERT_NE(symbol_it, index_map.end());
   const auto& symbol_index = symbol_it->second;

   std::unordered_set< int64_t > object_symbol_indices;
   for(const auto& object_name : builder.object_names) {
      const auto it = symbol_index.find(object_name);
      if(it != symbol_index.end()) {
         object_symbol_indices.insert(it->second);
      }
   }
   ASSERT_FALSE(object_symbol_indices.empty())
      << "No object symbols were mapped to indices in the symbol type.";

   const std::string symbol_edge_prefix = config.symbol_type_id + "|";
   const std::string src_suffix = "/edge_index_0";
   bool found_object_symbol_src_edge = false;

   for(const auto& [key, column] : builder.columns) {
      if(key.rfind(symbol_edge_prefix, 0) != 0) {
         continue;
      }
      if(key.size() < src_suffix.size()
         || key.compare(key.size() - src_suffix.size(), src_suffix.size(), src_suffix) != 0) {
         continue;
      }
      const auto& src_col = std::get< BatchBuilder::LongCol >(column.data);
      for(const auto src_idx : src_col) {
         if(object_symbol_indices.contains(src_idx)) {
            found_object_symbol_src_edge = true;
            break;
         }
      }
      if(found_object_symbol_src_edge) {
         break;
      }
   }

   EXPECT_TRUE(found_object_symbol_src_edge)
      << "Expected at least one symbol->relation edge whose source is an object symbol node.";
}

TEST_P(HorizonHGraphEncoderTest, ParentRelationsMatchDagTransitions)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   std::vector< mimir::search::State > successors;
   for(const auto& action : ctx.actions) {
      auto [succ_state, _metric] = ctx.repo->get_or_create_successor_state(
         ctx.root, action, ctx.root_metric
      );
      if(succ_state.get_index() != ctx.root.get_index()) {
         successors.push_back(succ_state);
      }
      if(successors.size() >= 2) {
         break;
      }
   }
   if(successors.empty()) {
      GTEST_SKIP() << "No successor states found for parent relation test.";
   }

   TransitionDAG dag(ctx.root);
   for(const auto& succ_state : successors) {
      dag.register_transition(ctx.root, succ_state, std::nullopt);
   }

   HorizonHGraphEncoderEngine::Config config;
   config.enable_parent_relation = true;
   config.enable_sibling_relation = false;
   config.enable_cousin_relation = false;

   HorizonHGraphEncoderEngine engine(ctx.problem->get_domain(), config);
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(ctx.root, dag, goals, builder);

   const auto index_map = mifrost_test::build_index_map(builder);
   const auto symbol_it = index_map.find(config.symbol_type_id);
   ASSERT_NE(symbol_it, index_map.end());
   const auto& symbol_index = symbol_it->second;

   const auto rel_it = index_map.find(config.parent_relation);
   ASSERT_NE(rel_it, index_map.end()) << "Missing parent relation node type";

   const auto edges_sym_to_rel0 = mifrost_test::edge_pairs_for(
      builder, config.symbol_type_id + "|0|" + config.parent_relation
   );
   const auto edges_sym_to_rel1 = mifrost_test::edge_pairs_for(
      builder, config.symbol_type_id + "|1|" + config.parent_relation
   );
   const auto edges_rel_to_sym0 = mifrost_test::edge_pairs_for(
      builder, config.parent_relation + "|0|" + config.symbol_type_id
   );
   const auto edges_rel_to_sym1 = mifrost_test::edge_pairs_for(
      builder, config.parent_relation + "|1|" + config.symbol_type_id
   );

   auto has_pair = [](const mifrost_test::EdgePairs& pairs, int64_t a, int64_t b) {
      for(const auto& entry : pairs) {
         if(entry.first == a && entry.second == b) {
            return true;
         }
      }
      return false;
   };

   for(const auto& pair : dag.transitions()) {
      const int parent_idx = pair.first;
      const int child_idx = pair.second;
      const std::string rel_key = fmt::format(
         "{}({}->{})", config.parent_relation, parent_idx, child_idx
      );
      const auto rel_node_it = rel_it->second.find(rel_key);
      ASSERT_NE(rel_node_it, rel_it->second.end()) << "Missing parent relation node: " << rel_key;

      const auto parent_symbol = config.target_symbol_prefix + std::to_string(parent_idx);
      const auto child_symbol = config.target_symbol_prefix + std::to_string(child_idx);
      const auto parent_it = symbol_index.find(parent_symbol);
      const auto child_it = symbol_index.find(child_symbol);
      ASSERT_NE(parent_it, symbol_index.end()) << "Missing parent target symbol: " << parent_symbol;
      ASSERT_NE(child_it, symbol_index.end()) << "Missing child target symbol: " << child_symbol;

      const int64_t rel_idx = rel_node_it->second;
      const int64_t parent_node = parent_it->second;
      const int64_t child_node = child_it->second;

      EXPECT_TRUE(has_pair(edges_sym_to_rel0, parent_node, rel_idx));
      EXPECT_TRUE(has_pair(edges_rel_to_sym0, rel_idx, parent_node));
      EXPECT_TRUE(has_pair(edges_sym_to_rel1, child_node, rel_idx));
      EXPECT_TRUE(has_pair(edges_rel_to_sym1, rel_idx, child_node));
   }
}

TEST_P(HorizonHGraphEncoderTest, SiblingAndCousinRelationsMatchDagStructure)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto search_ctx = mimir::search::SearchContextImpl::create(ctx.problem);
   auto repo = search_ctx->get_state_repository();
   auto applicable_actions = search_ctx->get_applicable_action_generator();
   auto [root_state, root_metric] = repo->get_or_create_initial_state();

   constexpr int kMaxDepth = 5;
   struct QueueEntry {
      mimir::search::State state;
      mimir::ContinuousCost metric;
   };

   TransitionDAG dag(root_state);
   std::unordered_map< int, int > depths;
   std::unordered_map< int, mimir::ContinuousCost > metrics;
   std::vector< QueueEntry > queue;
   queue.push_back({root_state, root_metric});
   depths.emplace(root_state.get_index(), 0);
   metrics.emplace(root_state.get_index(), root_metric);

   for(size_t qi = 0; qi < queue.size(); ++qi) {
      const auto entry = queue[qi];
      const int depth = depths.at(entry.state.get_index());
      if(depth >= kMaxDepth) {
         continue;
      }
      for(const auto& action :
          applicable_actions->create_applicable_action_generator(entry.state)) {
         auto [succ_state, succ_metric] = repo->get_or_create_successor_state(
            entry.state, action, entry.metric
         );
         if(succ_state.get_index() == entry.state.get_index()) {
            continue;
         }
         const int succ_idx = succ_state.get_index();
         if(dag.contains(succ_state)) {
            continue;
         }
         dag.register_transition(entry.state, succ_state, std::nullopt);
         depths.emplace(succ_idx, depth + 1);
         metrics.emplace(succ_idx, succ_metric);
         queue.push_back({succ_state, succ_metric});
      }
   }

   if(dag.transitions().size() < 2) {
      GTEST_SKIP() << "Not enough transitions to form sibling/cousin relations.";
   }

   HorizonHGraphEncoderEngine::Config config;
   config.enable_parent_relation = false;
   config.enable_sibling_relation = true;
   config.enable_cousin_relation = true;

   HorizonHGraphEncoderEngine engine(ctx.problem->get_domain(), config);
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(root_state, dag, goals, builder);

   const auto index_map = mifrost_test::build_index_map(builder);
   const auto symbol_it = index_map.find(config.symbol_type_id);
   ASSERT_NE(symbol_it, index_map.end());
   const auto& symbol_index = symbol_it->second;

   auto has_pair = [](const mifrost_test::EdgePairs& pairs, int64_t a, int64_t b) {
      for(const auto& entry : pairs) {
         if(entry.first == a && entry.second == b) {
            return true;
         }
      }
      return false;
   };

   auto assert_relation = [&](const std::string& relation, int a, int b) {
      const auto rel_it = index_map.find(relation);
      ASSERT_NE(rel_it, index_map.end()) << "Missing relation type: " << relation;

      const std::string rel_key = fmt::format("{}({}->{})", relation, a, b);
      const auto rel_node_it = rel_it->second.find(rel_key);
      ASSERT_NE(rel_node_it, rel_it->second.end()) << "Missing relation node: " << rel_key;

      const auto a_symbol = config.target_symbol_prefix + std::to_string(a);
      const auto b_symbol = config.target_symbol_prefix + std::to_string(b);
      const auto a_it = symbol_index.find(a_symbol);
      const auto b_it = symbol_index.find(b_symbol);
      ASSERT_NE(a_it, symbol_index.end()) << "Missing target symbol: " << a_symbol;
      ASSERT_NE(b_it, symbol_index.end()) << "Missing target symbol: " << b_symbol;

      const int64_t rel_idx = rel_node_it->second;
      const int64_t a_idx = a_it->second;
      const int64_t b_idx = b_it->second;

      const auto edges_sym_to_rel0 = mifrost_test::edge_pairs_for(
         builder, config.symbol_type_id + "|0|" + relation
      );
      const auto edges_sym_to_rel1 = mifrost_test::edge_pairs_for(
         builder, config.symbol_type_id + "|1|" + relation
      );
      const auto edges_rel_to_sym0 = mifrost_test::edge_pairs_for(
         builder, relation + "|0|" + config.symbol_type_id
      );
      const auto edges_rel_to_sym1 = mifrost_test::edge_pairs_for(
         builder, relation + "|1|" + config.symbol_type_id
      );

      EXPECT_TRUE(has_pair(edges_sym_to_rel0, a_idx, rel_idx));
      EXPECT_TRUE(has_pair(edges_rel_to_sym0, rel_idx, a_idx));
      EXPECT_TRUE(has_pair(edges_sym_to_rel1, b_idx, rel_idx));
      EXPECT_TRUE(has_pair(edges_rel_to_sym1, rel_idx, b_idx));
   };

   std::unordered_map< int, std::vector< int > > parent_to_children;
   for(const auto& pair : dag.transitions()) {
      parent_to_children[pair.first].push_back(pair.second);
   }
   for(auto& [_, children] : parent_to_children) {
      std::sort(children.begin(), children.end());
      children.erase(std::unique(children.begin(), children.end()), children.end());
   }

   std::set< std::pair< int, int > > sibling_pairs;
   for(const auto& [_, children] : parent_to_children) {
      for(size_t i = 0; i + 1 < children.size(); ++i) {
         for(size_t j = i + 1; j < children.size(); ++j) {
            const int a = std::min(children[i], children[j]);
            const int b = std::max(children[i], children[j]);
            sibling_pairs.insert({a, b});
         }
      }
   }

   std::set< std::pair< int, int > > cousin_pairs;
   for(const auto& [g, parents] : parent_to_children) {
      if(parents.size() < 2) {
         continue;
      }
      for(size_t i = 0; i + 1 < parents.size(); ++i) {
         for(size_t j = i + 1; j < parents.size(); ++j) {
            const int pu = parents[i];
            const int pv = parents[j];
            const auto cu_it = parent_to_children.find(pu);
            const auto cv_it = parent_to_children.find(pv);
            if(cu_it == parent_to_children.end() || cv_it == parent_to_children.end()) {
               continue;
            }
            const auto& cu = cu_it->second;
            const auto& cv = cv_it->second;
            for(int u : cu) {
               for(int v : cv) {
                  if(u == v) {
                     continue;
                  }
                  const int a = std::min(u, v);
                  const int b = std::max(u, v);
                  if(sibling_pairs.contains({a, b})) {
                     continue;
                  }
                  cousin_pairs.insert({a, b});
               }
            }
         }
      }
   }

   if(sibling_pairs.empty() && cousin_pairs.empty()) {
      GTEST_SKIP() << "No sibling or cousin pairs found in depth-5 DAG.";
   }

   for(const auto& [a, b] : sibling_pairs) {
      assert_relation(config.sibling_relation, a, b);
      assert_relation(config.sibling_relation, b, a);
   }

   for(const auto& [a, b] : cousin_pairs) {
      assert_relation(config.cousin_relation, a, b);
      assert_relation(config.cousin_relation, b, a);
   }
}

TEST_P(HorizonHGraphEncoderTest, ParentRelationEdgesMatchDagTransitions)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   TransitionDAG dag(ctx.root);
   std::vector< std::pair< int, int > > transitions;
   int count = 0;
   for(const auto& action : ctx.actions) {
      auto [succ_state, _metric] = ctx.repo->get_or_create_successor_state(
         ctx.root, action, ctx.root_metric
      );
      if(succ_state.get_index() == ctx.root.get_index()) {
         continue;
      }
      dag.register_transition(ctx.root, succ_state, action);
      transitions.emplace_back(0, static_cast< int >(succ_state.get_index()));
      if(++count >= 2) {
         break;
      }
   }
   if(transitions.empty()) {
      GTEST_SKIP() << "No successor transitions available for parent relation test.";
   }

   HorizonHGraphEncoderEngine::Config config;
   config.enable_parent_relation = true;
   config.transition_mode = HorizonHGraphEncoderEngine::Mode::Full;
   HorizonHGraphEncoderEngine engine(ctx.problem->get_domain(), config);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(ctx.root, dag, goals, builder);

   const auto index_map = mifrost_test::build_index_map(builder);
   const auto& symbol_index = index_map.at(engine.get_config().symbol_type_id);

   std::vector< std::pair< int64_t, int64_t > > expected_sym0;
   std::vector< std::pair< int64_t, int64_t > > expected_rel0;
   std::vector< std::pair< int64_t, int64_t > > expected_sym1;
   std::vector< std::pair< int64_t, int64_t > > expected_rel1;

   for(const auto& pair : dag.transitions()) {
      const int parent_idx = pair.first;
      const int child_idx = pair.second;
      const std::string rel_key = fmt::format(
         "{}({}->{})", config.parent_relation, parent_idx, child_idx
      );
      const std::string parent_key = config.target_symbol_prefix + std::to_string(parent_idx);
      const std::string child_key = config.target_symbol_prefix + std::to_string(child_idx);

      const auto rel_it = index_map.at(config.parent_relation).find(rel_key);
      ASSERT_NE(rel_it, index_map.at(config.parent_relation).end())
         << "Missing parent relation node: " << rel_key;

      const auto p_it = symbol_index.find(parent_key);
      const auto c_it = symbol_index.find(child_key);
      ASSERT_NE(p_it, symbol_index.end()) << "Missing parent symbol: " << parent_key;
      ASSERT_NE(c_it, symbol_index.end()) << "Missing child symbol: " << child_key;

      const int64_t rel_idx = rel_it->second;
      const int64_t p_idx = p_it->second;
      const int64_t c_idx = c_it->second;

      expected_sym0.emplace_back(p_idx, rel_idx);
      expected_rel0.emplace_back(rel_idx, p_idx);
      expected_sym1.emplace_back(c_idx, rel_idx);
      expected_rel1.emplace_back(rel_idx, c_idx);
   }

   auto actual_sym0 = mifrost_test::edge_pairs_for(
      builder, engine.get_config().symbol_type_id + "|0|" + config.parent_relation
   );
   auto actual_rel0 = mifrost_test::edge_pairs_for(
      builder, config.parent_relation + "|0|" + engine.get_config().symbol_type_id
   );
   auto actual_sym1 = mifrost_test::edge_pairs_for(
      builder, engine.get_config().symbol_type_id + "|1|" + config.parent_relation
   );
   auto actual_rel1 = mifrost_test::edge_pairs_for(
      builder, config.parent_relation + "|1|" + engine.get_config().symbol_type_id
   );

   mifrost_test::sort_edge_pairs(expected_sym0);
   mifrost_test::sort_edge_pairs(expected_rel0);
   mifrost_test::sort_edge_pairs(expected_sym1);
   mifrost_test::sort_edge_pairs(expected_rel1);
   mifrost_test::sort_edge_pairs(actual_sym0);
   mifrost_test::sort_edge_pairs(actual_rel0);
   mifrost_test::sort_edge_pairs(actual_sym1);
   mifrost_test::sort_edge_pairs(actual_rel1);

   EXPECT_EQ(actual_sym0, expected_sym0);
   EXPECT_EQ(actual_rel0, expected_rel0);
   EXPECT_EQ(actual_sym1, expected_sym1);
   EXPECT_EQ(actual_rel1, expected_rel1);
}

TEST_P(HorizonHGraphEncoderTest, SiblingAndCousinRelationsMatchDag)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   auto find_successor_from = [&](const mimir::search::State& state) {
      for(const auto& action : ctx.actions) {
         auto [succ_state, _metric] = ctx.repo->get_or_create_successor_state(
            state, action, ctx.root_metric
         );
         if(succ_state.get_index() != state.get_index()) {
            return std::optional<
               std::pair< mimir::search::State, mimir::formalism::GroundAction > >{
               {succ_state, action}
            };
         }
      }
      return std::optional< std::pair< mimir::search::State, mimir::formalism::GroundAction > >{};
   };

   std::vector< std::pair< mimir::search::State, mimir::formalism::GroundAction > > root_succs;
   for(const auto& action : ctx.actions) {
      auto [succ_state, _metric] = ctx.repo->get_or_create_successor_state(
         ctx.root, action, ctx.root_metric
      );
      if(succ_state.get_index() == ctx.root.get_index()) {
         continue;
      }
      bool seen = false;
      for(const auto& existing : root_succs) {
         if(existing.first.get_index() == succ_state.get_index()) {
            seen = true;
            break;
         }
      }
      if(not seen) {
         root_succs.emplace_back(succ_state, action);
      }
      if(root_succs.size() >= 2) {
         break;
      }
   }
   if(root_succs.size() < 2) {
      GTEST_SKIP() << "Need two distinct children of root for sibling relation.";
   }

   const auto& succ1 = root_succs[0];
   const auto& succ2 = root_succs[1];

   auto gc1 = find_successor_from(succ1.first);
   auto gc2 = find_successor_from(succ2.first);
   if(not gc1.has_value() || not gc2.has_value()) {
      GTEST_SKIP() << "Need grandchildren for cousin relation.";
   }

   TransitionDAG dag(ctx.root);
   dag.register_transition(ctx.root, succ1.first, succ1.second);
   dag.register_transition(ctx.root, succ2.first, succ2.second);
   dag.register_transition(succ1.first, gc1->first, gc1->second);
   dag.register_transition(succ2.first, gc2->first, gc2->second);

   HorizonHGraphEncoderEngine::Config config;
   config.enable_sibling_relation = true;
   config.enable_cousin_relation = true;
   config.transition_mode = HorizonHGraphEncoderEngine::Mode::Full;
   HorizonHGraphEncoderEngine engine(ctx.problem->get_domain(), config);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(ctx.root, dag, goals, builder);

   const auto index_map = mifrost_test::build_index_map(builder);
   const auto& symbol_index = index_map.at(engine.get_config().symbol_type_id);

   auto add_relation_edges = [&](
                                const std::string& relation,
                                int a,
                                int b,
                                std::vector< std::pair< int64_t, int64_t > >& sym0,
                                std::vector< std::pair< int64_t, int64_t > >& rel0,
                                std::vector< std::pair< int64_t, int64_t > >& sym1,
                                std::vector< std::pair< int64_t, int64_t > >& rel1
                             ) {
      for(int dir = 0; dir < 2; ++dir) {
         int src = dir == 0 ? a : b;
         int dst = dir == 0 ? b : a;
         const std::string rel_key = fmt::format("{}({}->{})", relation, src, dst);
         const auto rel_it = index_map.at(relation).find(rel_key);
         ASSERT_NE(rel_it, index_map.at(relation).end()) << "Missing relation node: " << rel_key;
         const std::string src_key = config.target_symbol_prefix + std::to_string(src);
         const std::string dst_key = config.target_symbol_prefix + std::to_string(dst);
         const auto src_it = symbol_index.find(src_key);
         const auto dst_it = symbol_index.find(dst_key);
         ASSERT_NE(src_it, symbol_index.end()) << "Missing symbol: " << src_key;
         ASSERT_NE(dst_it, symbol_index.end()) << "Missing symbol: " << dst_key;
         const int64_t rel_idx = rel_it->second;
         const int64_t src_idx = src_it->second;
         const int64_t dst_idx = dst_it->second;
         sym0.emplace_back(src_idx, rel_idx);
         rel0.emplace_back(rel_idx, src_idx);
         sym1.emplace_back(dst_idx, rel_idx);
         rel1.emplace_back(rel_idx, dst_idx);
      }
   };

   const int sib_a = dag.index(succ1.first);
   const int sib_b = dag.index(succ2.first);
   std::vector< std::pair< int64_t, int64_t > > expected_sym0_sib;
   std::vector< std::pair< int64_t, int64_t > > expected_rel0_sib;
   std::vector< std::pair< int64_t, int64_t > > expected_sym1_sib;
   std::vector< std::pair< int64_t, int64_t > > expected_rel1_sib;
   add_relation_edges(
      config.sibling_relation,
      sib_a,
      sib_b,
      expected_sym0_sib,
      expected_rel0_sib,
      expected_sym1_sib,
      expected_rel1_sib
   );

   const int cousin_a = dag.index(gc1->first);
   const int cousin_b = dag.index(gc2->first);
   std::vector< std::pair< int64_t, int64_t > > expected_sym0_c;
   std::vector< std::pair< int64_t, int64_t > > expected_rel0_c;
   std::vector< std::pair< int64_t, int64_t > > expected_sym1_c;
   std::vector< std::pair< int64_t, int64_t > > expected_rel1_c;
   const bool has_cousins = cousin_a != cousin_b;
   if(has_cousins) {
      add_relation_edges(
         config.cousin_relation,
         cousin_a,
         cousin_b,
         expected_sym0_c,
         expected_rel0_c,
         expected_sym1_c,
         expected_rel1_c
      );
   }

   auto actual_sym0 = mifrost_test::edge_pairs_for(
      builder, engine.get_config().symbol_type_id + "|0|" + config.sibling_relation
   );
   auto actual_rel0 = mifrost_test::edge_pairs_for(
      builder, config.sibling_relation + "|0|" + engine.get_config().symbol_type_id
   );
   auto actual_sym1 = mifrost_test::edge_pairs_for(
      builder, engine.get_config().symbol_type_id + "|1|" + config.sibling_relation
   );
   auto actual_rel1 = mifrost_test::edge_pairs_for(
      builder, config.sibling_relation + "|1|" + engine.get_config().symbol_type_id
   );

   auto actual_sym0_c = mifrost_test::edge_pairs_for(
      builder, engine.get_config().symbol_type_id + "|0|" + config.cousin_relation
   );
   auto actual_rel0_c = mifrost_test::edge_pairs_for(
      builder, config.cousin_relation + "|0|" + engine.get_config().symbol_type_id
   );
   auto actual_sym1_c = mifrost_test::edge_pairs_for(
      builder, engine.get_config().symbol_type_id + "|1|" + config.cousin_relation
   );
   auto actual_rel1_c = mifrost_test::edge_pairs_for(
      builder, config.cousin_relation + "|1|" + engine.get_config().symbol_type_id
   );

   mifrost_test::sort_edge_pairs(expected_sym0_sib);
   mifrost_test::sort_edge_pairs(expected_rel0_sib);
   mifrost_test::sort_edge_pairs(expected_sym1_sib);
   mifrost_test::sort_edge_pairs(expected_rel1_sib);
   if(has_cousins) {
      mifrost_test::sort_edge_pairs(expected_sym0_c);
      mifrost_test::sort_edge_pairs(expected_rel0_c);
      mifrost_test::sort_edge_pairs(expected_sym1_c);
      mifrost_test::sort_edge_pairs(expected_rel1_c);
   }

   mifrost_test::sort_edge_pairs(actual_sym0);
   mifrost_test::sort_edge_pairs(actual_rel0);
   mifrost_test::sort_edge_pairs(actual_sym1);
   mifrost_test::sort_edge_pairs(actual_rel1);
   mifrost_test::sort_edge_pairs(actual_sym0_c);
   mifrost_test::sort_edge_pairs(actual_rel0_c);
   mifrost_test::sort_edge_pairs(actual_sym1_c);
   mifrost_test::sort_edge_pairs(actual_rel1_c);

   EXPECT_EQ(actual_sym0, expected_sym0_sib);
   EXPECT_EQ(actual_rel0, expected_rel0_sib);
   EXPECT_EQ(actual_sym1, expected_sym1_sib);
   EXPECT_EQ(actual_rel1, expected_rel1_sib);

   if(has_cousins) {
      EXPECT_EQ(actual_sym0_c, expected_sym0_c);
      EXPECT_EQ(actual_rel0_c, expected_rel0_c);
      EXPECT_EQ(actual_sym1_c, expected_sym1_c);
      EXPECT_EQ(actual_rel1_c, expected_rel1_c);
   } else {
      EXPECT_TRUE(actual_sym0_c.empty());
      EXPECT_TRUE(actual_rel0_c.empty());
      EXPECT_TRUE(actual_sym1_c.empty());
      EXPECT_TRUE(actual_rel1_c.empty());
   }
}
