#include "mifrost/core/successor_hgraph_encoder.hpp"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "mifrost/core/relation_formatter.hpp"
#include "test_utils.hpp"

using namespace mifrost;

namespace {

template < typename Tag >
std::unordered_set< uint32_t > collect_indices(const mimir::search::State& state)
{
   std::unordered_set< uint32_t > indices;
   for(const auto idx : state.get_atoms< Tag >()) {
      indices.insert(static_cast< uint32_t >(idx));
   }
   return indices;
}

}  // namespace

class SuccessorHGraphEncoderTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(SuccessorHGraphEncoderTest, FullModeEncodesSuccessorNodesWithSuffix)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   SuccessorHGraphEncoderEngine::Config config;
   config.successor_mode = SuccessorHGraphEncoderEngine::Mode::Full;
   SuccessorHGraphEncoderEngine engine(ctx.problem->get_domain(), config);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(ctx.root, succ_state, goals, builder);

   const auto& repos = ctx.problem->get_repositories();
   std::unordered_map< std::string, std::unordered_set< std::string > > expected;
   std::unordered_map< std::string, mifrost_test::EdgePairs > expected_edges;
   const auto index_map = mifrost_test::build_index_map(builder);
   const auto& symbol_index = index_map.at(engine.get_config().symbol_type_id);

   auto add_edges = [&](
                       const std::string& node_type,
                       const std::string& node_key,
                       const std::vector< std::string >& object_keys
                    ) {
      const auto rel_it = index_map.at(node_type).find(node_key);
      ASSERT_NE(rel_it, index_map.at(node_type).end());
      const int64_t rel_idx = rel_it->second;
      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto obj_it = symbol_index.find(object_keys[pos]);
         ASSERT_NE(obj_it, symbol_index.end());
         const int64_t obj_idx = obj_it->second;
         const std::string pos_str = std::to_string(pos);
         expected_edges[engine.get_config().symbol_type_id + "|" + pos_str + "|" + node_type]
            .emplace_back(obj_idx, rel_idx);
         expected_edges[node_type + "|" + pos_str + "|" + engine.get_config().symbol_type_id]
            .emplace_back(rel_idx, obj_idx);
      }
   };

   auto add_atom = [&](auto atom) {
      using Tag = typename std::remove_pointer_t< decltype(atom) >::Type;
      const auto predicate = atom->get_predicate();
      const std::string node_type = RelationFormatter::format_predicate(
         predicate, std::nullopt, std::nullopt, std::nullopt, config.successor_suffix
      );
      const std::string node_key = RelationFormatter::format_atom< Tag >(
         atom, config.successor_suffix
      );
      expected[node_type].insert(node_key);

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         object_keys.emplace_back(engine.get_config().nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(RelationFormatter::format_object(*obj));
         }
      }
      add_edges(node_type, node_key, object_keys);
   };

   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      succ_state.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      add_atom(atom);
   }
   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      succ_state.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      add_atom(atom);
   }

   ASSERT_FALSE(expected.empty());

   for(const auto& [node_type, nodes] : expected) {
      const auto it = builder.node_names.find(node_type);
      ASSERT_NE(it, builder.node_names.end()) << "Missing successor node type: " << node_type;
      std::unordered_set< std::string > actual(it->second.begin(), it->second.end());
      for(const auto& node_key : nodes) {
         EXPECT_TRUE(actual.contains(node_key)) << "Missing successor node key: " << node_key;
      }
   }

   for(const auto& [edge_type, expected_pairs] : expected_edges) {
      auto actual_pairs = mifrost_test::edge_pairs_for(builder, edge_type);
      auto expected_sorted = expected_pairs;
      mifrost_test::sort_edge_pairs(actual_pairs);
      mifrost_test::sort_edge_pairs(expected_sorted);
      EXPECT_EQ(actual_pairs, expected_sorted) << "Edge mismatch for " << edge_type;
   }
}

TEST_P(SuccessorHGraphEncoderTest, DeltaModeEncodesOnlyAddedRemovedAtoms)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   SuccessorHGraphEncoderEngine::Config config;
   config.successor_mode = SuccessorHGraphEncoderEngine::Mode::Delta;
   SuccessorHGraphEncoderEngine engine(ctx.problem->get_domain(), config);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(ctx.root, succ_state, goals, builder);

   const auto& repos = ctx.problem->get_repositories();
   std::unordered_map< std::string, std::unordered_set< std::string > > expected;
   std::unordered_map< std::string, mifrost_test::EdgePairs > expected_edges;
   const auto index_map = mifrost_test::build_index_map(builder);
   const auto& symbol_index = index_map.at(engine.get_config().symbol_type_id);

   auto add_edges = [&](
                       const std::string& node_type,
                       const std::string& node_key,
                       const std::vector< std::string >& object_keys
                    ) {
      const auto rel_it = index_map.at(node_type).find(node_key);
      ASSERT_NE(rel_it, index_map.at(node_type).end());
      const int64_t rel_idx = rel_it->second;
      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto obj_it = symbol_index.find(object_keys[pos]);
         ASSERT_NE(obj_it, symbol_index.end());
         const int64_t obj_idx = obj_it->second;
         const std::string pos_str = std::to_string(pos);
         expected_edges[engine.get_config().symbol_type_id + "|" + pos_str + "|" + node_type]
            .emplace_back(obj_idx, rel_idx);
         expected_edges[node_type + "|" + pos_str + "|" + engine.get_config().symbol_type_id]
            .emplace_back(rel_idx, obj_idx);
      }
   };

   auto handle_delta = [&](auto tag_ptr) {
      using Tag = std::remove_pointer_t< decltype(tag_ptr) >;
      const auto cur = collect_indices< Tag >(ctx.root);
      const auto suc = collect_indices< Tag >(succ_state);

      for(const auto idx : suc) {
         if(cur.contains(idx)) {
            continue;
         }
         const auto atom = repos.get_ground_atom< Tag >(idx);
         const auto predicate = atom->get_predicate();
         if(predicate->get_arity() == 0 && not config.add_nullary_predicates) {
            continue;
         }
         const std::string node_type = RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, true, config.successor_suffix
         );
         const std::string atom_str = RelationFormatter::format_atom< Tag >(
            atom, config.successor_suffix
         );
         const std::string node_key = std::string(RelationFormatter::polarity_prefix(true))
                                      + atom_str;
         expected[node_type].insert(node_key);

         std::vector< std::string > object_keys;
         if(predicate->get_arity() == 0) {
            object_keys.emplace_back(engine.get_config().nullary_object_name);
         } else {
            for(const auto& obj : atom->get_objects()) {
               object_keys.emplace_back(RelationFormatter::format_object(*obj));
            }
         }
         add_edges(node_type, node_key, object_keys);
      }
      for(const auto idx : cur) {
         if(suc.contains(idx)) {
            continue;
         }
         const auto atom = repos.get_ground_atom< Tag >(idx);
         const auto predicate = atom->get_predicate();
         if(predicate->get_arity() == 0 && not config.add_nullary_predicates) {
            continue;
         }
         const std::string node_type = RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, false, config.successor_suffix
         );
         const std::string atom_str = RelationFormatter::format_atom< Tag >(
            atom, config.successor_suffix
         );
         const std::string node_key = std::string(RelationFormatter::polarity_prefix(false))
                                      + atom_str;
         expected[node_type].insert(node_key);

         std::vector< std::string > object_keys;
         if(predicate->get_arity() == 0) {
            object_keys.emplace_back(engine.get_config().nullary_object_name);
         } else {
            for(const auto& obj : atom->get_objects()) {
               object_keys.emplace_back(RelationFormatter::format_object(*obj));
            }
         }
         add_edges(node_type, node_key, object_keys);
      }
   };

   handle_delta((mimir::formalism::FluentTag*) nullptr);
   handle_delta((mimir::formalism::DerivedTag*) nullptr);

   std::unordered_set< std::string > expected_types;
   for(const auto& [node_type, _] : expected) {
      expected_types.insert(node_type);
   }

   for(const auto& [node_type, nodes] : expected) {
      const auto it = builder.node_names.find(node_type);
      ASSERT_NE(it, builder.node_names.end()) << "Missing delta node type: " << node_type;
      std::unordered_set< std::string > actual(it->second.begin(), it->second.end());
      EXPECT_EQ(actual.size(), nodes.size()) << "Unexpected delta nodes for " << node_type;
      for(const auto& node_key : nodes) {
         EXPECT_TRUE(actual.contains(node_key)) << "Missing delta node key: " << node_key;
      }
   }

   for(const auto& [node_type, _] : builder.node_names) {
      if(node_type.find(config.successor_suffix) != std::string::npos) {
         EXPECT_TRUE(expected_types.contains(node_type))
            << "Unexpected successor-suffix node type: " << node_type;
      }
   }

   for(const auto& [edge_type, expected_pairs] : expected_edges) {
      auto actual_pairs = mifrost_test::edge_pairs_for(builder, edge_type);
      auto expected_sorted = expected_pairs;
      mifrost_test::sort_edge_pairs(actual_pairs);
      mifrost_test::sort_edge_pairs(expected_sorted);
      EXPECT_EQ(actual_pairs, expected_sorted) << "Edge mismatch for " << edge_type;
   }
}

INSTANTIATE_TEST_SUITE_P(
   ParityDomains,
   SuccessorHGraphEncoderTest,
   ::testing::ValuesIn(mifrost_test::kParityDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);
