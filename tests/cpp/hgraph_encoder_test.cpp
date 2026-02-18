#include <gtest/gtest.h>

#include <algorithm>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/relation_formatter.hpp"
#include "test_utils.hpp"

namespace {

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

using EdgePairs = mifrost_test::EdgePairs;
using EdgeMap = std::unordered_map< std::string, EdgePairs >;
using NodeSetMap = std::unordered_map< std::string, std::unordered_set< std::string > >;

}  // namespace

class HGraphEncoderTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(HGraphEncoderTest, InitialStateIncludesObjectsFactsAndEdges)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain());
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("hetero");
   engine.encode(ctx.root, builder);

   const auto& cfg = engine.get_config();
   ASSERT_EQ(builder.graph_kind, "hetero");

   const auto symbol_it = builder.node_names.find(cfg.symbol_type_id);
   ASSERT_NE(symbol_it, builder.node_names.end());
   const auto& symbol_names = symbol_it->second;
   std::unordered_set< std::string > symbol_set(symbol_names.begin(), symbol_names.end());
   const auto index_map = mifrost_test::build_index_map(builder);

   for(const auto& obj : ctx.problem->get_problem_and_domain_objects()) {
      const std::string key = mifrost::RelationFormatter::format_object(*obj);
      EXPECT_TRUE(symbol_set.contains(key)) << "Missing object node: " << key;
   }
   if(cfg.add_nullary_predicates) {
      EXPECT_TRUE(symbol_set.contains(cfg.nullary_object_name));
   }

   NodeSetMap expected_nodes;
   EdgeMap expected_edges;
   const auto& repos = ctx.problem->get_repositories();
   const auto& symbol_index = index_map.at(cfg.symbol_type_id);

   auto node_index = [&](const std::string& node_type, const std::string& node_key) -> int64_t {
      const auto type_it = index_map.find(node_type);
      EXPECT_NE(type_it, index_map.end()) << "Missing node type: " << node_type;
      if(type_it == index_map.end()) {
         return -1;
      }
      const auto key_it = type_it->second.find(node_key);
      EXPECT_NE(key_it, type_it->second.end())
         << "Missing node key '" << node_key << "' for type '" << node_type << "'";
      if(key_it == type_it->second.end()) {
         return -1;
      }
      return key_it->second;
   };

   auto handle_atom = [&](auto atom) {
      using Tag = typename std::remove_pointer_t< decltype(atom) >::Type;
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0 && not cfg.add_nullary_predicates) {
         return;
      }
      const std::string node_type = mifrost::RelationFormatter::format_predicate(predicate);
      const std::string node_key = mifrost::RelationFormatter::format_atom< Tag >(atom);
      expected_nodes[node_type].insert(node_key);

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         object_keys.emplace_back(cfg.nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(mifrost::RelationFormatter::format_object(*obj));
         }
      }

      const auto relation_idx = node_index(node_type, node_key);
      if(relation_idx < 0) {
         return;
      }

      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const std::string pos_str = std::to_string(pos);
         const auto sym_it = symbol_index.find(object_keys[pos]);
         EXPECT_NE(sym_it, symbol_index.end())
            << "Missing symbol node '" << object_keys[pos] << "'";
         if(sym_it == symbol_index.end()) {
            continue;
         }
         const int64_t obj_idx = sym_it->second;
         expected_edges[edge_key(cfg.symbol_type_id, pos_str, node_type)].emplace_back(
            obj_idx, relation_idx
         );
         expected_edges[edge_key(node_type, pos_str, cfg.symbol_type_id)].emplace_back(
            relation_idx, obj_idx
         );
      }
   };

   if(cfg.include_static) {
      const auto& literals = ctx.problem->get_initial_literals< mimir::formalism::StaticTag >();
      for(const auto& literal : literals) {
         if(literal->get_polarity()) {
            handle_atom(literal->get_atom());
         }
      }
   }

   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      ctx.root.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      handle_atom(atom);
   }

   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      ctx.root.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      handle_atom(atom);
   }

   for(const auto& [node_type, nodes] : expected_nodes) {
      const auto it = builder.node_names.find(node_type);
      ASSERT_NE(it, builder.node_names.end()) << "Missing node type: " << node_type;
      std::unordered_set< std::string > actual(it->second.begin(), it->second.end());
      for(const auto& node_key : nodes) {
         EXPECT_TRUE(actual.contains(node_key))
            << "Missing node '" << node_key << "' for type '" << node_type << "'";
      }
   }

   for(const auto& [edge_type, expected] : expected_edges) {
      auto actual = mifrost_test::edge_pairs_for(builder, edge_type);
      auto expected_pairs = expected;
      mifrost_test::sort_edge_pairs(actual);
      mifrost_test::sort_edge_pairs(expected_pairs);
      EXPECT_EQ(actual, expected_pairs) << "Edge mismatch for " << edge_type;
   }
}

TEST_P(HGraphEncoderTest, NullaryPredicatesIncludeNullaryObjectWhenEnabled)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   mifrost::HGraphEncoderEngine::Config config;
   config.add_nullary_predicates = true;
   mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain(), config);
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("hetero");
   engine.encode(ctx.root, builder);

   bool has_nullary = false;
   const auto& repos = ctx.problem->get_repositories();
   auto check_atom = [&](auto atom) {
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0) {
         has_nullary = true;
      }
   };

   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      ctx.root.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      check_atom(atom);
   }
   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      ctx.root.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      check_atom(atom);
   }
   if(config.include_static) {
      const auto& literals = ctx.problem->get_initial_literals< mimir::formalism::StaticTag >();
      for(const auto& literal : literals) {
         if(literal->get_polarity()) {
            check_atom(literal->get_atom());
         }
      }
   }

   if(not has_nullary) {
      GTEST_SKIP() << "Domain has no nullary predicates in the initial state.";
   }

   const auto symbol_it = builder.node_names.find(config.symbol_type_id);
   ASSERT_NE(symbol_it, builder.node_names.end());
   const auto& symbol_names = symbol_it->second;
   EXPECT_NE(
      std::find(symbol_names.begin(), symbol_names.end(), config.nullary_object_name),
      symbol_names.end()
   );
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   HGraphEncoderTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

class HGraphEncoderInputsTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(HGraphEncoderInputsTest, GoalsActionsSubgoalsMatchConfig)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   const auto all_goals_static = ctx.problem->get_goal_literals< mimir::formalism::StaticTag >();
   const auto all_goals_fluent = ctx.problem->get_goal_literals< mimir::formalism::FluentTag >();
   const auto all_goals_derived = ctx.problem->get_goal_literals< mimir::formalism::DerivedTag >();

   const std::vector< bool > flags = {false, true};
   for(const bool include_goals : flags) {
      for(const bool include_actions : flags) {
         for(const bool include_subgoals : flags) {
            const bool use_goals = include_goals || include_subgoals;

            mifrost::HGraphEncoderEngine::Config config;
            config.ignore_actions = not include_actions;
            config.max_goal_level = 1;

            mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain(), config);
            mifrost::BatchBuilder builder;
            builder.set_graph_kind("hetero");

            mifrost::GoalInputs inputs;
            if(use_goals) {
               inputs = mifrost_test::make_goal_inputs(ctx.problem);
               if(include_subgoals) {
                  if(not inputs.fluent_goals.empty()) {
                     inputs.fluent_goal_levels[inputs.fluent_goals.front()] = 1;
                  } else if(not inputs.derived_goals.empty()) {
                     inputs.derived_goal_levels[inputs.derived_goals.front()] = 1;
                  } else if(not inputs.static_goals.empty()) {
                     inputs.static_goal_levels[inputs.static_goals.front()] = 1;
                  }
               }
            }

            std::vector< mimir::formalism::GroundAction > actions;
            if(include_actions && not ctx.actions.empty()) {
               actions.emplace_back(ctx.actions.front());
            }

            if(use_goals || include_actions) {
               engine.encode(ctx.root, inputs, actions, builder);
            } else {
               engine.encode(ctx.root, builder);
            }

            if(include_actions && not actions.empty()) {
               const auto action = actions.front();
               const std::string node_type = mifrost::RelationFormatter::format_action_schema(
                  *action->get_action()
               );
               const std::string node_key = mifrost::RelationFormatter::format_action(action);
               const auto it = builder.node_names.find(node_type);
               ASSERT_NE(it, builder.node_names.end()) << "Missing action node type: " << node_type;
               EXPECT_NE(
                  std::find(it->second.begin(), it->second.end(), node_key), it->second.end()
               );
            }

            if(use_goals) {
               bool checked = false;
               auto check_goal = [&](auto literal, int level) {
                  const auto atom = literal->get_atom();
                  const auto predicate = atom->get_predicate();
                  const auto node_type = mifrost::RelationFormatter::format_predicate(
                     predicate, mifrost::GoalLevel(level), std::nullopt, literal->get_polarity()
                  );
                  const auto node_key = mifrost::RelationFormatter::format_literal(
                     literal, mifrost::GoalLevel(level), std::nullopt, literal->get_polarity(), ""
                  );
                  const auto it = builder.node_names.find(node_type);
                  ASSERT_NE(it, builder.node_names.end())
                     << "Missing goal node type: " << node_type;
                  EXPECT_NE(
                     std::find(it->second.begin(), it->second.end(), node_key), it->second.end()
                  );
                  checked = true;
               };

               if(not inputs.fluent_goals.empty()) {
                  const int level = inputs.fluent_goal_levels.at(inputs.fluent_goals.front());
                  check_goal(inputs.fluent_goals.front(), level);
               } else if(not inputs.derived_goals.empty()) {
                  const int level = inputs.derived_goal_levels.at(inputs.derived_goals.front());
                  check_goal(inputs.derived_goals.front(), level);
               } else if(not inputs.static_goals.empty()) {
                  const int level = inputs.static_goal_levels.at(inputs.static_goals.front());
                  check_goal(inputs.static_goals.front(), level);
               }

               if(not checked) {
                  GTEST_SKIP() << "No goals present in problem.";
               }
            }
         }
      }
   }
}

INSTANTIATE_TEST_SUITE_P(
   ParityDomains,
   HGraphEncoderInputsTest,
   ::testing::ValuesIn(mifrost_test::kParityDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

class HGraphEncoderFlagVariantsTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(HGraphEncoderFlagVariantsTest, FlagVariantsRespectStaticAndLganConfig)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   const std::vector< bool > flags = {false, true};
   for(const bool include_static : flags) {
      for(const bool include_lgan_edges : flags) {
         for(const bool add_nullary_predicates : flags) {
            for(const bool support_literals : flags) {
               mifrost::HGraphEncoderEngine::Config config;
               config.ignore_actions = true;
               config.include_static = include_static;
               config.include_lgan_edges = include_lgan_edges;
               config.add_nullary_predicates = add_nullary_predicates;
               config.support_literals = support_literals;
               config.max_goal_level = 1;

               mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain(), config);
               mifrost::BatchBuilder builder;
               builder.set_graph_kind("hetero");

               auto goals = mifrost_test::make_goal_inputs(ctx.problem);
               engine.encode(ctx.root, goals, {}, builder);

               if(not include_lgan_edges) {
                  for(const auto& [key, _] : builder.columns) {
                     EXPECT_EQ(key.find(config.lgan_nn_edge_pos), std::string::npos)
                        << "Unexpected lgan edge column: " << key;
                  }
               }

               bool has_static = false;
               const auto& literals = ctx.problem
                                         ->get_initial_literals< mimir::formalism::StaticTag >();
               for(const auto& literal : literals) {
                  if(literal->get_polarity()) {
                     has_static = true;
                     if(not include_static) {
                        const auto atom = literal->get_atom();
                        const auto node_type = mifrost::RelationFormatter::format_predicate(
                           atom->get_predicate()
                        );
                        const auto node_key = mifrost::RelationFormatter::format_atom<
                           mimir::formalism::StaticTag >(atom);
                        const auto it = builder.node_names.find(node_type);
                        if(it != builder.node_names.end()) {
                           EXPECT_EQ(
                              std::find(it->second.begin(), it->second.end(), node_key),
                              it->second.end()
                           );
                        }
                     }
                     break;
                  }
               }
               if(add_nullary_predicates) {
                  bool has_nullary = false;
                  for(const auto& literal : literals) {
                     if(literal->get_polarity()
                        && literal->get_atom()->get_predicate()->get_arity() == 0) {
                        has_nullary = true;
                        break;
                     }
                  }
                  if(has_nullary) {
                     const auto symbol_it = builder.node_names.find(config.symbol_type_id);
                     ASSERT_NE(symbol_it, builder.node_names.end());
                     EXPECT_NE(
                        std::find(
                           symbol_it->second.begin(),
                           symbol_it->second.end(),
                           config.nullary_object_name
                        ),
                        symbol_it->second.end()
                     );
                  }
               }
               (void) has_static;
            }
         }
      }
   }
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   HGraphEncoderFlagVariantsTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

TEST_P(HGraphEncoderTest, GoalSatisfactionEdgesMatchFacts)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   mifrost::HGraphEncoderEngine::Config config;
   config.ignore_actions = true;
   config.max_goal_level = 1;
   config.goal_satisfaction_derivations = {
      mifrost::GoalSatisfaction::satisfied,
      mifrost::GoalSatisfaction::unsatisfied,
   };

   mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain(), config);
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("hetero");

   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   engine.encode(ctx.root, goals, {}, builder);

   const auto index_map = mifrost_test::build_index_map(builder);
   const auto& symbol_index = index_map.at(config.symbol_type_id);

   mifrost::hash_set< std::string > fact_keys;
   const auto& repos = ctx.problem->get_repositories();

   if(config.include_static) {
      const auto& literals = ctx.problem->get_initial_literals< mimir::formalism::StaticTag >();
      for(const auto& literal : literals) {
         if(literal->get_polarity()) {
            fact_keys.insert(
               mifrost::RelationFormatter::format_atom< mimir::formalism::StaticTag >(
                  literal->get_atom()
               )
            );
         }
      }
   }

   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      ctx.root.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      fact_keys.insert(
         mifrost::RelationFormatter::format_atom< mimir::formalism::FluentTag >(atom)
      );
   }

   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      ctx.root.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      fact_keys.insert(
         mifrost::RelationFormatter::format_atom< mimir::formalism::DerivedTag >(atom)
      );
   }

   bool checked = false;
   auto check_goal = [&](auto literal, int level) {
      const auto atom = literal->get_atom();
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0 && not config.add_nullary_predicates) {
         return;
      }
      const auto key = mifrost::RelationFormatter::format_atom(atom);
      const bool satisfied = fact_keys.contains(key) == literal->get_polarity();
      const mifrost::GoalSatisfaction sat = satisfied ? mifrost::GoalSatisfaction::satisfied
                                                      : mifrost::GoalSatisfaction::unsatisfied;
      if(not config.goal_satisfaction_derivations.contains(sat)) {
         return;
      }

      const mifrost::GoalLevel level_obj(level);
      const std::string node_type = mifrost::RelationFormatter::format_predicate(
         predicate, level_obj, sat, literal->get_polarity(), ""
      );
      const std::string node_key = mifrost::RelationFormatter::format_literal(
         literal, level_obj, sat, std::nullopt, ""
      );

      auto type_it = index_map.find(node_type);
      ASSERT_NE(type_it, index_map.end()) << "Missing goal satisfaction node type: " << node_type;
      auto rel_it = type_it->second.find(node_key);
      ASSERT_NE(rel_it, type_it->second.end()) << "Missing goal satisfaction node: " << node_key;

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         object_keys.emplace_back(config.nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(mifrost::RelationFormatter::format_object(*obj));
         }
      }

      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto sym_it = symbol_index.find(object_keys[pos]);
         ASSERT_NE(sym_it, symbol_index.end()) << "Missing object node: " << object_keys[pos];
         const int64_t obj_idx = sym_it->second;
         const int64_t rel_idx = rel_it->second;
         const std::string pos_str = std::to_string(pos);

         const auto edge_type_sym = config.symbol_type_id + "|" + pos_str + "|" + node_type;
         const auto edge_type_rel = node_type + "|" + pos_str + "|" + config.symbol_type_id;

         auto actual_sym = mifrost_test::edge_pairs_for(builder, edge_type_sym);
         auto actual_rel = mifrost_test::edge_pairs_for(builder, edge_type_rel);
         mifrost_test::sort_edge_pairs(actual_sym);
         mifrost_test::sort_edge_pairs(actual_rel);

         EXPECT_NE(
            std::find(actual_sym.begin(), actual_sym.end(), std::pair{obj_idx, rel_idx}),
            actual_sym.end()
         );
         EXPECT_NE(
            std::find(actual_rel.begin(), actual_rel.end(), std::pair{rel_idx, obj_idx}),
            actual_rel.end()
         );
      }
      checked = true;
   };

   if(not goals.fluent_goals.empty()) {
      const int level = goals.fluent_goal_levels.at(goals.fluent_goals.front());
      check_goal(goals.fluent_goals.front(), level);
   }
   if(not checked && not goals.derived_goals.empty()) {
      const int level = goals.derived_goal_levels.at(goals.derived_goals.front());
      check_goal(goals.derived_goals.front(), level);
   }
   if(not checked && not goals.static_goals.empty()) {
      const int level = goals.static_goal_levels.at(goals.static_goals.front());
      check_goal(goals.static_goals.front(), level);
   }

   if(not checked) {
      GTEST_SKIP() << "No goals available to assert goal satisfaction edges.";
   }
}

TEST_P(HGraphEncoderTest, StreamEncodingProducesCorrectSymbolPtrs)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, _succ_action] = mifrost_test::find_successor(ctx);

   std::vector< mimir::search::State > states = {ctx.root, succ_state};

   mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain());
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);

   std::vector< int64_t > expected_counts;
   expected_counts.reserve(states.size());
   for(const auto& state : states) {
      mifrost::BatchBuilder single_builder;
      single_builder.set_graph_kind("hetero");
      engine.encode(state, goals, {}, single_builder);
      const auto symbol_it = single_builder.node_names.find(engine.get_config().symbol_type_id);
      ASSERT_NE(symbol_it, single_builder.node_names.end());
      expected_counts.push_back(static_cast< int64_t >(symbol_it->second.size()));
   }

   mifrost::BatchBuilder stream_builder;
   stream_builder.set_graph_kind("hetero");
   for(const auto& state : states) {
      engine.encode(state, goals, {}, stream_builder);
      stream_builder.next_graph();
   }

   const auto ptr_it = stream_builder.ptrs.find(engine.get_config().symbol_type_id);
   ASSERT_NE(ptr_it, stream_builder.ptrs.end());
   const auto& ptrs = ptr_it->second;
   ASSERT_EQ(ptrs.size(), expected_counts.size() + 1);

   for(size_t i = 0; i < expected_counts.size(); ++i) {
      const auto count = ptrs[i + 1] - ptrs[i];
      EXPECT_EQ(count, expected_counts[i]) << "Mismatch at graph index " << i;
   }
}
