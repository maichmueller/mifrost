#include "mifrost/core/encoders/homo/color_encoder.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mifrost/core/encoders/common/relation_formatter.hpp"
#include "test_utils.hpp"

using namespace mifrost;

namespace {

std::unordered_set< std::string > predicate_names_from_state(
   const mimir::search::State& state,
   const mimir::formalism::Problem& problem
)
{
   std::unordered_set< std::string > names;
   const auto& repos = problem->get_repositories();

   if(problem->get_domain()->get_predicates< mimir::formalism::StaticTag >().size() > 0) {
      const auto& literals = problem->get_initial_literals< mimir::formalism::StaticTag >();
      for(const auto& literal : literals) {
         if(literal->get_polarity()) {
            names.insert(RelationFormatter::format_predicate(literal->get_atom()->get_predicate()));
         }
      }
   }

   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      state.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      names.insert(RelationFormatter::format_predicate(atom->get_predicate()));
   }

   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      state.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      names.insert(RelationFormatter::format_predicate(atom->get_predicate()));
   }

   return names;
}

}  // namespace

class ColorEncoderTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(ColorEncoderTest, ConfigVariantsMatchSchemaAndColumns)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);

   const std::vector< bool > flags = {false, true};
   for(const bool edge_features : flags) {
      for(const bool predicate_nodes : flags) {
         ColorEncoderEngine::Config config;
         config.edge_features = edge_features;
         config.enable_global_predicate_nodes = predicate_nodes;

         ColorEncoderEngine engine(ctx.problem->get_domain(), config);
         BatchBuilder builder;
         builder.set_graph_kind("homo");
         engine.encode(ctx.root, builder);

         EXPECT_EQ(builder.graph_kind, "homo");
         EXPECT_EQ(builder.schema_flags["edge_features"], edge_features);
         EXPECT_EQ(builder.schema_flags["predicate_nodes"], predicate_nodes);

         const auto it = builder.node_names.find("node");
         ASSERT_NE(it, builder.node_names.end());
         const auto& node_names = it->second;
         std::unordered_set< std::string > name_set(node_names.begin(), node_names.end());
         const auto index_map = mifrost_test::build_index_map(builder);

         auto node_index = [&](const std::string& key) -> int64_t {
            const auto type_it = index_map.find("node");
            if(type_it == index_map.end()) {
               ADD_FAILURE() << "Missing node type: node";
               return -1;
            }
            const auto key_it = type_it->second.find(key);
            if(key_it == type_it->second.end()) {
               ADD_FAILURE() << "Missing node: " << key;
               return -1;
            }
            return key_it->second;
         };

         for(const auto& obj : ctx.problem->get_problem_and_domain_objects()) {
            const std::string key = RelationFormatter::format_object(*obj);
            EXPECT_TRUE(name_set.contains(key)) << "Missing object node: " << key;
         }

         if(predicate_nodes) {
            const auto expected_predicates = predicate_names_from_state(ctx.root, ctx.problem);
            for(const auto& pred_name : expected_predicates) {
               EXPECT_TRUE(name_set.contains(pred_name)) << "Missing predicate node: " << pred_name;
            }
         }

         const auto edge_src_it = builder.columns.find("node|edge|node/edge_index_0");
         const auto edge_dst_it = builder.columns.find("node|edge|node/edge_index_1");
         ASSERT_NE(edge_src_it, builder.columns.end());
         ASSERT_NE(edge_dst_it, builder.columns.end());

         std::vector< std::pair< int64_t, int64_t > > expected_edges;
         std::unordered_set< std::string > predicate_self_edges;

         auto add_edge = [&](int64_t src, int64_t dst) {
            if(src < 0 || dst < 0) {
               return;
            }
            expected_edges.emplace_back(src, dst);
         };

         auto add_predicate_self_edge = [&](const std::string& predicate_node, int64_t idx) {
            if(not edge_features) {
               return;
            }
            if(predicate_self_edges.contains(predicate_node)) {
               return;
            }
            predicate_self_edges.insert(predicate_node);
            add_edge(idx, idx);
         };

         auto encode_atom = [&](const auto& atom) {
            const auto predicate = atom->get_predicate();
            const auto arity = predicate->get_arity();

            std::optional< std::string > predicate_node;
            std::optional< int64_t > predicate_idx;
            if(predicate_nodes) {
               predicate_node = RelationFormatter::format_predicate(
                  predicate, std::nullopt, std::nullopt, std::nullopt, ""
               );
               predicate_idx = node_index(*predicate_node);
               add_predicate_self_edge(*predicate_node, *predicate_idx);
            }

            const std::string base_name = RelationFormatter::format_atom(atom);

            if(arity == 0) {
               if(edge_features) {
                  const auto idx = node_index(base_name);
                  add_edge(idx, idx);
               }
               return;
            }

            std::optional< int64_t > prev_item_idx;
            int pos = 0;
            for(const auto& obj : atom->get_objects()) {
               const std::string object_node = RelationFormatter::format_object(*obj);
               const std::string pos_name = fmt::format("{}:{}", base_name, pos);

               if(edge_features) {
                  const auto obj_idx = node_index(object_node);
                  const auto item_idx = node_index(base_name);
                  add_edge(obj_idx, item_idx);
                  if(predicate_idx.has_value() && pos == 0) {
                     add_edge(*predicate_idx, item_idx);
                  }
               } else {
                  const auto obj_idx = node_index(object_node);
                  const auto item_idx = node_index(pos_name);
                  add_edge(obj_idx, item_idx);
                  if(prev_item_idx.has_value()) {
                     add_edge(*prev_item_idx, item_idx);
                  } else if(predicate_idx.has_value()) {
                     add_edge(*predicate_idx, item_idx);
                  }
                  prev_item_idx = item_idx;
               }
               ++pos;
            }
         };

         auto encode_literal = [&](const auto& literal, int goal_level) {
            const auto atom = literal->get_atom();
            const auto predicate = atom->get_predicate();
            const auto arity = predicate->get_arity();
            const bool polarity = literal->get_polarity();

            std::optional< std::string > predicate_node;
            std::optional< int64_t > predicate_idx;
            if(predicate_nodes) {
               const GoalLevel level(static_cast< std::size_t >(goal_level));
               predicate_node = RelationFormatter::format_predicate(
                  predicate, level, std::nullopt, polarity, ""
               );
               predicate_idx = node_index(*predicate_node);
               add_predicate_self_edge(*predicate_node, *predicate_idx);
            }

            const std::string base_name = [&]() {
               const GoalLevel level(static_cast< std::size_t >(goal_level));
               return RelationFormatter::format_literal(literal, level, std::nullopt, polarity, "");
            }();

            if(arity == 0) {
               if(edge_features) {
                  const auto idx = node_index(base_name);
                  add_edge(idx, idx);
               }
               return;
            }

            std::optional< int64_t > prev_item_idx;
            int pos = 0;
            for(const auto& obj : atom->get_objects()) {
               const std::string object_node = RelationFormatter::format_object(*obj);
               const std::string pos_name = fmt::format("{}:{}", base_name, pos);

               if(edge_features) {
                  const auto obj_idx = node_index(object_node);
                  const auto item_idx = node_index(base_name);
                  add_edge(obj_idx, item_idx);
                  if(predicate_idx.has_value() && pos == 0) {
                     add_edge(*predicate_idx, item_idx);
                  }
               } else {
                  const auto obj_idx = node_index(object_node);
                  const auto item_idx = node_index(pos_name);
                  add_edge(obj_idx, item_idx);
                  if(prev_item_idx.has_value()) {
                     add_edge(*prev_item_idx, item_idx);
                  } else if(predicate_idx.has_value()) {
                     add_edge(*predicate_idx, item_idx);
                  }
                  prev_item_idx = item_idx;
               }
               ++pos;
            }
         };

         if(ctx.problem->get_domain()->get_predicates< mimir::formalism::StaticTag >().size() > 0) {
            const auto& literals = ctx.problem
                                      ->get_initial_literals< mimir::formalism::StaticTag >();
            for(const auto& literal : literals) {
               if(literal->get_polarity()) {
                  encode_atom(literal->get_atom());
               }
            }
         }

         const auto& repos = ctx.problem->get_repositories();
         const auto fluent_atoms = repos
                                      .get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
                                         ctx.root.get_atoms< mimir::formalism::FluentTag >()
                                      );
         for(const auto& atom : fluent_atoms) {
            encode_atom(atom);
         }
         const auto
            derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
               ctx.root.get_atoms< mimir::formalism::DerivedTag >()
            );
         for(const auto& atom : derived_atoms) {
            encode_atom(atom);
         }

         auto goals = mifrost_test::make_goal_inputs(ctx.problem);
         for(const auto& literal : goals.static_goals) {
            const int level = goals.static_goal_levels.at(literal);
            encode_literal(literal, level);
         }
         for(const auto& literal : goals.fluent_goals) {
            const int level = goals.fluent_goal_levels.at(literal);
            encode_literal(literal, level);
         }
         for(const auto& literal : goals.derived_goals) {
            const int level = goals.derived_goal_levels.at(literal);
            encode_literal(literal, level);
         }

         auto actual_edges = mifrost_test::edge_pairs_for(builder, "node|edge|node");
         mifrost_test::sort_edge_pairs(actual_edges);
         mifrost_test::sort_edge_pairs(expected_edges);
         EXPECT_EQ(actual_edges, expected_edges);

         if(edge_features) {
            const auto edge_attr_it = builder.columns.find("node|edge|node/edge_attr");
            ASSERT_NE(edge_attr_it, builder.columns.end());
            EXPECT_EQ(builder.columns.count("node/x"), 0u);
         } else {
            const auto col_it = builder.columns.find("node/x");
            ASSERT_NE(col_it, builder.columns.end());
            const auto& col = std::get< BatchBuilder::FloatCol >(col_it->second.data);
            EXPECT_EQ(col.size(), node_names.size());
            EXPECT_EQ(builder.columns.count("node|edge|node/edge_attr"), 0u);
         }
      }
   }
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   ColorEncoderTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);
