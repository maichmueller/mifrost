#include <gtest/gtest.h>

#include "mifrost/core/encoders/hetero/hgraph_stream_encoder.hpp"
#include "test_utils.hpp"

using namespace mifrost;

class HGraphStreamContractTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(HGraphStreamContractTest, BatchBuilderAccumulatesPtrsAcrossGraphs)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   HGraphEncoderEngine engine(ctx.problem->get_domain());
   BatchBuilder builder;
   builder.set_graph_kind("hetero");

   engine.encode(ctx.root, builder);
   builder.next_graph();
   engine.encode(succ_state, builder);
   builder.next_graph();

   const auto& cfg = engine.get_config();
   const auto ptr_it = builder.ptrs.find(cfg.symbol_type_id);
   ASSERT_NE(ptr_it, builder.ptrs.end());
   const auto& ptr = ptr_it->second;
   ASSERT_EQ(ptr.size(), 3u);
   EXPECT_EQ(ptr.front(), 0);
   EXPECT_LE(ptr[1], ptr[2]);

   const auto names_it = builder.node_names.find(cfg.symbol_type_id);
   ASSERT_NE(names_it, builder.node_names.end());
   EXPECT_EQ(ptr.back(), static_cast< int64_t >(names_it->second.size()));
}

TEST_P(HGraphStreamContractTest, PerGraphCountsMatchIndividualEncodes)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   HGraphEncoderEngine engine(ctx.problem->get_domain());

   auto encode_counts = [&](const mimir::search::State& state) {
      BatchBuilder tmp_builder;
      tmp_builder.set_graph_kind("hetero");
      engine.encode(state, tmp_builder);
      std::unordered_map< std::string, int64_t > counts;
      for(const auto& [node_type, names] : tmp_builder.node_names) {
         counts.emplace(node_type, static_cast< int64_t >(names.size()));
      }
      return counts;
   };

   const auto counts_root = encode_counts(ctx.root);
   const auto counts_succ = encode_counts(succ_state);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   engine.encode(ctx.root, builder);
   builder.next_graph();
   engine.encode(succ_state, builder);
   builder.next_graph();

   std::unordered_set< std::string > node_types;
   for(const auto& [node_type, _] : counts_root) {
      node_types.insert(node_type);
   }
   for(const auto& [node_type, _] : counts_succ) {
      node_types.insert(node_type);
   }

   for(const auto& node_type : node_types) {
      const int64_t expected_root = counts_root.contains(node_type) ? counts_root.at(node_type) : 0;
      const int64_t expected_succ = counts_succ.contains(node_type) ? counts_succ.at(node_type) : 0;
      const auto ptr_it = builder.ptrs.find(node_type);
      if(ptr_it == builder.ptrs.end()) {
         EXPECT_EQ(expected_root, 0) << "Missing ptrs for node type " << node_type;
         EXPECT_EQ(expected_succ, 0) << "Missing ptrs for node type " << node_type;
         continue;
      }
      const auto& ptr = ptr_it->second;
      ASSERT_EQ(ptr.size(), 3u);
      EXPECT_EQ(ptr[1] - ptr[0], expected_root) << "Count mismatch for " << node_type;
      EXPECT_EQ(ptr[2] - ptr[1], expected_succ) << "Count mismatch for " << node_type;

      const auto names_it = builder.node_names.find(node_type);
      ASSERT_NE(names_it, builder.node_names.end());
      EXPECT_EQ(ptr.back(), static_cast< int64_t >(names_it->second.size()));
   }
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   HGraphStreamContractTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);
