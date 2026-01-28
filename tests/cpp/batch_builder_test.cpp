#include "mifrost/core/batch_builder.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace mifrost {
namespace {

TEST(BatchBuilderTest, AddNodesUpdatesPtr)
{
   BatchBuilder builder;

   std::vector< float > nodes1 = {1.0f, 2.0f};
   builder.add_node_features("node", "x", nodes1, 1);
   builder.next_graph();

   std::vector< float > nodes2 = {3.0f, 4.0f, 5.0f};
   builder.add_node_features("node", "x", nodes2, 1);
   builder.next_graph();

   const auto& ptr = builder.ptrs.at("node");
   ASSERT_EQ(ptr.size(), 3u);
   EXPECT_EQ(ptr[0], 0);
   EXPECT_EQ(ptr[1], 2);
   EXPECT_EQ(ptr[2], 5);
}

TEST(BatchBuilderTest, AddEdgesAppliesOffsets)
{
   BatchBuilder builder;

   std::vector< float > nodes1 = {1.0f, 2.0f, 3.0f};
   builder.add_node_features("node", "x", nodes1, 1);

   std::vector< int64_t > src1 = {0, 1};
   std::vector< int64_t > dst1 = {1, 2};
   builder.add_edges("node", "rel", "node", src1, dst1);
   builder.next_graph();

   std::vector< float > nodes2 = {4.0f, 5.0f};
   builder.add_node_features("node", "x", nodes2, 1);

   std::vector< int64_t > src2 = {0};
   std::vector< int64_t > dst2 = {1};
   builder.add_edges("node", "rel", "node", src2, dst2);

   const auto& edge_src = std::get< BatchBuilder::LongCol >(
      builder.columns.at("node|rel|node/edge_index_0").data
   );
   const auto& edge_dst = std::get< BatchBuilder::LongCol >(
      builder.columns.at("node|rel|node/edge_index_1").data
   );

   ASSERT_EQ(edge_src.size(), 3u);
   ASSERT_EQ(edge_dst.size(), 3u);
   EXPECT_EQ(edge_src[0], 0);
   EXPECT_EQ(edge_src[1], 1);
   EXPECT_EQ(edge_src[2], 3);
   EXPECT_EQ(edge_dst[0], 1);
   EXPECT_EQ(edge_dst[1], 2);
   EXPECT_EQ(edge_dst[2], 4);
}

}  // namespace
}  // namespace mifrost
