#include "mifrost/core/batch_builder.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "mifrost/core/encoders/common/default_relations.hpp"

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

TEST(BatchBuilderTest, HeteroNamesAndEdgesAreRecorded)
{
   BatchBuilder builder;
   builder.set_graph_kind("hetero");

   std::vector< float > symbol_nodes = {0.0f, 0.0f};
   std::vector< float > atom_nodes = {1.0f, 2.0f, 3.0f, 4.0f};

   builder.add_node_features(defaults::symbol_type_id, "x", symbol_nodes, 1);
   builder.add_node_features("atom", "x", atom_nodes, 2);
   builder.set_node_names(defaults::symbol_type_id, {"o0", "o1"});
   builder.set_node_names("atom", {"a0", "a1"});
   builder.set_object_names({"o0", "o1"});

   std::vector< int64_t > src = {0, 1};
   std::vector< int64_t > dst = {0, 1};
   builder.add_edges(defaults::symbol_type_id, "0", "atom", src, dst);

   ASSERT_EQ(builder.node_names.at(defaults::symbol_type_id).size(), 2u);
   ASSERT_EQ(builder.node_names.at("atom").size(), 2u);
   ASSERT_EQ(builder.object_names.size(), 2u);

   const auto& edge_src = std::get< BatchBuilder::LongCol >(
      builder.columns.at(std::string(defaults::symbol_type_id) + "|0|atom/edge_index_0").data
   );
   const auto& edge_dst = std::get< BatchBuilder::LongCol >(
      builder.columns.at(std::string(defaults::symbol_type_id) + "|0|atom/edge_index_1").data
   );
   ASSERT_EQ(edge_src.size(), 2u);
   ASSERT_EQ(edge_dst.size(), 2u);
   EXPECT_EQ(edge_src[0], 0);
   EXPECT_EQ(edge_dst[0], 0);
   EXPECT_EQ(edge_src[1], 1);
   EXPECT_EQ(edge_dst[1], 1);
}

TEST(BatchBuilderTest, RejectsNonPositiveFeatureDimensions)
{
   BatchBuilder builder;
   const std::vector< float > data{1.0F};

   EXPECT_THROW(builder.add_node_features("node", "x", data, 0), std::invalid_argument);
   EXPECT_THROW(builder.add_node_features("node", "x", data, -1), std::invalid_argument);
   EXPECT_THROW(
      builder.add_edge_features("node", "rel", "node", "weight", data, 0), std::invalid_argument
   );
   EXPECT_TRUE(builder.node_feature_dims.empty());
   EXPECT_TRUE(builder.columns.empty());
}

TEST(BatchBuilderTest, MetadataAppendsGrowGeometrically)
{
   BatchBuilder builder;
   builder.set_object_names({"object-0"});

   for(int index = 1; index < 65; ++index) {
      builder.set_object_names({"object-" + std::to_string(index)});
   }

   ASSERT_EQ(builder.object_names.size(), 65U);
   EXPECT_GT(builder.object_names.capacity(), builder.object_names.size());
}

}  // namespace
}  // namespace mifrost
