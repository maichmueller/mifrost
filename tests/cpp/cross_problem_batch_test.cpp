/**
 * @file cross_problem_batch_test.cpp
 * @brief One encoder, several problems of its domain, one batch.
 *
 * This is the architectural contract of the schema/problem context split. An
 * encoder is built from a domain and holds only a `SemanticSchemaContext`; the
 * object table, static facts and goals travel with each graph in its
 * `SemanticProblemContext`. Two instances of one domain differ in object count,
 * object names, static facts and goals -- and none of that is a reason they
 * cannot be batched, because `BatchBuilder` offsets graph-local node indices
 * independently.
 *
 * Each family is checked the same way: encode instance A alone, encode instance
 * B alone, then encode both through *one* engine and require the mixed batch to
 * hold exactly the two graphs, node for node. If the engine were still bound to
 * a problem, the mixed call would either throw or silently encode B's rows
 * against A's object table.
 */
#include <gtest/gtest.h>

#include <mimir/formalism/domain.hpp>
#include <span>
#include <string>
#include <vector>

#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_successor_hgraph_encoder.hpp"
#include "mifrost/core/encoders/homo/semantic_color_encoder.hpp"
#include "test_utils.hpp"

namespace {

using mifrost::canonical::detail::ViewPreparation;

/// Two instances of one domain, deliberately of different size.
struct InstancePair {
   const char* domain;
   const char* first;
   const char* second;
};

constexpr InstancePair kInstancePairs[] = {
   {"blocks", "small", "smedium"},
   {"gripper", "gripper_b-1", "gripper_b-5"},
   {"spanner", "small", "medium"},
   {"delivery", "instance_2x2_p-1_0", "instance_3x3_p-3_0"},
};

std::string pair_name(const ::testing::TestParamInfo< InstancePair >& info)
{
   return mifrost_test::sanitize_test_name(
      std::string(info.param.domain) + "_" + info.param.first + "_" + info.param.second
   );
}

class CrossProblemBatchTest: public ::testing::TestWithParam< InstancePair > {};

/// Both instances go through one engine; the mixed batch must equal the singles.
template < typename Engine, typename Prepare >
void expect_mixed_batch_matches_singles(const Engine& engine, Prepare&& prepare)
{
   const auto first = prepare(0);
   const auto second = prepare(1);

   const std::vector< const ViewPreparation* > first_only{&first};
   const std::vector< const ViewPreparation* > second_only{&second};
   const std::vector< const ViewPreparation* > mixed{&first, &second};

   const auto first_encoding = engine.encode_batch(std::span{first_only});
   const auto second_encoding = engine.encode_batch(std::span{second_only});
   // The call that used to throw.
   const auto mixed_encoding = engine.encode_batch(std::span{mixed});

   EXPECT_EQ(first_encoding.num_graphs, 1);
   EXPECT_EQ(second_encoding.num_graphs, 1);
   EXPECT_EQ(mixed_encoding.num_graphs, 2);

   for(const auto& [node_type, count] : mixed_encoding.node_counts) {
      const auto first_count = first_encoding.node_counts.contains(node_type)
                                  ? first_encoding.node_counts.at(node_type)
                                  : 0;
      const auto second_count = second_encoding.node_counts.contains(node_type)
                                   ? second_encoding.node_counts.at(node_type)
                                   : 0;
      EXPECT_EQ(count, first_count + second_count) << "node type " << node_type;
   }

   // Graph-local indices: the second instance's rows start where the first's
   // end, rather than being read against the first instance's object table.
   for(const auto& [node_type, offsets] : mixed_encoding.ptrs) {
      ASSERT_EQ(offsets.size(), 3u) << "node type " << node_type;
      EXPECT_EQ(offsets.front(), 0) << "node type " << node_type;
      const auto first_count = first_encoding.node_counts.contains(node_type)
                                  ? first_encoding.node_counts.at(node_type)
                                  : 0;
      EXPECT_EQ(offsets[1], first_count) << "node type " << node_type;
      EXPECT_EQ(offsets[2], mixed_encoding.node_counts.at(node_type)) << "node type " << node_type;
   }
}

struct Instances {
   mifrost_test::Context contexts[2];
   mifrost::pymimir::SemanticProblemAdapter adapters[2];

   explicit Instances(const InstancePair& param)
       : contexts{
            mifrost_test::make_context(param.domain, param.first),
            mifrost_test::make_context(param.domain, param.second),
         },
         adapters{
            mifrost::pymimir::SemanticProblemAdapter(*contexts[0].problem),
            mifrost::pymimir::SemanticProblemAdapter(*contexts[1].problem),
         }
   {
   }
};

TEST_P(CrossProblemBatchTest, InstancesOfOneDomainDifferInEverythingButSchema)
{
   const Instances instances(GetParam());
   const auto first = instances.adapters[0].get_problem_context();
   const auto second = instances.adapters[1].get_problem_context();

   // What the split claims: the problem data differs, the schema does not.
   EXPECT_NE(first->objects.size(), second->objects.size());
   EXPECT_EQ(first->predicates(), second->predicates());
   EXPECT_EQ(first->actions(), second->actions());
   EXPECT_TRUE(mifrost::semantic_schema_compatible(first->schema, second->schema));
}

TEST_P(CrossProblemBatchTest, FlatEncodesTwoInstancesInOneBatch)
{
   const Instances instances(GetParam());
   const mifrost::SemanticFlatRelationEncoderEngine engine(
      instances.adapters[0].get_schema_context()
   );

   expect_mixed_batch_matches_singles(engine, [&](size_t index) {
      const auto& adapter = instances.adapters[index];
      return engine.prepare(
         adapter.get_problem_context(),
         adapter.make_state_view(instances.contexts[index].root),
         adapter.make_action_views(std::span{instances.contexts[index].actions})
      );
   });
}

TEST_P(CrossProblemBatchTest, HGraphEncodesTwoInstancesInOneBatch)
{
   const Instances instances(GetParam());
   const mifrost::SemanticHGraphEncoderEngine engine(instances.adapters[0].get_schema_context());

   expect_mixed_batch_matches_singles(engine, [&](size_t index) {
      const auto& adapter = instances.adapters[index];
      return engine.prepare(
         adapter.get_problem_context(),
         adapter.make_state_view(instances.contexts[index].root),
         adapter.make_action_views(std::span{instances.contexts[index].actions})
      );
   });
}

TEST_P(CrossProblemBatchTest, ColorEncodesTwoInstancesInOneBatch)
{
   const Instances instances(GetParam());
   const mifrost::SemanticColorEncoderEngine engine(instances.adapters[0].get_schema_context());

   expect_mixed_batch_matches_singles(engine, [&](size_t index) {
      const auto& adapter = instances.adapters[index];
      return engine.prepare(
         adapter.get_problem_context(),
         adapter.make_state_view(instances.contexts[index].root),
         adapter.make_action_views(std::span< const mimir::formalism::GroundAction >{})
      );
   });
}

/**
 * The successor family's constraint is per transition, not per batch: a state
 * and its successor share an object table, but two transitions in one batch
 * need not come from the same instance.
 */
TEST_P(CrossProblemBatchTest, SuccessorEncodesTransitionsFromTwoInstancesInOneBatch)
{
   const Instances instances(GetParam());
   const mifrost::SemanticSuccessorHGraphEncoderEngine engine(
      instances.adapters[0].get_schema_context()
   );

   std::vector< ViewPreparation > currents;
   std::vector< ViewPreparation > successors;
   for(size_t index = 0; index < 2; ++index) {
      const auto& adapter = instances.adapters[index];
      const auto& ctx = instances.contexts[index];
      const auto [successor, action] = mifrost_test::find_successor(ctx);
      (void) action;
      currents.push_back(engine.prepare_current(
         adapter.get_problem_context(),
         adapter.make_state_view(ctx.root),
         adapter.make_action_views(std::span< const mimir::formalism::GroundAction >{})
      ));
      successors.push_back(
         engine.prepare_successor(adapter.get_problem_context(), adapter.make_state_view(successor))
      );
   }

   const std::vector< const ViewPreparation* > current_refs{&currents[0], &currents[1]};
   const std::vector< const ViewPreparation* > successor_refs{&successors[0], &successors[1]};
   const auto encoding = engine.encode_batch(std::span{current_refs}, std::span{successor_refs});

   EXPECT_EQ(encoding.num_graphs, 2);
}

/// The pair itself still has to agree: the two lanes share one object table.
TEST_P(CrossProblemBatchTest, SuccessorRejectsAPairSplitAcrossInstances)
{
   const Instances instances(GetParam());
   const mifrost::SemanticSuccessorHGraphEncoderEngine engine(
      instances.adapters[0].get_schema_context()
   );

   const auto current = engine.prepare_current(
      instances.adapters[0].get_problem_context(),
      instances.adapters[0].make_state_view(instances.contexts[0].root),
      instances.adapters[0].make_action_views(std::span< const mimir::formalism::GroundAction >{})
   );
   const auto successor = engine.prepare_successor(
      instances.adapters[1].get_problem_context(),
      instances.adapters[1].make_state_view(instances.contexts[1].root)
   );

   const std::vector< const ViewPreparation* > current_refs{&current};
   const std::vector< const ViewPreparation* > successor_refs{&successor};
   EXPECT_THROW(
      (void) engine.encode_batch(std::span{current_refs}, std::span{successor_refs}),
      std::invalid_argument
   );
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   CrossProblemBatchTest,
   ::testing::ValuesIn(kInstancePairs),
   pair_name
);

}  // namespace
