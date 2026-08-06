/**
 * @file stream_view_lifetime_test.cpp
 * @brief Streams must not retain anything borrowed past the append that used it.
 *
 * A direct-View encode borrows the caller's goals, actions and history for the
 * duration of one call. A stream keeps entries across calls, so if any of those
 * borrows leaked into stored state the stream would read freed memory at
 * `flush()` -- a bug that a same-scope test cannot see, because the source
 * objects are still alive when it looks.
 *
 * Each test here destroys (and overwrites) the caller-side inputs between the
 * append and the flush, then demands the same output a stream produced while
 * everything stayed alive. Under AddressSanitizer these are also the tests that
 * would report a use-after-scope.
 */
#include <gtest/gtest.h>

#include <memory>
#include <mimir/formalism/domain.hpp>
#include <optional>
#include <span>
#include <vector>

#include "encoding_parity.hpp"
#include "mifrost/backends/pymimir/encoders/flat/flat_relation_encoder.hpp"
#include "mifrost/backends/pymimir/encoders/hetero/hgraph_stream_encoder.hpp"
#include "test_utils.hpp"

namespace {

using mifrost_test::expect_encoding_equal;

class StreamViewLifetimeTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(StreamViewLifetimeTest, FlatStreamSurvivesDestroyedStepInputs)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   (void) action;

   mifrost::FlatRelationEncoderEngine::Config config;
   config.target_sources = {mifrost::TargetSource::goals, mifrost::TargetSource::actions};

   const auto reference = [&] {
      mifrost::FlatRelationEncoderEngine engine(ctx.problem->get_domain(), config);
      mifrost::FlatRelationStreamEncoder stream(engine);
      const auto goals = mifrost_test::make_goal_inputs(ctx.problem);
      const std::vector< mimir::formalism::GroundAction >
         actions = ctx.actions.empty()
                      ? std::vector< mimir::formalism::GroundAction >{}
                      : std::vector< mimir::formalism::GroundAction >{ctx.actions.front()};
      stream.append(ctx.root, goals, actions);
      stream.append(successor, goals, actions);
      return stream.flush();
   }();

   mifrost::FlatRelationEncoderEngine engine(ctx.problem->get_domain(), config);
   mifrost::FlatRelationStreamEncoder stream(engine);
   for(const auto& state : {ctx.root, successor}) {
      // Every borrowed input dies before the next append and long before flush.
      auto goals = std::make_unique< mifrost::GoalInputs >(
         mifrost_test::make_goal_inputs(ctx.problem)
      );
      auto actions = std::make_unique< std::vector< mimir::formalism::GroundAction > >();
      if(not ctx.actions.empty()) {
         actions->push_back(ctx.actions.front());
      }
      stream.append(state, *goals, *actions);
      // Overwrite before releasing, so a retained pointer reads changed data
      // rather than merely-freed data and the comparison fails loudly.
      actions->clear();
      *goals = mifrost::GoalInputs{};
   }

   expect_encoding_equal(reference, stream.flush());
}

TEST_P(StreamViewLifetimeTest, HGraphMutableStreamSurvivesUpdateAndRemove)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   (void) action;

   mifrost::HGraphEncoderEngine::Config config;
   config.target_sources = {mifrost::TargetSource::goals};

   // Expected: one graph for the successor state, encoded plainly.
   const auto reference = [&] {
      mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain(), config);
      mifrost::HGraphMutableStreamEncoder stream(engine);
      const auto goals = mifrost_test::make_goal_inputs(ctx.problem);
      const std::vector< mimir::formalism::GroundAction > actions;
      stream.append(successor, goals, actions);
      return stream.flush();
   }();

   mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain(), config);
   mifrost::HGraphMutableStreamEncoder stream(engine);
   int64_t id = 0;
   {
      const auto goals = mifrost_test::make_goal_inputs(ctx.problem);
      const std::vector< mimir::formalism::GroundAction > actions;
      id = stream.append(ctx.root, goals, actions);
   }
   {
      // Replace the entry after its original inputs are gone.
      const auto goals = mifrost_test::make_goal_inputs(ctx.problem);
      const std::vector< mimir::formalism::GroundAction > actions;
      stream.update(id, successor, goals, actions);
   }
   {
      // And append/remove a third entry whose inputs also die immediately, so a
      // removed entry cannot leave a dangling borrow behind either.
      const auto goals = mifrost_test::make_goal_inputs(ctx.problem);
      const std::vector< mimir::formalism::GroundAction > actions;
      const auto removed = stream.append(ctx.root, goals, actions);
      stream.remove(removed);
   }

   expect_encoding_equal(reference, stream.flush());
}

// The engine holds no borrowed state between calls either: encoding the same
// state twice, with the caller's inputs recreated in between, must be stable.
TEST_P(StreamViewLifetimeTest, EngineStoresNoBorrowedStateBetweenEncodes)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);

   mifrost::FlatRelationEncoderEngine::Config config;
   config.target_sources = {mifrost::TargetSource::goals};
   mifrost::FlatRelationEncoderEngine engine(ctx.problem->get_domain(), config);

   const auto encode_once = [&] {
      const auto goals = mifrost_test::make_goal_inputs(ctx.problem);
      mifrost::BatchBuilder builder;
      builder.set_graph_kind("flat");
      engine.encode(ctx.root, goals, builder);
      builder.next_graph();
      return builder.build();
   };
   const auto first = encode_once();
   const auto second = encode_once();

   expect_encoding_equal(first, second);
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   StreamViewLifetimeTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

}  // namespace
