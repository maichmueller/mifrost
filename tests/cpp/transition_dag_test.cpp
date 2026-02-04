#include "mifrost/core/transition_dag.hpp"

#include <gtest/gtest.h>

#include "test_utils.hpp"

using namespace mifrost;

class TransitionDAGTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(TransitionDAGTest, RegistersTransitionsAndDepths)
{
   const auto param = GetParam();
   auto ctx = mifrost_test::make_context(param.domain, param.problem);
   auto [succ_state, succ_action] = mifrost_test::find_successor(ctx);

   TransitionDAG dag(ctx.root);
   const auto [parent_idx, child_idx] = dag.register_transition(ctx.root, succ_state, succ_action);

   EXPECT_EQ(parent_idx, dag.root_index());
   EXPECT_EQ(dag.index(ctx.root), dag.root_index());
   EXPECT_EQ(dag.index(succ_state), child_idx);
   EXPECT_TRUE(dag.contains(ctx.root));
   EXPECT_TRUE(dag.contains(succ_state));
   EXPECT_EQ(dag.depth(parent_idx), 0);
   EXPECT_EQ(dag.depth(child_idx), 1);

   const auto action = dag.action(child_idx);
   ASSERT_TRUE(action.has_value());
   EXPECT_EQ(action.value()->get_index(), succ_action->get_index());

   const auto children = dag.children(parent_idx);
   EXPECT_EQ(children.size(), 1u);
   EXPECT_EQ(children.front(), child_idx);

   const auto transitions = dag.transitions();
   ASSERT_EQ(transitions.size(), 1u);
   EXPECT_EQ(transitions.front().first, parent_idx);
   EXPECT_EQ(transitions.front().second, child_idx);

   const auto successors = dag.successors();
   ASSERT_EQ(successors.size(), 1u);
   EXPECT_EQ(successors.front().index, child_idx);
   EXPECT_EQ(successors.front().depth, 1);
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   TransitionDAGTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);
