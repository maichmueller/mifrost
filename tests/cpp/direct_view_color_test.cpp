/**
 * @file direct_view_color_test.cpp
 * @brief Color-family direct/compatibility parity across the config matrix.
 *
 * Color is the smallest family and the first one to consume borrowed Views
 * through the canonical statically dispatched algorithm. These tests pin that
 * the direct path and the owning compatibility path agree for every
 * combination of the encoder's configuration flags, and that unsupported
 * subgoal levels are rejected by Color's own limit rather than silently
 * encoded.
 *
 * `direct_view_encoder_test.cpp` covers the cross-family parity cases; the
 * Color configuration matrix lives here to keep that file readable.
 */
#include <gtest/gtest.h>

#include <mimir/formalism/domain.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "encoding_parity.hpp"
#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/encoders/homo/semantic_color_encoder.hpp"
#include "mifrost/core/semantic/views.hpp"
#include "test_utils.hpp"

namespace {

using mifrost_test::expect_encoding_equal;

/** Every combination of the three Color configuration flags. */
std::vector< mifrost::SemanticColorEncoderConfig > all_configs()
{
   std::vector< mifrost::SemanticColorEncoderConfig > configs;
   for(const bool edge_features : {false, true}) {
      for(const bool predicate_nodes : {false, true}) {
         for(const bool export_names : {false, true}) {
            configs.push_back(
               mifrost::SemanticColorEncoderConfig{
                  .edge_features = edge_features,
                  .enable_global_predicate_nodes = predicate_nodes,
                  .export_node_names = export_names,
               }
            );
         }
      }
   }
   return configs;
}

std::string config_label(const mifrost::SemanticColorEncoderConfig& config)
{
   return std::string("edge_features=") + (config.edge_features ? "1" : "0")
          + " predicate_nodes=" + (config.enable_global_predicate_nodes ? "1" : "0")
          + " export_node_names=" + (config.export_node_names ? "1" : "0");
}

class DirectViewColorTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(DirectViewColorTest, StateOnlyMatchesAcrossTheConfigMatrix)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto semantic_input = adapter.make_input(ctx.root);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );

   for(const auto& config : all_configs()) {
      const mifrost::SemanticColorEncoderEngine engine(adapter.get_task_context(), config);
      expect_encoding_equal(
         engine.encode(semantic_input),
         engine.encode(state_view, empty_actions),
         config_label(config)
      );
   }
}

// Native Pymimir goals, not semantic Views over already-owned records: the
// direct path has to traverse NativeGoalLiteralsView and the sparse layer view.
TEST_P(DirectViewColorTest, NativePymimirGoalsAndLayersMatchAcrossTheConfigMatrix)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);

   // Spread the fixture's goals over the three supported subgoal levels, and
   // keep a level-0 goal so both the goal and subgoal lanes are exercised.
   mifrost::GoalInputs layered;
   size_t appended = 0;
   const auto append_all = [&](const auto& literals) {
      for(const auto literal : literals) {
         layered.append(literal, appended % 4);
         ++appended;
      }
   };
   append_all(source_goals.static_goals);
   append_all(source_goals.fluent_goals);
   append_all(source_goals.derived_goals);
   if(appended == 0) {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const auto semantic_input = adapter.make_input(ctx.root, layered, 3);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto goal_views = adapter.make_goal_views(layered);

   for(const auto& config : all_configs()) {
      const mifrost::SemanticColorEncoderEngine engine(adapter.get_task_context(), config);
      expect_encoding_equal(
         engine.encode(semantic_input),
         engine.encode(
            state_view, goal_views.goals_view(), goal_views.subgoal_layers_view(), empty_actions
         ),
         config_label(config)
      );
   }
}

// Negative goal literals reach a different node-name prefix and colour, so they
// need their own parity case.
TEST_P(DirectViewColorTest, NegativeGoalLiteralsMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   auto semantic_input = adapter.make_input(ctx.root);
   semantic_input.use_default_goals = false;
   semantic_input.goals = adapter.get_task_context()->default_goals;
   if(semantic_input.goals.empty()) {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }
   for(auto& goal : semantic_input.goals) {
      goal.positive = not goal.positive;
   }

   const auto state_view = adapter.make_state_view(ctx.root);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::semantic::LiteralsView goals{std::span{semantic_input.goals}};

   for(const auto& config : all_configs()) {
      const mifrost::SemanticColorEncoderEngine engine(adapter.get_task_context(), config);
      expect_encoding_equal(
         engine.encode(semantic_input),
         engine.encode(state_view, goals, empty_actions),
         config_label(config)
      );
   }
}

// Color's level limit is Color's own, and it must fire on both paths.
TEST_P(DirectViewColorTest, RejectsUnsupportedSubgoalLevelOnBothPaths)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::GoalInputs too_deep;
   if(not source_goals.static_goals.empty()) {
      too_deep.append(source_goals.static_goals.front(), 4);
   } else if(not source_goals.fluent_goals.empty()) {
      too_deep.append(source_goals.fluent_goals.front(), 4);
   } else if(not source_goals.derived_goals.empty()) {
      too_deep.append(source_goals.derived_goals.front(), 4);
   } else {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const mifrost::SemanticColorEncoderEngine engine(adapter.get_task_context());
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto goal_views = adapter.make_goal_views(too_deep);

   // Direct path: Color rejects level 4 using its own suffix table.
   EXPECT_THROW(
      (void) engine.encode(
         state_view, goal_views.goals_view(), goal_views.subgoal_layers_view(), empty_actions
      ),
      std::invalid_argument
   );

   // Compatibility path: the conversion is allowed to build four dense layers
   // (that bound belongs to the caller, not to Color), and Color still rejects.
   const auto semantic_input = adapter.make_input(ctx.root, too_deep, 4);
   ASSERT_EQ(semantic_input.subgoal_layers.size(), 4U);
   EXPECT_THROW((void) engine.encode(semantic_input), std::invalid_argument);
}

// Color has no action lane at all; both paths must say so.
TEST_P(DirectViewColorTest, RejectsActionsOnBothPaths)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   if(ctx.actions.empty()) {
      GTEST_SKIP() << "Fixture does not provide applicable actions.";
   }
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const mifrost::SemanticColorEncoderEngine engine(adapter.get_task_context());
   const auto state_view = adapter.make_state_view(ctx.root);
   const std::span< const mimir::formalism::GroundAction > actions{ctx.actions};
   const auto action_views = adapter.make_action_views(actions);

   EXPECT_THROW((void) engine.encode(state_view, action_views), std::invalid_argument);

   auto semantic_input = adapter.make_input(ctx.root);
   const auto& view_context = adapter.get_view_context();
   using NativeAction = std::remove_cvref_t< decltype(ctx.actions.front()) >;
   semantic_input.actions.push_back(
      mifrost::canonical::materialize_semantic_action(
         mifrost::pymimir::views::GroundActionView< NativeAction >{
            ctx.actions.front(), view_context
         }
      )
   );
   EXPECT_THROW((void) engine.encode(semantic_input), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   DirectViewColorTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

}  // namespace
