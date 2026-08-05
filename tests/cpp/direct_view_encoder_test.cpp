#include <gtest/gtest.h>

#include <mimir/formalism/domain.hpp>
#include <span>
#include <type_traits>
#include <variant>

#include "mifrost/backends/pymimir/encoders/hetero/hgraph_stream_encoder.hpp"
#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_successor_hgraph_encoder.hpp"
#include "mifrost/core/encoders/homo/semantic_color_encoder.hpp"
#include "mifrost/core/semantic/views.hpp"
#include "test_utils.hpp"

namespace {

using mifrost::BatchBuilder;

template < typename Variant >
void expect_numeric_variant_equal(const Variant& expected, const Variant& actual)
{
   ASSERT_EQ(expected.index(), actual.index());
   std::visit(
      [&]< typename T >(const T& expected_values) {
         ASSERT_TRUE(std::holds_alternative< T >(actual));
         EXPECT_EQ(expected_values, std::get< T >(actual));
      },
      expected
   );
}

void expect_encoding_equal(
   const BatchBuilder::BatchEncoding& expected,
   const BatchBuilder::BatchEncoding& actual
)
{
   EXPECT_EQ(expected.graph_kind, actual.graph_kind);
   EXPECT_EQ(expected.schema_flags, actual.schema_flags);
   EXPECT_EQ(expected.node_names, actual.node_names);
   EXPECT_EQ(expected.object_names, actual.object_names);
   EXPECT_EQ(expected.node_feature_dims, actual.node_feature_dims);
   EXPECT_EQ(expected.graph_attrs, actual.graph_attrs);
   EXPECT_EQ(expected.lazy_target_name_strings, actual.lazy_target_name_strings);
   EXPECT_EQ(expected.ptrs, actual.ptrs);
   EXPECT_EQ(expected.num_graphs, actual.num_graphs);
   EXPECT_EQ(expected.node_counts, actual.node_counts);

   ASSERT_EQ(expected.columns.size(), actual.columns.size());
   for(const auto& [key, expected_column] : expected.columns) {
      const auto actual_it = actual.columns.find(key);
      ASSERT_NE(actual_it, actual.columns.end()) << key;
      EXPECT_EQ(expected_column.dim, actual_it->second.dim) << key;
      expect_numeric_variant_equal(expected_column.data, actual_it->second.data);
   }

   ASSERT_EQ(expected.graph_fields.size(), actual.graph_fields.size());
   for(const auto& [key, expected_field] : expected.graph_fields) {
      const auto actual_it = actual.graph_fields.find(key);
      ASSERT_NE(actual_it, actual.graph_fields.end()) << key;
      EXPECT_EQ(expected_field.spec, actual_it->second.spec) << key;
      EXPECT_EQ(expected_field.ptr, actual_it->second.ptr) << key;
      expect_numeric_variant_equal(expected_field.values, actual_it->second.values);
      ASSERT_EQ(expected_field.pending.has_value(), actual_it->second.pending.has_value()) << key;
      if(expected_field.pending.has_value()) {
         expect_numeric_variant_equal(*expected_field.pending, *actual_it->second.pending);
      }
   }
}

class DirectViewEncoderTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(DirectViewEncoderTest, FlatDirectViewMatchesSemanticCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto semantic_context = adapter.get_task_context();
   const auto semantic_input = adapter.make_input(ctx.root);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::SemanticFlatRelationEncoderEngine engine(semantic_context);

   expect_encoding_equal(engine.encode(semantic_input), engine.encode(state_view, action_views));
}

TEST_P(DirectViewEncoderTest, FlatExplicitGoalsAndActionsMatchSemanticCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   const std::span< const mimir::formalism::GroundAction > actions{ctx.actions};
   auto semantic_input = adapter.make_input(ctx.root, goals);
   const auto& view_context = adapter.get_view_context();
   for(const auto& action : actions) {
      using NativeAction = std::remove_cvref_t< decltype(action) >;
      semantic_input.actions.push_back(
         mifrost::canonical::materialize_semantic_action(
            mifrost::pymimir::views::GroundActionView< NativeAction >{action, view_context}
         )
      );
   }
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(actions);
   const mifrost::semantic::LiteralsView goal_views(
      std::span{semantic_input.goals.data(), semantic_input.goals.size()}
   );
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context());

   expect_encoding_equal(
      engine.encode(semantic_input), engine.encode(state_view, goal_views, action_views)
   );
}

TEST_P(DirectViewEncoderTest, ColorDirectViewMatchesSemanticCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto semantic_context = adapter.get_task_context();
   const auto semantic_input = adapter.make_input(ctx.root);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::SemanticColorEncoderEngine engine(semantic_context);

   expect_encoding_equal(engine.encode(semantic_input), engine.encode(state_view, action_views));
}

TEST_P(DirectViewEncoderTest, HGraphDirectViewMatchesSemanticCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto semantic_context = adapter.get_task_context();
   const auto semantic_input = adapter.make_input(ctx.root);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::SemanticHGraphEncoderEngine engine(semantic_context);

   expect_encoding_equal(engine.encode(semantic_input), engine.encode(state_view, action_views));
}

TEST_P(DirectViewEncoderTest, SuccessorDirectViewsMatchSemanticCompatibilityInputs)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto semantic_context = adapter.get_task_context();
   const auto current_input = adapter.make_input(ctx.root);
   const auto successor_input = adapter.make_input(successor);
   const auto current_view = adapter.make_state_view(ctx.root);
   const auto successor_view = adapter.make_state_view(successor);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::SemanticSuccessorHGraphEncoderEngine engine(semantic_context);

   (void) action;
   expect_encoding_equal(
      engine.encode(current_input, successor_input),
      engine.encode(current_view, empty_actions, successor_view, empty_actions)
   );
}

TEST_P(DirectViewEncoderTest, FlatViewBatchMatchesSemanticCompatibilityBatch)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto semantic_context = adapter.get_task_context();
   const std::vector< mifrost::SemanticFlatRelationInput > semantic_inputs{
      adapter.make_input(ctx.root),
      adapter.make_input(successor),
   };
   const mifrost::SemanticFlatRelationEncoderEngine engine(semantic_context);
   const auto root_view = adapter.make_state_view(ctx.root);
   const auto successor_view = adapter.make_state_view(successor);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   BatchBuilder direct_builder;
   engine.encode(root_view, empty_actions, direct_builder);
   direct_builder.next_graph();
   engine.encode(successor_view, empty_actions, direct_builder);
   direct_builder.next_graph();

   (void) action;
   expect_encoding_equal(engine.encode_batch(semantic_inputs), direct_builder.build());
}

TEST_P(DirectViewEncoderTest, FlatViewSubgoalAndHistoryLanesMatchCompatibilityInput)
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
   semantic_input.subgoal_layers = {semantic_input.goals};
   semantic_input.history = {
      mifrost::SemanticHistoryEntry{.dt = -1, .literals = semantic_input.goals}
   };
   semantic_input.history_max_steps = 1;

   mifrost::FlatRelationEncoderConfig config;
   config.max_goal_level = 1;
   config.target_sources = {
      mifrost::TargetSource::goals,
      mifrost::TargetSource::subgoals,
      mifrost::TargetSource::history,
   };
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context(), config);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::semantic::LiteralsView goals(
      std::span{semantic_input.goals.data(), semantic_input.goals.size()}
   );
   const mifrost::semantic::SubgoalLayersView subgoal_layers(
      std::span{
         semantic_input.subgoal_layers.data(),
         semantic_input.subgoal_layers.size(),
      }
   );
   const mifrost::semantic::HistoryView history(
      std::span{semantic_input.history.data(), semantic_input.history.size()}
   );

   expect_encoding_equal(
      engine.encode(semantic_input),
      engine.encode(
         state_view, goals, subgoal_layers, empty_actions, history, semantic_input.history_max_steps
      )
   );
}

TEST_P(DirectViewEncoderTest, HGraphStreamMatchesDirectViewBatch)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   mifrost::HGraphEncoderEngine engine(ctx.problem->get_domain());
   mifrost::BatchBuilder expected_builder;
   expected_builder.set_graph_kind("hetero");
   engine.encode(ctx.root, expected_builder);
   expected_builder.next_graph();
   engine.encode(successor, expected_builder);
   expected_builder.next_graph();
   const auto expected = expected_builder.build();

   mifrost::HGraphStreamEncoder stream(engine);
   stream.append(ctx.root);
   stream.append(successor);

   (void) action;
   expect_encoding_equal(expected, stream.flush());
}

TEST_P(DirectViewEncoderTest, DirectViewsRemainValidForTheEncodeCall)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   auto task_input_context = adapter.get_task_context();
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::SemanticFlatRelationEncoderEngine engine(task_input_context);

   task_input_context.reset();
   EXPECT_NO_THROW((void) engine.encode(state_view, action_views));
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   DirectViewEncoderTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

}  // namespace
