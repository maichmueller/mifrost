#include <gtest/gtest.h>

#include <limits>
#include <mimir/formalism/domain.hpp>
#include <span>
#include <stdexcept>
#include <string_view>
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
   const auto goal_views = adapter.make_goal_views(goals);
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context());

   expect_encoding_equal(
      engine.encode(semantic_input),
      engine.encode(state_view, goal_views.goals_view(), action_views)
   );
}

TEST_P(DirectViewEncoderTest, FlatNativeGoalAndHistoryViewsMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   auto semantic_input = adapter.make_input(ctx.root, goals);
   if(semantic_input.goals.empty()) {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   semantic_input.history = {
      mifrost::SemanticHistoryEntry{.dt = -1, .literals = semantic_input.goals}
   };
   semantic_input.history_max_steps = 1;

   std::vector< mifrost::LiteralVariant > history_literals;
   history_literals.reserve(
      goals.static_goals.size() + goals.fluent_goals.size() + goals.derived_goals.size()
   );
   for(const auto literal : goals.static_goals) {
      history_literals.emplace_back(literal);
   }
   for(const auto literal : goals.fluent_goals) {
      history_literals.emplace_back(literal);
   }
   for(const auto literal : goals.derived_goals) {
      history_literals.emplace_back(literal);
   }
   const std::vector< std::pair< int, std::vector< mifrost::LiteralVariant > > > history{
      {-1, std::move(history_literals)}
   };

   const auto state_view = adapter.make_state_view(ctx.root);
   const auto goal_views = adapter.make_goal_views(goals);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto history_view = mifrost::pymimir::make_history_view(
      history, adapter.get_view_context()
   );
   mifrost::FlatRelationEncoderConfig config;
   config.target_sources = {mifrost::TargetSource::goals, mifrost::TargetSource::history};
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context(), config);

   expect_encoding_equal(
      engine.encode(semantic_input),
      engine.encode(
         state_view,
         goal_views.goals_view(),
         goal_views.subgoal_layers_view(),
         action_views,
         history_view,
         semantic_input.history_max_steps
      )
   );
}

TEST_P(DirectViewEncoderTest, FlatSparseRepeatedNativeGoalLevelsMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::GoalInputs sparse_goals;
   if(not source_goals.static_goals.empty()) {
      const auto literal = source_goals.static_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else if(not source_goals.fluent_goals.empty()) {
      const auto literal = source_goals.fluent_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else if(not source_goals.derived_goals.empty()) {
      const auto literal = source_goals.derived_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const auto semantic_input = adapter.make_input(ctx.root, sparse_goals);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto goal_views = adapter.make_goal_views(sparse_goals);
   const std::vector< mifrost::SemanticHistoryEntry > empty_history;
   const mifrost::semantic::HistoryView history_view(std::span{empty_history});
   mifrost::FlatRelationEncoderConfig config;
   config.max_goal_level = 3;
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context(), config);

   expect_encoding_equal(
      engine.encode(semantic_input),
      engine.encode(
         state_view,
         goal_views.goals_view(),
         goal_views.subgoal_layers_view(),
         action_views,
         history_view
      )
   );
}

TEST_P(DirectViewEncoderTest, NativeGoalLayersKeepSparseOccupiedLevels)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::GoalInputs sparse_goals;
   if(source_goals.static_goals.size() >= 2) {
      sparse_goals.append(source_goals.static_goals.front(), 1);
      sparse_goals.append(source_goals.static_goals.back(), 1'000'000);
   } else if(source_goals.fluent_goals.size() >= 2) {
      sparse_goals.append(source_goals.fluent_goals.front(), 1);
      sparse_goals.append(source_goals.fluent_goals.back(), 1'000'000);
   } else if(source_goals.derived_goals.size() >= 2) {
      sparse_goals.append(source_goals.derived_goals.front(), 1);
      sparse_goals.append(source_goals.derived_goals.back(), 1'000'000);
   } else {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const auto goal_views = adapter.make_goal_views(sparse_goals);
   const auto layers = goal_views.subgoal_layers_view();
   ASSERT_EQ(layers.size(), 2U);
   auto iterator = layers.begin();
   EXPECT_EQ((*iterator).level(), 1U);
   ++iterator;
   EXPECT_EQ((*iterator).level(), 1'000'000U);
   EXPECT_EQ(++iterator, layers.end());
   EXPECT_THROW((void) adapter.make_input(ctx.root, sparse_goals), std::invalid_argument);
}

// The occupied-level list is owned by NativeGoalLayersView, so a layer view
// taken from a temporary NativeGoalViews stays valid. Only the literal spans
// and the view context are borrowed, and both outlive this expression.
TEST_P(DirectViewEncoderTest, NativeGoalLayersFromTemporaryGoalViewsStayValid)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::GoalInputs sparse_goals;
   if(not source_goals.static_goals.empty()) {
      sparse_goals.append(source_goals.static_goals.front(), 2);
   } else if(not source_goals.fluent_goals.empty()) {
      sparse_goals.append(source_goals.fluent_goals.front(), 2);
   } else if(not source_goals.derived_goals.empty()) {
      sparse_goals.append(source_goals.derived_goals.front(), 2);
   } else {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   // The NativeGoalViews temporary dies at the end of the full expression; the
   // returned layer view must not reference it.
   const auto layers = adapter.make_goal_views(sparse_goals).subgoal_layers_view();

   ASSERT_EQ(layers.size(), 1U);
   ASSERT_EQ(layers.occupied_levels().size(), 1U);
   EXPECT_EQ(layers.occupied_levels().front(), 2U);
   size_t visited = 0;
   for(const auto layer : layers) {
      EXPECT_EQ(layer.level(), 2U);
      EXPECT_EQ(layer.size(), 1U);
      ++visited;
   }
   EXPECT_EQ(visited, 1U);
}

// The dense-layer bound belongs to the compatibility conversion, not to any one
// encoder family. It must be caller-configurable and must reject before the
// dense resize.
TEST_P(DirectViewEncoderTest, CompatibilityGoalLevelBoundIsCallerConfigurable)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::GoalInputs level_five_goals;
   if(not source_goals.static_goals.empty()) {
      level_five_goals.append(source_goals.static_goals.front(), 5);
   } else if(not source_goals.fluent_goals.empty()) {
      level_five_goals.append(source_goals.fluent_goals.front(), 5);
   } else if(not source_goals.derived_goals.empty()) {
      level_five_goals.append(source_goals.derived_goals.front(), 5);
   } else {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   // A caller whose own configuration allows level 5 gets a level-5 conversion:
   // the adapter no longer hardcodes any family's three-level limit.
   const auto permitted = adapter.make_input(ctx.root, level_five_goals, 5);
   ASSERT_EQ(permitted.subgoal_layers.size(), 5U);
   EXPECT_EQ(permitted.subgoal_layers.at(4).size(), 1U);
   for(size_t level = 0; level < 4; ++level) {
      EXPECT_TRUE(permitted.subgoal_layers.at(level).empty());
   }

   // A caller whose configuration only reaches level 3 is rejected, and the
   // message names the responsible boundary rather than an encoder.
   EXPECT_THROW(
      {
         try {
            (void) adapter.make_input(ctx.root, level_five_goals, 3);
         } catch(const std::invalid_argument& error) {
            EXPECT_NE(
               std::string_view{error.what()}.find("SemanticProblemAdapter"), std::string_view::npos
            );
            throw;
         }
      },
      std::invalid_argument
   );
}

// A sparse level far above the transport bound must be rejected outright rather
// than resized into a huge vector of empty layers.
TEST_P(DirectViewEncoderTest, CompatibilityConversionRejectsHugeSparseGoalLevel)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::GoalInputs huge_goals;
   constexpr size_t huge_level = mifrost::pymimir::kDenseGoalLayerTransportLimit + 1;
   if(not source_goals.static_goals.empty()) {
      huge_goals.append(source_goals.static_goals.front(), huge_level);
   } else if(not source_goals.fluent_goals.empty()) {
      huge_goals.append(source_goals.fluent_goals.front(), huge_level);
   } else if(not source_goals.derived_goals.empty()) {
      huge_goals.append(source_goals.derived_goals.front(), huge_level);
   } else {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   // Even when the caller asks for an unlimited bound, the transport limit wins.
   EXPECT_THROW(
      (void) adapter.make_input(ctx.root, huge_goals, std::numeric_limits< size_t >::max()),
      std::invalid_argument
   );

   // The native layer view still represents the same level sparsely.
   const auto layers = adapter.make_goal_views(huge_goals).subgoal_layers_view();
   ASSERT_EQ(layers.occupied_levels().size(), 1U);
   EXPECT_EQ(layers.occupied_levels().front(), huge_level);
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

TEST_P(DirectViewEncoderTest, ColorDirectGoalAndSubgoalViewsMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   auto semantic_input = adapter.make_input(ctx.root, goals);
   if(semantic_input.goals.empty()) {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }
   semantic_input.subgoal_layers = {semantic_input.goals};

   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::semantic::LiteralsView goal_views(std::span{semantic_input.goals});
   const mifrost::semantic::SubgoalLayersView subgoal_layers(
      std::span{semantic_input.subgoal_layers}
   );
   const mifrost::SemanticColorEncoderEngine engine(adapter.get_task_context());

   expect_encoding_equal(
      engine.encode(semantic_input),
      engine.encode(state_view, goal_views, subgoal_layers, action_views)
   );
}

TEST_P(DirectViewEncoderTest, ColorSparseRepeatedNativeGoalLevelsMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::GoalInputs sparse_goals;
   if(not source_goals.static_goals.empty()) {
      const auto literal = source_goals.static_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else if(not source_goals.fluent_goals.empty()) {
      const auto literal = source_goals.fluent_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else if(not source_goals.derived_goals.empty()) {
      const auto literal = source_goals.derived_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const auto semantic_input = adapter.make_input(ctx.root, sparse_goals);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto goal_views = adapter.make_goal_views(sparse_goals);
   const mifrost::SemanticColorEncoderEngine engine(adapter.get_task_context());

   expect_encoding_equal(
      engine.encode(semantic_input),
      engine.encode(
         state_view, goal_views.goals_view(), goal_views.subgoal_layers_view(), action_views
      )
   );
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

TEST_P(DirectViewEncoderTest, HGraphSparseRepeatedNativeGoalLevelsMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);
   mifrost::GoalInputs sparse_goals;
   if(not source_goals.static_goals.empty()) {
      const auto literal = source_goals.static_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else if(not source_goals.fluent_goals.empty()) {
      const auto literal = source_goals.fluent_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else if(not source_goals.derived_goals.empty()) {
      const auto literal = source_goals.derived_goals.front();
      sparse_goals.append(literal, 1);
      sparse_goals.append(literal, 3);
   } else {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const auto semantic_input = adapter.make_input(ctx.root, sparse_goals);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto goal_views = adapter.make_goal_views(sparse_goals);
   const std::vector< mifrost::SemanticHistoryEntry > empty_history;
   const mifrost::semantic::HistoryView history_view(std::span{empty_history});
   mifrost::SemanticHGraphEncoderConfig config;
   config.max_goal_level = 3;
   const mifrost::SemanticHGraphEncoderEngine engine(adapter.get_task_context(), config);

   expect_encoding_equal(
      engine.encode(semantic_input),
      engine.encode(
         state_view,
         goal_views.goals_view(),
         goal_views.subgoal_layers_view(),
         action_views,
         history_view
      )
   );
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

// Action graph nodes are deduplicated while every occurrence still contributes
// a target row. The direct path reaches that through the intern pool plus an
// occurrence-index lane; the compatibility path has no pool at all. Both must
// produce byte-identical output, including for repeated actions.
TEST_P(DirectViewEncoderTest, DuplicateActionOccurrencesMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   if(ctx.actions.empty()) {
      GTEST_SKIP() << "Fixture does not provide applicable actions.";
   }
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto& view_context = adapter.get_view_context();

   // Repeat the first action, keep a distinct one, then repeat the first again,
   // so first-use ordering and multiplicity are both observable.
   std::vector< mimir::formalism::GroundAction > repeated{
      ctx.actions.front(),
      ctx.actions.back(),
      ctx.actions.front(),
   };
   const std::span< const mimir::formalism::GroundAction > action_span{repeated};

   auto semantic_input = adapter.make_input(ctx.root);
   for(const auto& action : action_span) {
      using NativeAction = std::remove_cvref_t< decltype(action) >;
      semantic_input.actions.push_back(
         mifrost::canonical::materialize_semantic_action(
            mifrost::pymimir::views::GroundActionView< NativeAction >{action, view_context}
         )
      );
   }

   mifrost::FlatRelationEncoderConfig config;
   config.target_sources = {mifrost::TargetSource::actions};
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context(), config);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(action_span);

   expect_encoding_equal(engine.encode(semantic_input), engine.encode(state_view, action_views));
}

// The successor lane must not build goal/action/history records the successor
// algorithm never reads, and doing less work must not change the output.
TEST_P(DirectViewEncoderTest, SuccessorPreparationIsStateOnly)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto semantic_context = adapter.get_task_context();
   const auto successor_view = adapter.make_state_view(successor);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );

   const auto state_only = mifrost::canonical::detail::make_state_only_view_preparation(
      semantic_context, successor_view
   );
   const auto full = mifrost::canonical::detail::make_hgraph_view_preparation(
      semantic_context, successor_view, empty_actions
   );

   // Same state facts either way.
   EXPECT_EQ(state_only.state_facts, full.state_facts);
   EXPECT_EQ(state_only.fact_lookup.size(), full.fact_lookup.size());

   // No goal lane at all, and no atoms interned purely to back one.
   EXPECT_TRUE(state_only.goal_level_refs.empty());
   EXPECT_EQ(state_only.goal_layer_count, 0U);
   EXPECT_TRUE(state_only.action_pool.empty());
   EXPECT_TRUE(state_only.action_occurrence_indices.empty());
   EXPECT_TRUE(state_only.history_data.empty());
   EXPECT_LE(state_only.atom_pool.size(), full.atom_pool.size());
   if(not semantic_context->default_goals.empty()) {
      // The full preparation only differs by the discarded default goals.
      EXPECT_FALSE(full.goal_level_refs.empty());
   }

   (void) action;
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
