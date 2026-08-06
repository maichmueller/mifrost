/**
 * @file direct_view_flat_test.cpp
 * @brief Flat-family direct/compatibility parity for the remaining input shapes.
 *
 * `direct_view_encoder_test.cpp` covers the Flat default policy and the goal,
 * action, history and batch lanes. This file covers the shapes that reach
 * different code in `prepare_source`/`emit_relation_lane`: nullary predicates,
 * negative literals across all three predicate categories, batches whose graphs
 * do not agree on which lanes they populate, a legacy input with no task
 * context, and history truncation.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <mimir/formalism/domain.hpp>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "encoding_parity.hpp"
#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/semantic/views.hpp"
#include "test_utils.hpp"

namespace {

using mifrost_test::expect_encoding_equal;

mifrost::FlatRelationEncoderConfig rich_config()
{
   mifrost::FlatRelationEncoderConfig config;
   config.max_goal_level = 3;
   config.target_sources = {
      mifrost::TargetSource::goals,
      mifrost::TargetSource::subgoals,
      mifrost::TargetSource::actions,
      mifrost::TargetSource::history,
   };
   return config;
}

class DirectViewFlatTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

// A zero-arity relation has no argument slots, so the emitter takes a different
// branch for its tuples and its node table. Both toggles must agree across
// paths, because only one of them keeps those relations at all.
TEST_P(DirectViewFlatTest, NullaryRelationTogglesMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   auto semantic_input = adapter.make_input(ctx.root);
   const auto& view_context = adapter.get_view_context();
   for(const auto& entry : ctx.actions) {
      using NativeAction = std::remove_cvref_t< decltype(entry) >;
      semantic_input.actions.push_back(
         mifrost::canonical::materialize_semantic_action(
            mifrost::pymimir::views::GroundActionView< NativeAction >{entry, view_context}
         )
      );
   }
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(std::span{ctx.actions});

   for(const bool ignore_zero_arity : {true, false}) {
      for(const bool predicate_virtual_nodes : {false, true}) {
         auto config = rich_config();
         config.ignore_zero_arity_relations = ignore_zero_arity;
         config.use_predicate_virtual_nodes = predicate_virtual_nodes;
         const std::string label = std::string("ignore_zero_arity=")
                                   + (ignore_zero_arity ? "1" : "0")
                                   + " virtual_nodes=" + (predicate_virtual_nodes ? "1" : "0");
         const mifrost::SemanticFlatRelationEncoderEngine engine(
            adapter.get_task_context(), config
         );
         expect_encoding_equal(
            engine.encode(semantic_input), engine.encode(state_view, action_views), label
         );
      }
   }
}

// The compatibility one-shot encode finalizes through the compiled plan. The
// direct one builds its own BatchBuilder, so it has to finalize too -- a
// relation-major layout that reaches only one path is a silent divergence
// (`relation_args_layout` differs while every tensor still matches).
TEST_P(DirectViewFlatTest, RelationMajorPackingMatchesCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto semantic_input = adapter.make_input(ctx.root);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );

   for(const bool relation_major : {false, true}) {
      auto config = rich_config();
      config.pack_relation_args_relation_major = relation_major;
      const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context(), config);
      expect_encoding_equal(
         engine.encode(semantic_input),
         engine.encode(state_view, empty_actions),
         std::string("relation_major=") + (relation_major ? "1" : "0")
      );
   }
}

// Polarity selects a different relation suffix per predicate category. The
// fixtures only supply the polarity their PDDL has, so the negative branch is
// reached by flipping owned literals and reading them back through a semantic
// View -- the same shape a caller with precomputed literals uses.
TEST_P(DirectViewFlatTest, NegativeLiteralsAcrossCategoriesMatchCompatibilityInput)
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

   for(const bool support_literals : {false, true}) {
      for(const bool include_static : {true, false}) {
         auto config = rich_config();
         config.support_literals = support_literals;
         config.include_static = include_static;
         const std::string label = std::string("support_literals=") + (support_literals ? "1" : "0")
                                   + " include_static=" + (include_static ? "1" : "0");
         const mifrost::SemanticFlatRelationEncoderEngine engine(
            adapter.get_task_context(), config
         );
         expect_encoding_equal(
            engine.encode(semantic_input), engine.encode(state_view, goals, empty_actions), label
         );
      }
   }
}

// A batch whose graphs disagree about which lanes are populated is the case
// where a stale per-graph preparation would survive into the next graph. The
// lanes here differ deliberately: state only, then goals, then goals plus
// actions, then state only again.
TEST_P(DirectViewFlatTest, MixedBatchLanesMatchCompatibilityBatch)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   (void) action;
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   const auto& view_context = adapter.get_view_context();

   auto state_only = adapter.make_input(ctx.root);
   auto with_goals = adapter.make_input(successor, goals);
   auto with_goals_and_actions = adapter.make_input(ctx.root, goals);
   std::vector< mimir::formalism::GroundAction > action_list;
   if(not ctx.actions.empty()) {
      action_list = {ctx.actions.front(), ctx.actions.front()};
      for(const auto& entry : action_list) {
         using NativeAction = std::remove_cvref_t< decltype(entry) >;
         with_goals_and_actions.actions.push_back(
            mifrost::canonical::materialize_semantic_action(
               mifrost::pymimir::views::GroundActionView< NativeAction >{entry, view_context}
            )
         );
      }
   }
   auto trailing_state_only = adapter.make_input(successor);

   const std::vector< mifrost::SemanticFlatRelationInput > inputs{
      state_only, with_goals, with_goals_and_actions, trailing_state_only
   };

   const auto config = rich_config();
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context(), config);

   const auto root_view = adapter.make_state_view(ctx.root);
   const auto successor_view = adapter.make_state_view(successor);
   const auto goal_views = adapter.make_goal_views(goals);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto action_views = adapter.make_action_views(std::span{action_list});
   const std::vector< mifrost::SemanticHistoryEntry > empty_history;
   const mifrost::semantic::HistoryView history_view{std::span{empty_history}};

   mifrost::BatchBuilder direct;
   direct.set_graph_kind("flat");
   engine.encode(root_view, empty_actions, direct);
   direct.next_graph();
   engine.encode(
      successor_view,
      goal_views.goals_view(),
      goal_views.subgoal_layers_view(),
      empty_actions,
      history_view,
      std::nullopt,
      direct
   );
   direct.next_graph();
   engine.encode(
      root_view,
      goal_views.goals_view(),
      goal_views.subgoal_layers_view(),
      action_views,
      history_view,
      std::nullopt,
      direct
   );
   direct.next_graph();
   engine.encode(successor_view, empty_actions, direct);
   direct.next_graph();

   expect_encoding_equal(engine.encode_batch(inputs), direct.build());
}

// A legacy input owns its object table instead of sharing a task context. That
// changes which accessor every lane reads, and it is the one input shape with
// no direct-View counterpart, so it is checked against an engine built from the
// same specs rather than against a View encode.
TEST_P(DirectViewFlatTest, LegacyObjectTableWithoutTaskContextEncodes)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto task_context = adapter.get_task_context();
   const auto shared_input = adapter.make_input(ctx.root);

   // Same content, expressed without a task context: objects inline, and the
   // static facts folded into the state lane because a legacy input has no
   // separate static table.
   mifrost::SemanticFlatRelationInput legacy;
   legacy.objects = task_context->objects;
   legacy.state_facts = task_context->static_facts;
   legacy.state_facts.insert(
      legacy.state_facts.end(), shared_input.state_facts.begin(), shared_input.state_facts.end()
   );
   legacy.goals = task_context->default_goals;

   const auto config = rich_config();
   const mifrost::SemanticFlatRelationEncoderEngine legacy_engine(
      task_context->predicates, task_context->actions, config
   );
   const auto encoding = legacy_engine.encode(legacy);

   EXPECT_EQ(encoding.num_graphs, 1);
   EXPECT_EQ(encoding.object_names.size(), task_context->objects.size());
   // The legacy lane must still reach the same object identities.
   for(size_t index = 0; index < task_context->objects.size(); ++index) {
      EXPECT_EQ(encoding.object_names[index], task_context->objects[index]);
   }
}

// `history_max_steps` truncates the history lane. The direct path applies it to
// a borrowed history range while the compatibility path applies it to an owned
// vector; a limit smaller than the number of entries is what separates them.
TEST_P(DirectViewFlatTest, HistoryTruncationMatchesCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto goals = mifrost_test::make_goal_inputs(ctx.problem);
   const auto flat_input = adapter.make_input(ctx.root, goals);
   if(flat_input.goals.empty()) {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   std::vector< mifrost::LiteralVariant > literals;
   const auto append_all = [&](const auto& source) {
      for(const auto literal : source) {
         literals.emplace_back(literal);
      }
   };
   append_all(goals.static_goals);
   append_all(goals.fluent_goals);
   append_all(goals.derived_goals);

   // Three steps back, so a limit of one and two both truncate.
   const std::vector< std::pair< int, std::vector< mifrost::LiteralVariant > > > history{
      {-1, literals}, {-2, literals}, {-3, literals}
   };

   const auto state_view = adapter.make_state_view(ctx.root);
   const auto goal_views = adapter.make_goal_views(goals);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto history_view = mifrost::pymimir::make_history_view(
      history, adapter.get_view_context()
   );

   for(const int64_t max_steps : {1, 2, 3, 8}) {
      auto semantic_input = adapter.make_input(ctx.root, goals);
      for(const auto& [dt, entry_literals] : history) {
         (void) entry_literals;
         semantic_input.history.push_back(
            mifrost::SemanticHistoryEntry{.dt = dt, .literals = flat_input.goals}
         );
      }
      semantic_input.history_max_steps = max_steps;

      const auto config = rich_config();
      const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context(), config);
      expect_encoding_equal(
         engine.encode(semantic_input),
         engine.encode(
            state_view,
            goal_views.goals_view(),
            goal_views.subgoal_layers_view(),
            empty_actions,
            history_view,
            max_steps
         ),
         "history_max_steps=" + std::to_string(max_steps)
      );
   }
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   DirectViewFlatTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

}  // namespace
