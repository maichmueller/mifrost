/**
 * @file direct_view_hgraph_test.cpp
 * @brief HGraph direct/compatibility parity across the encoder policy matrix.
 *
 * HGraph is the widest family: actions, history, static facts, nullary
 * predicates, LGAN anchors, target sources and goal derivations each open a
 * separate emission path. `direct_view_encoder_test.cpp` covers the default
 * policy; this file walks the policy matrix, because a lane that only the
 * compatibility input reaches would otherwise go unnoticed -- the default
 * configuration never emits it.
 *
 * Every case runs one rich input (layered goals, repeated actions, a non-empty
 * history) through both paths and demands identical output.
 */
#include <gtest/gtest.h>

#include <map>
#include <mimir/formalism/domain.hpp>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "encoding_parity.hpp"
#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_successor_hgraph_encoder.hpp"
#include "mifrost/core/semantic/views.hpp"
#include "test_utils.hpp"

namespace {

using mifrost_test::expect_encoding_equal;

/**
 * One input, expressed twice.
 *
 * The compatibility member is an owned `SemanticFlatRelationInput`; the View
 * members borrow from the fixture's native task and from this struct's own
 * vectors. Both describe the same encode, so the encoders must agree.
 */
struct Scenario {
   mifrost::GoalInputs goals;
   std::vector< mimir::formalism::GroundAction > actions;
   std::vector< std::pair< int, std::vector< mifrost::LiteralVariant > > > history;
   int history_max_steps = 0;
   mifrost::SemanticFlatRelationInput compatibility;
};

/**
 * Build the richest input the fixture supports.
 *
 * Goals are spread over four subgoal levels so sparse-level handling is live;
 * the action lane repeats its first entry so pool deduplication and occurrence
 * multiplicity both matter; the history lane replays the goal literals at two
 * distinct time offsets.
 */
Scenario make_scenario(
   const mifrost_test::Context& ctx,
   const mifrost::pymimir::SemanticProblemAdapter& adapter
)
{
   Scenario scenario;
   const auto source_goals = mifrost_test::make_goal_inputs(ctx.problem);

   size_t appended = 0;
   // Every literal, kept flat at level 0. The layered goal lane splits its
   // literals across `goals` and `subgoal_layers`, so it cannot supply the
   // owned records the history lane needs; this second conversion can.
   mifrost::GoalInputs flat_goals;
   std::vector< mifrost::LiteralVariant > history_literals;
   const auto append_all = [&](const auto& literals) {
      for(const auto literal : literals) {
         scenario.goals.append(literal, appended % 4);
         flat_goals.append(literal, 0);
         history_literals.emplace_back(literal);
         ++appended;
      }
   };
   append_all(source_goals.static_goals);
   append_all(source_goals.fluent_goals);
   append_all(source_goals.derived_goals);

   if(not ctx.actions.empty()) {
      scenario.actions = {ctx.actions.front(), ctx.actions.back(), ctx.actions.front()};
   }

   if(not history_literals.empty()) {
      scenario.history.emplace_back(-1, history_literals);
      scenario.history.emplace_back(-2, std::move(history_literals));
      scenario.history_max_steps = 2;
   }

   scenario.compatibility = adapter.make_input(ctx.root, scenario.goals, 3);
   const auto flat_input = adapter.make_input(ctx.root, flat_goals);
   const auto& view_context = adapter.get_view_context();
   for(const auto& action : scenario.actions) {
      using NativeAction = std::remove_cvref_t< decltype(action) >;
      scenario.compatibility.actions.push_back(
         mifrost::canonical::materialize_semantic_action(
            mifrost::pymimir::views::GroundActionView< NativeAction >{action, view_context}
         )
      );
   }
   for(const auto& [dt, literals] : scenario.history) {
      mifrost::SemanticHistoryEntry entry;
      entry.dt = dt;
      entry.literals = flat_input.goals;
      scenario.compatibility.history.push_back(std::move(entry));
   }
   scenario.compatibility.history_max_steps = scenario.history_max_steps;
   return scenario;
}

struct NamedConfig {
   std::string name;
   mifrost::SemanticHGraphEncoderConfig config;
};

/**
 * The policy matrix.
 *
 * Each entry starts from a baseline that already exercises every lane and then
 * moves one dimension, so a parity failure names the responsible policy instead
 * of a combination. The last few entries stack dimensions that interact:
 * nullary predicates change the object table that LGAN anchors index into, and
 * dropping static facts changes which goals count as satisfied.
 */
std::vector< NamedConfig > policy_matrix()
{
   mifrost::SemanticHGraphEncoderConfig baseline;
   baseline.max_goal_level = 3;
   baseline.ignore_actions = false;
   baseline.target_sources = {
      mifrost::TargetSource::goals,
      mifrost::TargetSource::subgoals,
      mifrost::TargetSource::actions,
      mifrost::TargetSource::history,
   };

   std::vector< NamedConfig > matrix;
   matrix.push_back({"baseline", baseline});

   auto ignore_actions = baseline;
   ignore_actions.ignore_actions = true;
   ignore_actions.target_sources = {mifrost::TargetSource::goals};
   matrix.push_back({"ignore_actions", ignore_actions});

   auto no_static = baseline;
   no_static.include_static = false;
   matrix.push_back({"include_static=false", no_static});

   auto no_empty_edges = baseline;
   no_empty_edges.include_empty_edge_types = false;
   matrix.push_back({"include_empty_edge_types=false", no_empty_edges});

   auto nullary = baseline;
   nullary.add_nullary_predicates = true;
   matrix.push_back({"add_nullary_predicates", nullary});

   auto literals = baseline;
   literals.support_literals = true;
   matrix.push_back({"support_literals", literals});

   auto no_names = baseline;
   no_names.export_node_names = false;
   matrix.push_back({"export_node_names=false", no_names});

   auto goal_targets_only = baseline;
   goal_targets_only.target_sources = {mifrost::TargetSource::goals};
   matrix.push_back({"target_sources=goals", goal_targets_only});

   auto history_targets_only = baseline;
   history_targets_only.target_sources = {mifrost::TargetSource::history};
   matrix.push_back({"target_sources=history", history_targets_only});

   auto lgan = baseline;
   lgan.include_lgan_edges = true;
   lgan.lgan_anchor_sources = {mifrost::TargetSource::goals, mifrost::TargetSource::actions};
   matrix.push_back({"lgan_anchors=goals+actions", lgan});

   auto derivations = baseline;
   derivations.goal_derivations = {
      mifrost::GoalDerivation::plain,
      mifrost::GoalDerivation::satisfied,
      mifrost::GoalDerivation::unsatisfied,
      mifrost::GoalDerivation::added_satisfied,
      mifrost::GoalDerivation::added_unsatisfied,
   };
   matrix.push_back({"goal_derivations=all", derivations});

   auto plain_only = baseline;
   plain_only.goal_derivations = {mifrost::GoalDerivation::plain};
   matrix.push_back({"goal_derivations=plain", plain_only});

   auto combined = baseline;
   combined.add_nullary_predicates = true;
   combined.include_static = false;
   combined.include_lgan_edges = true;
   combined.lgan_anchor_sources = {mifrost::TargetSource::subgoals};
   combined.support_literals = true;
   matrix.push_back({"nullary+no_static+lgan_subgoals+literals", combined});

   return matrix;
}

class DirectViewHGraphTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

TEST_P(DirectViewHGraphTest, PolicyMatrixMatchesCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto scenario = make_scenario(ctx, adapter);
   if(scenario.compatibility.goals.empty()) {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const auto state_view = adapter.make_state_view(ctx.root);
   const auto goal_views = adapter.make_goal_views(scenario.goals);
   const auto action_views = adapter.make_action_views(std::span{scenario.actions});
   const auto history_view = mifrost::pymimir::make_history_view(
      scenario.history, adapter.get_view_context()
   );

   for(const auto& [name, config] : policy_matrix()) {
      const mifrost::SemanticHGraphEncoderEngine engine(adapter.get_task_context(), config);
      expect_encoding_equal(
         engine.encode(scenario.compatibility),
         engine.encode(
            state_view,
            goal_views.goals_view(),
            goal_views.subgoal_layers_view(),
            action_views,
            history_view,
            scenario.history_max_steps
         ),
         name
      );
   }
}

// Polarity picks a different relation suffix and a different satisfaction
// answer, and the native fixture only supplies whatever polarity the PDDL has.
// Flipping owned literals and reading them back through a semantic View is the
// only way to reach the negative branch on both paths.
TEST_P(DirectViewHGraphTest, NegativeGoalLiteralsMatchCompatibilityInput)
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

   for(const auto& [name, config] : policy_matrix()) {
      const mifrost::SemanticHGraphEncoderEngine engine(adapter.get_task_context(), config);
      expect_encoding_equal(
         engine.encode(semantic_input), engine.encode(state_view, goals, empty_actions), name
      );
   }
}

/**
 * One input whose every goal literal appears at two different levels.
 *
 * The native `GoalInputs` carrier stores one level per literal, so it cannot
 * express this shape and `make_scenario` never reaches it -- but a caller that
 * supplies the goal and subgoal lanes separately can, and the public PyTyr and
 * Pymimir transition APIs do exactly that. This family resolves a repeated goal
 * to its highest level, so both occurrences must land on the same node.
 */
struct RepeatedLevelInput {
   mifrost::SemanticFlatRelationInput compatibility;
   std::vector< std::vector< mifrost::SemanticLiteral > > layers;
};

RepeatedLevelInput repeated_level_input(
   const mifrost_test::Context& ctx,
   const mifrost::pymimir::SemanticProblemAdapter& adapter
)
{
   RepeatedLevelInput result;
   result.compatibility = adapter.make_input(ctx.root);
   result.compatibility.use_default_goals = false;
   result.compatibility.goals = adapter.get_task_context()->default_goals;
   result.compatibility.subgoal_layers = {result.compatibility.goals};
   result.layers = result.compatibility.subgoal_layers;
   return result;
}

// A goal repeated across levels must resolve to one level on both paths.
TEST_P(DirectViewHGraphTest, RepeatedGoalLevelsMatchCompatibilityInput)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto input = repeated_level_input(ctx, adapter);
   if(input.compatibility.goals.empty()) {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const auto state_view = adapter.make_state_view(ctx.root);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const mifrost::semantic::LiteralsView goals_view{std::span{input.compatibility.goals}};
   const mifrost::semantic::SubgoalLayersView layers_view{std::span{input.layers}};
   const std::vector< mifrost::SemanticHistoryEntry > empty_history;
   const mifrost::semantic::HistoryView history_view{std::span{empty_history}};

   for(const auto& [name, config] : policy_matrix()) {
      const mifrost::SemanticHGraphEncoderEngine engine(adapter.get_task_context(), config);
      expect_encoding_equal(
         engine.encode(input.compatibility),
         engine.encode(
            state_view, goals_view, layers_view, empty_actions, history_view, std::nullopt
         ),
         name
      );
   }
}

// `update_relations` replaces the relation arity table after construction. The
// direct path caches graph-derived preparation state; the compatibility path
// does not. A stale cache would only show up after the table changed.
TEST_P(DirectViewHGraphTest, RelationUpdatePreservesParity)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto scenario = make_scenario(ctx, adapter);
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto goal_views = adapter.make_goal_views(scenario.goals);
   const auto action_views = adapter.make_action_views(std::span{scenario.actions});
   const auto history_view = mifrost::pymimir::make_history_view(
      scenario.history, adapter.get_view_context()
   );

   mifrost::SemanticHGraphEncoderConfig config;
   config.max_goal_level = 3;
   config.ignore_actions = false;
   mifrost::SemanticHGraphEncoderEngine engine(adapter.get_task_context(), config);

   const auto encode_both = [&] {
      expect_encoding_equal(
         engine.encode(scenario.compatibility),
         engine.encode(
            state_view,
            goal_views.goals_view(),
            goal_views.subgoal_layers_view(),
            action_views,
            history_view,
            scenario.history_max_steps
         )
      );
   };
   encode_both();

   auto relations = engine.get_relation_arities();
   relations["mifrost_test_extra_relation"] = 2;
   engine.update_relations(relations);
   encode_both();
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   DirectViewHGraphTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

class DirectViewSuccessorTest: public ::testing::TestWithParam< mifrost_test::DomainCase > {};

// Full and delta successor modes read the two state lanes differently: full
// emits both, delta emits only the difference. The state-only preparation on
// the successor lane must not change either answer.
TEST_P(DirectViewSuccessorTest, SuccessorModesMatchCompatibilityInputs)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   (void) action;
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto scenario = make_scenario(ctx, adapter);
   const auto successor_input = adapter.make_input(successor);

   // The successor lane's public contract is (current state, successor state,
   // goals): no history, no successor actions. Reuse the scenario's layered
   // goals and repeated actions, but rebuild the current input without the
   // history the successor family never accepts.
   auto current_input = adapter.make_input(ctx.root, scenario.goals, 3);
   current_input.actions = scenario.compatibility.actions;

   const auto current_view = adapter.make_state_view(ctx.root);
   const auto successor_view = adapter.make_state_view(successor);
   const auto goal_views = adapter.make_goal_views(scenario.goals);
   const auto action_views = adapter.make_action_views(std::span{scenario.actions});
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );

   for(const auto mode :
       {mifrost::SemanticSuccessorMode::full, mifrost::SemanticSuccessorMode::delta}) {
      for(const bool goal_satisfaction : {false, true}) {
         mifrost::SemanticSuccessorHGraphEncoderConfig config;
         config.max_goal_level = 3;
         config.successor_mode = mode;
         config.include_successor_goal_satisfaction = goal_satisfaction;
         config.ignore_actions = false;
         config.target_sources = {mifrost::TargetSource::goals, mifrost::TargetSource::actions};
         const std::string label = std::string("mode=")
                                   + (mode == mifrost::SemanticSuccessorMode::full ? "full"
                                                                                   : "delta")
                                   + " goal_satisfaction=" + (goal_satisfaction ? "1" : "0");

         const mifrost::SemanticSuccessorHGraphEncoderEngine engine(
            adapter.get_task_context(), config
         );
         expect_encoding_equal(
            engine.encode(current_input, successor_input),
            engine.encode(
               current_view,
               goal_views.goals_view(),
               goal_views.subgoal_layers_view(),
               action_views,
               successor_view,
               empty_actions
            ),
            label
         );
      }
   }
}

// Successor goal satisfaction re-reads the prepared goals against the successor
// state facts, so it is the pass most exposed to a goal whose level was
// resolved differently on the two paths.
TEST_P(DirectViewSuccessorTest, RepeatedGoalLevelsMatchWithSuccessorSatisfaction)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   (void) action;
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto input = repeated_level_input(ctx, adapter);
   const auto successor_input = adapter.make_input(successor);
   if(input.compatibility.goals.empty()) {
      GTEST_SKIP() << "Fixture does not provide goal literals.";
   }

   const auto current_view = adapter.make_state_view(ctx.root);
   const auto successor_view = adapter.make_state_view(successor);
   const mifrost::semantic::LiteralsView goals_view{std::span{input.compatibility.goals}};
   const mifrost::semantic::SubgoalLayersView layers_view{std::span{input.layers}};
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );

   mifrost::SemanticSuccessorHGraphEncoderConfig config;
   config.max_goal_level = 3;
   config.support_literals = true;
   config.include_successor_goal_satisfaction = true;
   config.goal_derivations = {
      mifrost::GoalDerivation::plain,
      mifrost::GoalDerivation::satisfied,
      mifrost::GoalDerivation::unsatisfied,
   };
   const mifrost::SemanticSuccessorHGraphEncoderEngine engine(adapter.get_task_context(), config);

   expect_encoding_equal(
      engine.encode(input.compatibility, successor_input),
      engine.encode(
         current_view, goals_view, layers_view, empty_actions, successor_view, empty_actions
      )
   );
}

// The prepared-batch entry point holds both lanes of every transition before
// encoding any of them; a stream flushes exactly this way.
TEST_P(DirectViewSuccessorTest, PreparedBatchMatchesCompatibilityBatch)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   (void) action;
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto goals = mifrost_test::make_goal_inputs(ctx.problem);

   mifrost::SemanticSuccessorHGraphEncoderConfig config;
   config.max_goal_level = 3;
   config.support_literals = true;
   config.include_successor_goal_satisfaction = true;
   const mifrost::SemanticSuccessorHGraphEncoderEngine engine(adapter.get_task_context(), config);

   const auto current_view = adapter.make_state_view(ctx.root);
   const auto successor_view = adapter.make_state_view(successor);
   const auto goal_views = adapter.make_goal_views(goals);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );

   // Two transitions with different lanes: one takes the task context's default
   // goals, the other explicit layered goals.
   const std::vector< mifrost::SemanticFlatRelationInput > currents{
      adapter.make_input(ctx.root), adapter.make_input(ctx.root, goals, 3)
   };
   const std::vector< mifrost::SemanticFlatRelationInput > successors{
      adapter.make_input(successor), adapter.make_input(successor)
   };

   const std::vector< mifrost::canonical::detail::ViewPreparation > prepared_currents{
      engine.prepare_current(current_view, empty_actions),
      engine.prepare_current(
         current_view, goal_views.goals_view(), goal_views.subgoal_layers_view(), empty_actions
      ),
   };
   const std::vector< mifrost::canonical::detail::ViewPreparation > prepared_successors{
      engine.prepare_successor(successor_view),
      engine.prepare_successor(successor_view),
   };
   std::vector< const mifrost::canonical::detail::ViewPreparation* > current_refs;
   std::vector< const mifrost::canonical::detail::ViewPreparation* > successor_refs;
   for(size_t index = 0; index < prepared_currents.size(); ++index) {
      current_refs.push_back(&prepared_currents[index]);
      successor_refs.push_back(&prepared_successors[index]);
   }

   expect_encoding_equal(
      engine.encode_batch(currents, successors),
      engine.encode_batch(std::span{current_refs}, std::span{successor_refs})
   );
}

// The successor engine forwards the arity table to its inner HGraph engine.
TEST_P(DirectViewSuccessorTest, RelationUpdatePreservesParity)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   (void) action;
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto current_input = adapter.make_input(ctx.root);
   const auto successor_input = adapter.make_input(successor);
   const auto current_view = adapter.make_state_view(ctx.root);
   const auto successor_view = adapter.make_state_view(successor);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );

   mifrost::SemanticSuccessorHGraphEncoderEngine engine(adapter.get_task_context());
   const auto encode_both = [&] {
      expect_encoding_equal(
         engine.encode(current_input, successor_input),
         engine.encode(current_view, empty_actions, successor_view, empty_actions)
      );
   };
   encode_both();

   auto relations = engine.get_relation_arities();
   relations["mifrost_test_extra_relation"] = 1;
   engine.update_relations(relations);
   encode_both();
}

// The two lanes must describe the same task. The compatibility path can be
// handed mismatched object tables and has to reject them; the direct path
// cannot produce a mismatch at all, because both preparations read their object
// table from the engine's single task context. Pinning both halves keeps the
// rejection from being quietly dropped as "unreachable" later.
TEST_P(DirectViewSuccessorTest, MismatchedObjectTablesAreRejected)
{
   const auto param = GetParam();
   const auto ctx = mifrost_test::make_context(param.domain, param.problem);
   const auto [successor, action] = mifrost_test::find_successor(ctx);
   (void) action;
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto current_input = adapter.make_input(ctx.root);
   auto foreign_input = adapter.make_input(successor);
   // Objects come from the shared task context unless the input owns a table of
   // its own, so drop the context to give the successor lane a shorter one.
   foreign_input.objects = adapter.get_task_context()->objects;
   if(foreign_input.objects.size() < 2) {
      GTEST_SKIP() << "Fixture does not provide enough objects.";
   }
   foreign_input.objects.pop_back();
   foreign_input.task_context.reset();

   const mifrost::SemanticSuccessorHGraphEncoderEngine engine(adapter.get_task_context());
   EXPECT_THROW((void) engine.encode(current_input, foreign_input), std::invalid_argument);

   // Direct path: both lanes carry the engine's own object table by construction.
   const auto current_view = adapter.make_state_view(ctx.root);
   const auto successor_view = adapter.make_state_view(successor);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto current_prepared = mifrost::canonical::detail::make_hgraph_view_preparation(
      adapter.get_task_context(), current_view, empty_actions
   );
   const auto successor_prepared = mifrost::canonical::detail::make_state_only_view_preparation(
      adapter.get_task_context(), successor_view
   );
   EXPECT_EQ(
      mifrost::canonical::detail::semantic_objects(current_prepared),
      mifrost::canonical::detail::semantic_objects(successor_prepared)
   );
   EXPECT_NO_THROW(
      (void) engine.encode(current_view, empty_actions, successor_view, empty_actions)
   );
}

INSTANTIATE_TEST_SUITE_P(
   SmallDomains,
   DirectViewSuccessorTest,
   ::testing::ValuesIn(mifrost_test::kSmallDomains),
   [](const ::testing::TestParamInfo< mifrost_test::DomainCase >& info) {
      return mifrost_test::case_name(info.param);
   }
);

}  // namespace
