#include "mifrost/core/semantic/views.hpp"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>

#if __has_include("mifrost/core/encoders/flat/view_flat_relation_input.hpp")
   #error "The deleted callback-based FlatRelationViewInput carrier must not return."
#endif

#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
#include "mifrost/core/encoders/flat/view_flat_relation_encoder.hpp"
#include "mifrost/core/views/canonical.hpp"

namespace mifrost {
namespace {

struct ViewEncoderProbe {
   [[nodiscard]] std::shared_ptr< const SemanticTaskContext > get_task_context() const
   {
      return {};
   }
   [[nodiscard]] int encode(const SemanticFlatRelationInput&) const { return 0; }

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   [[nodiscard]] int encode(const State&, Goals&&, Actions&&) const
   {
      return 0;
   }
};

static_assert(requires(
   const ViewEncoderProbe& encoder,
   const semantic::StateView& state,
   semantic::LiteralsView goals,
   semantic::GroundActionsView actions
) { encoder.encode(state, goals, actions); });

static_assert(views::LiteralLayerRange< semantic::SubgoalLayersView >);
static_assert(views::HistoryRange< semantic::HistoryView >);
static_assert(requires(
   const SemanticFlatRelationEncoderEngine& encoder,
   const semantic::StateView& state,
   semantic::LiteralsView goals,
   semantic::SubgoalLayersView subgoal_layers,
   semantic::GroundActionsView actions,
   semantic::HistoryView history
) { encoder.encode(state, goals, subgoal_layers, actions, history, std::optional< int64_t >{}); });

static_assert(views::PredicateView< semantic::PredicateView >);
static_assert(views::ObjectView< semantic::ObjectView >);
static_assert(views::ActionSchemaView< semantic::ActionSchemaView >);
static_assert(views::HistoryEntryView< semantic::HistoryEntryView >);

TEST(ViewsTest, SemanticViewsExposeRecordsLazily)
{
   const SemanticAtom fluent_atom{3, {7, 11}};
   const SemanticAtom derived_atom{4, {11}};
   const SemanticLiteral goal{fluent_atom, false};
   const SemanticGroundAction action{2, {7, 11}};

   const semantic::StateView state{
      semantic::AtomsView(std::span{&fluent_atom, 1}),
      semantic::AtomsView(std::span{&derived_atom, 1}),
   };

   const auto fluent = *state.fluent_atoms().begin();
   EXPECT_EQ(fluent.predicate_id(), 3);
   EXPECT_EQ(fluent.arguments().size(), 2U);
   EXPECT_EQ(fluent.arguments()[1], 11);

   const auto literal = *semantic::LiteralsView(std::span{&goal, 1}).begin();
   EXPECT_TRUE(literal.is_negated());
   EXPECT_EQ(literal.atom().predicate_id(), 3);

   const auto ground_action = *semantic::GroundActionsView(std::span{&action, 1}).begin();
   EXPECT_EQ(ground_action.schema_id(), 2);
   EXPECT_EQ(ground_action.arguments().front(), 7);
}

TEST(ViewsTest, CanonicalAlgorithmsOperateOnViews)
{
   const SemanticAtom atom{3, {7, 11}};
   const SemanticAtom same{3, {7, 11}};
   const SemanticAtom different{3, {11, 7}};
   const SemanticLiteral positive{same, true};
   const SemanticLiteral negative{different, false};

   const semantic::StateView state{
      semantic::AtomsView(std::span{&atom, 1}),
      semantic::AtomsView(std::span< const SemanticAtom >{}),
   };

   const auto fluent = state.fluent_atoms();
   EXPECT_TRUE(canonical::contains_atom(fluent, semantic::AtomView{&same}));
   EXPECT_FALSE(canonical::contains_atom(fluent, semantic::AtomView{&different}));
   EXPECT_TRUE(canonical::satisfies_literal(fluent, semantic::LiteralView{&positive}));
   EXPECT_TRUE(canonical::satisfies_literal(fluent, semantic::LiteralView{&negative}));
}

TEST(ViewsTest, CanonicalFlatTraversalVisitsEachLane)
{
   const SemanticAtom atom{3, {7}};
   const SemanticLiteral goal{atom, true};
   const SemanticGroundAction action{2, {7}};
   const semantic::StateView state{
      semantic::AtomsView(std::span{&atom, 1}),
      semantic::AtomsView(std::span< const SemanticAtom >{}),
   };
   const std::array lanes_expected{
      canonical::FlatLane::state,
      canonical::FlatLane::goal,
      canonical::FlatLane::action,
   };
   std::array< canonical::FlatLane, 3 > lanes{};
   std::size_t index = 0;

   canonical::visit_flat_lanes(
      state,
      semantic::LiteralsView(std::span{&goal, 1}),
      semantic::GroundActionsView(std::span{&action, 1}),
      [&](const canonical::FlatLane lane, const auto&) { lanes[index++] = lane; }
   );

   EXPECT_EQ(lanes, lanes_expected);
}

TEST(ViewsTest, SemanticViewBridgeMaterializesCanonicalInput)
{
   const SemanticAtom fluent{3, {7}};
   const SemanticAtom derived{4, {11}};
   const SemanticLiteral goal{fluent, false};
   const SemanticGroundAction action{2, {7, 11}};
   const semantic::StateView state{
      semantic::AtomsView(std::span{&fluent, 1}),
      semantic::AtomsView(std::span{&derived, 1}),
   };
   const auto context = std::make_shared< const SemanticTaskContext >(
      SemanticTaskContext{.objects = {"a", "b"}}
   );

   const auto input = canonical::make_semantic_flat_relation_input(
      context,
      state,
      semantic::LiteralsView(std::span{&goal, 1}),
      semantic::GroundActionsView(std::span{&action, 1})
   );

   ASSERT_FALSE(input.use_default_goals);
   ASSERT_EQ(input.state_facts.size(), 2U);
   ASSERT_EQ(input.goals.size(), 1U);
   ASSERT_EQ(input.actions.size(), 1U);
   EXPECT_EQ(input.state_facts[0], fluent);
   EXPECT_EQ(input.state_facts[1], derived);
   EXPECT_EQ(input.goals[0], goal);
   EXPECT_EQ(input.actions[0], action);
}

TEST(ViewsTest, SemanticSchemaViewsUseExplicitCompactIds)
{
   const SemanticPredicateSpec predicate{
      .category = SemanticPredicateCategory::derived,
      .name = "reachable",
      .arity = 2,
   };
   const SemanticActionSpec action{.name = "move", .arity = 2};

   const semantic::PredicateView predicate_view{&predicate, 17};
   const semantic::ActionSchemaView action_view{&action, 9};

   EXPECT_EQ(predicate_view.id(), 17);
   EXPECT_EQ(predicate_view.name(), "reachable");
   EXPECT_EQ(predicate_view.arity(), 2U);
   EXPECT_EQ(predicate_view.category(), views::PredicateCategory::derived);
   EXPECT_EQ(action_view.id(), 9);
   EXPECT_EQ(action_view.name(), "move");
   EXPECT_EQ(action_view.arity(), 2U);
}

}  // namespace
}  // namespace mifrost
