#include "mifrost/core/semantic/views.hpp"

#include <gtest/gtest.h>

#include <span>
#include <string>

#include "mifrost/core/views/canonical.hpp"

namespace mifrost {
namespace {

TEST(ViewsTest, SemanticViewsExposeRecordsLazily)
{
   const SemanticAtom fluent_atom{3, {7, 11}};
   const SemanticAtom derived_atom{4, {11}};
   const SemanticLiteral goal{fluent_atom, false};
   const SemanticGroundAction action{2, {7, 11}};

   const semantic::StateView state{
      .fluent = semantic::AtomsView(std::span{&fluent_atom, 1}),
      .derived = semantic::AtomsView(std::span{&derived_atom, 1}),
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
      .fluent = semantic::AtomsView(std::span{&atom, 1}),
      .derived = semantic::AtomsView(std::span< const SemanticAtom >{}),
   };

   const auto fluent = state.fluent_atoms();
   EXPECT_TRUE(canonical::contains_atom(fluent, semantic::AtomView{&same}));
   EXPECT_FALSE(canonical::contains_atom(fluent, semantic::AtomView{&different}));
   EXPECT_TRUE(canonical::satisfies_literal(fluent, semantic::LiteralView{&positive}));
   EXPECT_TRUE(canonical::satisfies_literal(fluent, semantic::LiteralView{&negative}));
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
