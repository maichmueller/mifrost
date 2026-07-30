#include "mifrost/core/encoders/common/relation_key.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <unordered_set>

namespace mifrost {
namespace {

TEST(RelationKeyTest, BarePredicateFormatsAsBaseNameOnly)
{
   const auto key = predicate_relation_key("at");
   EXPECT_EQ(format_relation_name(key), "at");
}

TEST(RelationKeyTest, ActionKeyFormatsAsBaseNameOnly)
{
   const auto key = action_relation_key("move");
   EXPECT_EQ(format_relation_name(key), "move");
}

TEST(RelationKeyTest, OpaqueKeyPassesThroughVerbatim)
{
   const auto key = opaque_relation_key("_parent_");
   EXPECT_EQ(key.family, RelationFamily::auxiliary);
   EXPECT_EQ(format_relation_name(key), "_parent_");
}

TEST(RelationKeyTest, GoalPredicateWithPolarityLevelAndDerivation)
{
   const auto key = predicate_relation_key(
      "at", /*polarity=*/true, GoalLevel(0), GoalDerivation::satisfied
   );
   EXPECT_EQ(format_relation_name(key), "[+]at[g][sat]");
}

TEST(RelationKeyTest, NegativePolarityAndUnsatisfiedDerivation)
{
   const auto key = predicate_relation_key(
      "at", /*polarity=*/false, GoalLevel(1), GoalDerivation::unsatisfied
   );
   EXPECT_EQ(format_relation_name(key), "[-]at[sg][unsat]");
}

TEST(RelationKeyTest, HistoryModifierMatchesLegacySuffixPlacement)
{
   // Legacy call site: RelationFormatter::format_predicate(name, nullopt, nullopt, polarity,
   // "[hist]") -> "[+]at[hist]" / "[-]at[hist]".
   const auto positive = predicate_relation_key(
      "at", /*polarity=*/true, std::nullopt, std::nullopt, "[hist]"
   );
   const auto negative = predicate_relation_key(
      "at", /*polarity=*/false, std::nullopt, std::nullopt, "[hist]"
   );
   EXPECT_EQ(format_relation_name(positive), "[+]at[hist]");
   EXPECT_EQ(format_relation_name(negative), "[-]at[hist]");
}

TEST(RelationKeyTest, ModifierPlacedBeforeGoalLevelAndDerivation)
{
   // The user-facing disambiguation example: augmenting "[+]at[g][sat]" with a "[next]" modifier
   // must produce "[+]at[next][g][sat]", not "[+]at[g][sat][next]".
   const auto key = predicate_relation_key(
      "at", /*polarity=*/true, GoalLevel(0), GoalDerivation::satisfied, "[next]"
   );
   EXPECT_EQ(format_relation_name(key), "[+]at[next][g][sat]");
   EXPECT_NE(format_relation_name(key), "[+]at[g][sat][next]");
}

TEST(RelationKeyTest, StateAnchoredRendersAfterGoalLevelAndDerivation)
{
   // Mirrors the pre-existing horizon "state-anchored" naming convention: `name + "[state]"`
   // applied to an already-fully-formatted name, independent of the modifier slot.
   const auto key = predicate_relation_key(
      "at",
      /*polarity=*/true,
      GoalLevel(0),
      GoalDerivation::satisfied,
      /*modifier=*/"",
      /*state_anchored=*/true
   );
   EXPECT_EQ(format_relation_name(key), "[+]at[g][sat][state]");
}

TEST(RelationKeyTest, ModifierAndStateAnchoredAreDistinctAndNonAmbiguous)
{
   // A modifier and state_anchored are separate structural fields; they must never format to the
   // same string as one another, and their combination must place the modifier before goal-level
   // and state_anchored strictly last.
   const auto with_modifier_only = predicate_relation_key(
      "at", true, GoalLevel(0), GoalDerivation::satisfied, "[next]", false
   );
   const auto with_state_anchor_only = predicate_relation_key(
      "at", true, GoalLevel(0), GoalDerivation::satisfied, "", true
   );
   const auto with_both = predicate_relation_key(
      "at", true, GoalLevel(0), GoalDerivation::satisfied, "[next]", true
   );

   EXPECT_EQ(format_relation_name(with_modifier_only), "[+]at[next][g][sat]");
   EXPECT_EQ(format_relation_name(with_state_anchor_only), "[+]at[g][sat][state]");
   EXPECT_EQ(format_relation_name(with_both), "[+]at[next][g][sat][state]");
   EXPECT_NE(
      format_relation_name(with_modifier_only), format_relation_name(with_state_anchor_only)
   );
}

TEST(RelationKeyTest, EqualKeysCompareEqualAndHashEqual)
{
   const auto lhs = predicate_relation_key("at", true, GoalLevel(0), GoalDerivation::satisfied);
   const auto rhs = predicate_relation_key("at", true, GoalLevel(0), GoalDerivation::satisfied);
   EXPECT_EQ(lhs, rhs);
   EXPECT_EQ(RelationKeyHash{}(lhs), RelationKeyHash{}(rhs));
}

TEST(RelationKeyTest, DistinctFieldsProduceDistinctKeys)
{
   const auto base = predicate_relation_key("at", true, GoalLevel(0), GoalDerivation::satisfied);
   const auto different_polarity = predicate_relation_key(
      "at", false, GoalLevel(0), GoalDerivation::satisfied
   );
   const auto different_level = predicate_relation_key(
      "at", true, GoalLevel(1), GoalDerivation::satisfied
   );
   const auto different_derivation = predicate_relation_key(
      "at", true, GoalLevel(0), GoalDerivation::unsatisfied
   );
   const auto different_family = action_relation_key("at");

   EXPECT_NE(base, different_polarity);
   EXPECT_NE(base, different_level);
   EXPECT_NE(base, different_derivation);
   EXPECT_NE(base, different_family);
}

TEST(RelationKeyTest, HashSetDeduplicatesEqualKeysOnly)
{
   std::unordered_set< RelationKey, RelationKeyHash > keys;
   keys.insert(predicate_relation_key("at", true, GoalLevel(0), GoalDerivation::satisfied));
   keys.insert(predicate_relation_key("at", true, GoalLevel(0), GoalDerivation::satisfied));
   keys.insert(predicate_relation_key("at", false, GoalLevel(0), GoalDerivation::satisfied));
   EXPECT_EQ(keys.size(), 2u);
}

TEST(RelationKeyTest, RelationUsageSourceLabelsMatchLegacyStrings)
{
   EXPECT_EQ(relation_usage_source_label(RelationUsage::state), "state");
   EXPECT_EQ(relation_usage_source_label(RelationUsage::goal), "goal");
   EXPECT_EQ(relation_usage_source_label(RelationUsage::goal_derivation), "goal_derivation");
   EXPECT_EQ(relation_usage_source_label(RelationUsage::goal_satisfaction), "goal_satisfaction");
   EXPECT_EQ(relation_usage_source_label(RelationUsage::action), "action");
   EXPECT_EQ(relation_usage_source_label(RelationUsage::history), "history");
   EXPECT_EQ(relation_usage_source_label(RelationUsage::parent), "parent");
   EXPECT_EQ(relation_usage_source_label(RelationUsage::sibling), "sibling");
   EXPECT_EQ(relation_usage_source_label(RelationUsage::cousin), "cousin");
}

}  // namespace
}  // namespace mifrost
