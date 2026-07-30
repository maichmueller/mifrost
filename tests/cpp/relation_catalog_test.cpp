#include "mifrost/core/encoders/common/relation_catalog.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace mifrost {
namespace {

std::vector< std::string > formatted_names(const RelationCatalog& catalog)
{
   std::vector< std::string > out;
   out.reserve(catalog.size());
   for(const auto& spec : catalog.specs()) {
      out.push_back(format_relation_name(spec.key));
   }
   return out;
}

bool contains(const std::vector< std::string >& names, const std::string& name)
{
   return std::ranges::find(names, name) != names.end();
}

TEST(RelationCatalogTest, AddAppendsSpecsInOrder)
{
   RelationCatalog catalog;
   catalog.add(RelationSpec{.key = predicate_relation_key("at"), .logical_arity = 2});
   catalog.add(RelationSpec{.key = action_relation_key("move"), .logical_arity = 1});

   ASSERT_EQ(catalog.size(), 2u);
   EXPECT_EQ(catalog.specs()[0].key.base_name, "at");
   EXPECT_EQ(catalog.specs()[1].key.base_name, "move");
}

TEST(RelationCatalogTest, DuplicatePredicateModifierPlacement)
{
   // The user-facing example: augmenting "[+]at[g][sat]" with a "[next]" modifier must produce
   // "[+]at[next][g][sat]", not "[+]at[g][sat][next]".
   RelationCatalog catalog;
   catalog.add(
      RelationSpec{
         .key = predicate_relation_key(
            "at", /*polarity=*/true, GoalLevel(0), GoalDerivation::satisfied
         ),
         .logical_arity = 2,
         .usage = RelationUsage::goal_derivation,
      }
   );

   const auto augmented = duplicate_predicate_relations_with_modifier(catalog, "[next]");
   const auto names = formatted_names(augmented);

   EXPECT_TRUE(contains(names, "[+]at[g][sat]"));
   EXPECT_TRUE(contains(names, "[+]at[next][g][sat]"));
   EXPECT_FALSE(contains(names, "[+]at[g][sat][next]"));
}

TEST(RelationCatalogTest, NonPredicateSpecsPassThroughUnduplicated)
{
   RelationCatalog catalog;
   catalog.add(RelationSpec{.key = action_relation_key("move"), .logical_arity = 1});
   catalog.add(RelationSpec{.key = opaque_relation_key("_parent_"), .logical_arity = 0});

   const auto augmented = duplicate_predicate_relations_with_modifier(catalog, "[next]");

   ASSERT_EQ(augmented.size(), 2u);
   const auto names = formatted_names(augmented);
   EXPECT_TRUE(contains(names, "move"));
   EXPECT_TRUE(contains(names, "_parent_"));
}

TEST(RelationCatalogTest, OriginalsPreservedAlongsideDuplicates)
{
   RelationCatalog catalog;
   catalog.add(RelationSpec{.key = predicate_relation_key("at"), .logical_arity = 2});
   catalog.add(RelationSpec{.key = predicate_relation_key("on"), .logical_arity = 2});
   catalog.add(RelationSpec{.key = action_relation_key("move"), .logical_arity = 1});

   const auto augmented = duplicate_predicate_relations_with_modifier(catalog, "[next]");

   // 2 predicate specs each produce an original + a duplicate; the 1 action spec passes through.
   EXPECT_EQ(augmented.size(), 5u);
   const auto names = formatted_names(augmented);
   EXPECT_TRUE(contains(names, "at"));
   EXPECT_TRUE(contains(names, "at[next]"));
   EXPECT_TRUE(contains(names, "on"));
   EXPECT_TRUE(contains(names, "on[next]"));
   EXPECT_TRUE(contains(names, "move"));
}

TEST(RelationCatalogTest, TransformDoesNotMutateInputCatalog)
{
   RelationCatalog catalog;
   catalog.add(RelationSpec{.key = predicate_relation_key("at"), .logical_arity = 2});

   const auto augmented = duplicate_predicate_relations_with_modifier(catalog, "[next]");
   (void) augmented;

   ASSERT_EQ(catalog.size(), 1u);
   EXPECT_TRUE(catalog.specs()[0].key.modifiers.empty());
}

TEST(RelationCatalogTest, PreservesExistingModifiers)
{
   RelationCatalog catalog;
   catalog.add(
      RelationSpec{
         .key = predicate_relation_key(
            "at", /*polarity=*/true, std::nullopt, std::nullopt, "[hist]"
         ),
         .logical_arity = 2,
      }
   );

   const auto augmented = duplicate_predicate_relations_with_modifier(catalog, "[next]");
   const auto names = formatted_names(augmented);

   EXPECT_TRUE(contains(names, "[+]at[hist]"));
   EXPECT_TRUE(contains(names, "[+]at[hist][next]"));
}

}  // namespace
}  // namespace mifrost
