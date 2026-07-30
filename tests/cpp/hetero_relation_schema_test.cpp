#include "mifrost/core/encoders/hetero/hetero_relation_schema.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace mifrost {
namespace {

TEST(HeteroRelationSchemaTest, RegistersAndFinalizesDeclaredRelations)
{
   HeteroRelationSchemaBuilder builder;
   builder.register_relation(
      predicate_relation_key("on"), HeteroRelationLayout{2}, RelationUsage::state
   );
   builder.register_relation(
      predicate_relation_key("at"), HeteroRelationLayout{1}, RelationUsage::state
   );
   auto schema = std::move(builder).finalize(0, false, {GoalDerivation::plain}, "empty");

   ASSERT_EQ(schema.size(), 2u);
   EXPECT_EQ(schema.names(), (std::vector< std::string >{"at", "on"}));
   EXPECT_EQ(schema.arities(), (std::vector< int64_t >{1, 2}));
   EXPECT_TRUE(schema.contains(predicate_relation_key("at")));
   EXPECT_FALSE(schema.contains(predicate_relation_key("held")));
}

TEST(HeteroRelationSchemaTest, IdempotentReregistrationWithSameArityIsAccepted)
{
   HeteroRelationSchemaBuilder builder;
   builder.register_relation(
      predicate_relation_key("at"), HeteroRelationLayout{1}, RelationUsage::state
   );
   builder.register_relation(
      predicate_relation_key("at"), HeteroRelationLayout{1}, RelationUsage::goal
   );
   EXPECT_EQ(builder.size(), 1u);
   auto schema = std::move(builder).finalize(0, false, {GoalDerivation::plain}, "empty");
   ASSERT_EQ(schema.size(), 1u);
   EXPECT_EQ(schema.sources(), (std::vector< std::string >{"state"}));
}

TEST(HeteroRelationSchemaTest, ArityMismatchOnReregistrationThrows)
{
   HeteroRelationSchemaBuilder builder;
   builder.register_relation(
      predicate_relation_key("at"), HeteroRelationLayout{1}, RelationUsage::state
   );
   EXPECT_THROW(
      builder.register_relation(
         predicate_relation_key("at"), HeteroRelationLayout{2}, RelationUsage::state
      ),
      std::invalid_argument
   );
}

TEST(HeteroRelationSchemaTest, NameForResolvesDeclaredKeyWithoutReformatting)
{
   HeteroRelationSchemaBuilder builder;
   const auto key = predicate_relation_key(
      "at", /*polarity=*/true, GoalLevel(0), GoalDerivation::satisfied
   );
   builder.register_relation(key, HeteroRelationLayout{1}, RelationUsage::goal_derivation);
   auto schema = std::move(builder).finalize(
      0, false, {GoalDerivation::plain, GoalDerivation::satisfied}, "empty"
   );

   EXPECT_EQ(schema.name_for(key), "[+]at[g][sat]");
}

TEST(HeteroRelationSchemaTest, NameForNeverThrowsForUndeclaredKeysAndMemoizes)
{
   HeteroRelationSchemaBuilder builder;
   builder.register_relation(
      predicate_relation_key("at"), HeteroRelationLayout{1}, RelationUsage::state
   );
   auto schema = std::move(builder).finalize(0, false, {GoalDerivation::plain}, "empty");

   // Verified real cases: literal-mode reuse when support_literals=false and successor-suffixed
   // types are never pre-declared, so name_for() must resolve them on demand instead of throwing.
   const auto undeclared = predicate_relation_key(
      "at", /*polarity=*/true, std::nullopt, std::nullopt, "[suc]"
   );
   EXPECT_FALSE(schema.contains(undeclared));
   const std::string& first = schema.name_for(undeclared);
   EXPECT_EQ(first, "[+]at[suc]");
   // Second lookup must return the exact same memoized entry (same address), not reformat.
   const std::string& second = schema.name_for(undeclared);
   EXPECT_EQ(&first, &second);
}

TEST(HeteroRelationSchemaTest, RelationDictShapeMatchesDeclaredRelations)
{
   HeteroRelationSchemaBuilder builder;
   builder.register_relation(
      predicate_relation_key("at"), HeteroRelationLayout{1}, RelationUsage::state
   );
   builder.register_relation(
      action_relation_key("move"), HeteroRelationLayout{2}, RelationUsage::action
   );
   auto schema = std::move(builder).finalize(1, true, {GoalDerivation::plain}, "empty");

   const auto& dict = schema.relation_dict();
   EXPECT_EQ(dict.arity.size(), 2u);
   EXPECT_EQ(dict.arity.at("at"), 1);
   EXPECT_EQ(dict.arity.at("move"), 2);
   EXPECT_EQ(dict.max_goal_level, 1);
   EXPECT_TRUE(dict.support_literals);
}

TEST(HeteroRelationSchemaTest, EmptyBuilderThrowsConfiguredMessage)
{
   HeteroRelationSchemaBuilder builder;
   try {
      (void) std::move(builder).finalize(0, false, {GoalDerivation::plain}, "no relations");
      FAIL() << "expected std::invalid_argument";
   } catch(const std::invalid_argument& error) {
      EXPECT_STREQ(error.what(), "no relations");
   }
}

}  // namespace
}  // namespace mifrost
