#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <string>

#include "mifrost/core/encoders/flat/flat_relation_schema.hpp"

namespace mifrost {
namespace {

FlatTupleLayout unary_layout()
{
   return make_predicate_tuple_layout(1, {}, /*use_predicate_virtual_nodes=*/false);
}

FlatTupleLayout binary_layout()
{
   return make_predicate_tuple_layout(2, {}, /*use_predicate_virtual_nodes=*/false);
}

TEST(FlatRelationSchemaTest, StableAlphabeticalOrderingAndIds)
{
   FlatRelationSchemaBuilder builder;
   builder.register_relation(predicate_relation_key("on"), unary_layout(), RelationUsage::state);
   builder.register_relation(predicate_relation_key("at"), unary_layout(), RelationUsage::state);
   builder.register_relation(
      predicate_relation_key("holding"), unary_layout(), RelationUsage::state
   );

   auto schema = std::move(builder).finalize(0, false, {GoalDerivation::plain}, "empty");

   ASSERT_EQ(schema.size(), 3u);
   EXPECT_EQ(schema.names(), (std::vector< std::string >{"at", "holding", "on"}));
   EXPECT_EQ(schema.id_for(std::string("at")), 0);
   EXPECT_EQ(schema.id_for(std::string("holding")), 1);
   EXPECT_EQ(schema.id_for(std::string("on")), 2);
}

TEST(FlatRelationSchemaTest, DuplicateRegistrationWithCompatibleLayoutIsAccepted)
{
   FlatRelationSchemaBuilder builder;
   builder.register_relation(predicate_relation_key("at"), unary_layout(), RelationUsage::state);
   builder.register_relation(predicate_relation_key("at"), unary_layout(), RelationUsage::goal);

   EXPECT_EQ(builder.size(), 1u);
   auto schema = std::move(builder).finalize(0, false, {GoalDerivation::plain}, "empty");
   ASSERT_EQ(schema.size(), 1u);
   // First-registered usage wins.
   EXPECT_EQ(schema.sources(), (std::vector< std::string >{"state"}));
}

TEST(FlatRelationSchemaTest, IncompatibleLayoutIsRejectedImmediately)
{
   FlatRelationSchemaBuilder builder;
   builder.register_relation(predicate_relation_key("at"), unary_layout(), RelationUsage::state);
   EXPECT_THROW(
      builder.register_relation(
         predicate_relation_key("at"), binary_layout(), RelationUsage::state
      ),
      std::invalid_argument
   );
}

TEST(FlatRelationSchemaTest, FormattedNameCollisionAcrossDistinctKeysThrowsAtFinalize)
{
   FlatRelationSchemaBuilder builder;
   // Two structurally distinct keys (opaque vs. bare predicate) that happen to format to the same
   // exported name must be rejected at finalize(), not silently merged.
   builder.register_relation(opaque_relation_key("at"), unary_layout(), RelationUsage::parent);
   builder.register_relation(predicate_relation_key("at"), unary_layout(), RelationUsage::state);

   EXPECT_THROW(
      (void) std::move(builder).finalize(0, false, {GoalDerivation::plain}, "empty"),
      std::invalid_argument
   );
}

TEST(FlatRelationSchemaTest, EmptyBuilderThrowsConfiguredMessage)
{
   FlatRelationSchemaBuilder builder;
   try {
      (void) std::move(builder).finalize(
         0, false, {GoalDerivation::plain}, "no relations for domain"
      );
      FAIL() << "expected std::invalid_argument";
   } catch(const std::invalid_argument& error) {
      EXPECT_STREQ(error.what(), "no relations for domain");
   }
}

TEST(FlatRelationSchemaTest, StructuredLookupMatchesStringLookup)
{
   FlatRelationSchemaBuilder builder;
   const auto key = predicate_relation_key(
      "at", /*polarity=*/true, GoalLevel(0), GoalDerivation::satisfied
   );
   builder.register_relation(key, unary_layout(), RelationUsage::goal_derivation);
   auto schema = std::move(builder).finalize(
      0, false, {GoalDerivation::plain, GoalDerivation::satisfied}, "empty"
   );

   ASSERT_TRUE(schema.try_id_for(key).has_value());
   EXPECT_EQ(schema.id_for(key), schema.id_for(std::string("[+]at[g][sat]")));
   EXPECT_FALSE(schema.try_id_for(predicate_relation_key("nope")).has_value());
   EXPECT_FALSE(schema.try_id_for(std::string("nope")).has_value());
}

TEST(FlatRelationSchemaTest, AsMetadataRoundTripsAllVectorsConsistently)
{
   FlatRelationSchemaBuilder builder;
   builder.register_relation(predicate_relation_key("at"), unary_layout(), RelationUsage::state);
   builder.register_relation(
      action_relation_key("move"),
      make_nonpredicate_tuple_layout(1, {FlatSlotRole::action_slot}),
      RelationUsage::action
   );
   auto schema = std::move(builder).finalize(0, false, {GoalDerivation::plain}, "empty");

   const auto& metadata = schema.as_metadata();
   ASSERT_EQ(metadata.relation_names.size(), 2u);
   EXPECT_EQ(metadata.relation_names, schema.names());
   EXPECT_EQ(metadata.relation_arities, schema.arities());
   EXPECT_EQ(metadata.relation_sources, schema.sources());
   EXPECT_EQ(metadata.relation_logical_arities, schema.logical_arities());
   EXPECT_EQ(metadata.relation_encoded_arities, schema.encoded_arities());
   EXPECT_EQ(metadata.relation_slot_roles, schema.slot_roles());
   EXPECT_EQ(metadata.relation_slot_role_offsets, schema.slot_role_offsets());
   EXPECT_EQ(metadata.slot_role_names, schema.slot_role_names());
   EXPECT_EQ(metadata.relation_dict.arity.size(), schema.relation_dict().arity.size());
   // "at" sorts before "move" alphabetically.
   EXPECT_EQ(schema.names()[0], "at");
   EXPECT_EQ(schema.names()[1], "move");
}

TEST(FlatRelationSchemaTest, RelationUsageExportedAsSourceLabel)
{
   FlatRelationSchemaBuilder builder;
   builder.register_relation(
      predicate_relation_key("at"), unary_layout(), RelationUsage::goal_satisfaction
   );
   auto schema = std::move(builder).finalize(0, false, {GoalDerivation::plain}, "empty");
   EXPECT_EQ(schema.sources(), (std::vector< std::string >{"goal_satisfaction"}));
}

}  // namespace
}  // namespace mifrost
