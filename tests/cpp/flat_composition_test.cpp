#include "mifrost/core/encoders/flat/flat_composition.hpp"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace mifrost {
namespace {

struct DemoInput {
   std::vector< std::string > objects;
   int64_t marker = 0;
};

FlatTupleLayout unary_layout()
{
   return make_predicate_tuple_layout(1, {}, false);
}

class FactsComponent final: public FlatEmitterComponent {
  public:
   [[nodiscard]] std::string_view name() const noexcept override { return "facts"; }

   void declare_schema(FlatSchemaPlanBuilder& builder) const override
   {
      (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, true);
      builder.register_relation(
         predicate_relation_key("fact"), unary_layout(), RelationUsage::state
      );
   }

   void declare_fields(FlatFieldPlanBuilder& builder) const override
   {
      builder.register_field(
         "marker",
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
      );
   }

   void plan_graph(const FlatInputView& input, FlatNodePlanBuilder& builder) const override
   {
      for(const auto& object : input.get< DemoInput >().objects) {
         (void) builder.add_node("entity", object);
      }
   }

   void emit(const FlatInputView& input, FlatGraphContext& context) const override
   {
      const auto relation_id = context.relation_id(predicate_relation_key("fact"));
      for(const auto& object : input.get< DemoInput >().objects) {
         const auto node = context.nodes.index(
            FlatNodeRef{context.nodes.schema().id_for("entity"), object}
         );
         const std::array args{node};
         context.emit(relation_id, args);
      }
   }

   void write_fields(const FlatGraphContext& context, FlatFieldWriter& writer) const override
   {
      writer.set("marker", std::span{&context.input.get< DemoInput >().marker, size_t{1}});
   }
};

class ProjectionComponent final: public FlatEmitterComponent {
  public:
   [[nodiscard]] std::string_view name() const noexcept override { return "projections"; }

   void declare_schema(FlatSchemaPlanBuilder& builder) const override
   {
      const auto entity_type = builder.declare_node_type("entity", FlatNodeKind::object, 1, true);
      builder.register_relation(
         predicate_relation_key("fact"), unary_layout(), RelationUsage::state
      );
      builder.register_relation(
         predicate_relation_key("anchor"), unary_layout(), RelationUsage::parent
      );
      builder.add_projection(
         FlatRelationProjection{
            .source_relation = predicate_relation_key("fact"),
            .output_relation = predicate_relation_key("anchor"),
            .slots = {FlatSlotResolver::node_ref(FlatNodeRef{entity_type, "a"})},
         }
      );
   }

   void emit(const FlatInputView&, FlatGraphContext& context) const override
   {
      const std::array source_args{int64_t{0}};
      context.emit_projection(0, source_args);
   }
};

class ConflictingFieldComponent final: public FlatEmitterComponent {
  public:
   [[nodiscard]] std::string_view name() const noexcept override { return "conflict"; }

   void declare_fields(FlatFieldPlanBuilder& builder) const override
   {
      builder.register_field(
         "marker",
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
      );
   }
};

TEST(FlatCompositionTest, CompilesSharedNodesAndRunsOneNativeBatch)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FactsComponent >();
   plan.emplace_component< ProjectionComponent >();

   FlatCompositionConfig config;
   config.goal_derivations = {GoalDerivation::plain};
   auto compiled = plan.compile(config);

   ASSERT_EQ(compiled.schema().size(), 2u);
   ASSERT_EQ(compiled.node_schema().id_for("entity"), 0);

   const DemoInput first{{"a", "b"}, 7};
   const DemoInput second{{"a"}, 9};
   const std::array inputs{FlatInputView::from(first), FlatInputView::from(second)};
   const auto encoding = compiled.encode_batch(inputs);
   const FlatBatchRuntime runtime(compiled);
   const auto runtime_encoding = runtime.encode_batch(inputs);

   EXPECT_EQ(encoding.num_graphs, 2);
   EXPECT_EQ(runtime_encoding.num_graphs, encoding.num_graphs);
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         runtime_encoding.graph_fields.at(std::string(kRelationArgsField)).values
      ),
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationArgsField)).values
      )
   );
   ASSERT_TRUE(encoding.graph_fields.contains(std::string(kRelationCountsField)));
   ASSERT_TRUE(encoding.graph_fields.contains(std::string(kRelationArgsField)));
   ASSERT_TRUE(encoding.graph_fields.contains("marker"));
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(encoding.graph_fields.at("marker").values),
      (std::vector< int64_t >{7, 9})
   );
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationCountsField)).values
      )
         .size(),
      4u
   );
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationArgsField)).values
      ),
      (std::vector< int64_t >{0, 2, 0, 1, 2})
   );
   EXPECT_EQ(
      std::get< std::string >(encoding.graph_attrs.at(std::string(kRelationArgsLayoutAttr))),
      std::string(kRelationArgsRelationMajorLayout)
   );
}

TEST(FlatCompositionTest, BuiltInComponentsComposeResolvedRelationsAndFields)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   plan.emplace_component< FlatRelationEmitterComponent >(
      "facts",
      std::vector< FlatCompositionRelationSpec >{
         FlatCompositionRelationSpec{
            .key = predicate_relation_key("fact"),
            .layout = unary_layout(),
            .usage = RelationUsage::state,
         },
         FlatCompositionRelationSpec{
            .key = predicate_relation_key("pair"),
            .layout = make_predicate_tuple_layout(2, {}, false),
            .usage = RelationUsage::parent,
         },
      }
   );
   plan.emplace_component< FlatFieldEmitterComponent >(
      "metadata",
      std::vector< FlatFieldEmitterComponent::FieldDeclaration >{
         {
            "marker",
            GraphFieldSpec{
               .dtype = GraphFieldDType::I64,
               .mode = GraphFieldMode::STACK,
               .dim = 1,
            },
         },
      }
   );

   const auto compiled = plan.compile();
   FlatCompositionInputBuilder input_builder(compiled.schema());
   input_builder.add_object("a");
   input_builder.add_object("b");
   input_builder.add_relation(predicate_relation_key("fact"), std::array< int64_t, 1 >{0});
   input_builder.add_relation(predicate_relation_key("pair"), std::array< int64_t, 2 >{0, 1});
   input_builder.set_field("marker", NumericColumnData{std::vector< int64_t >{17}});
   auto input = std::move(input_builder).finish();

   const auto encoding = compiled.encode(FlatInputView::from(input));
   ASSERT_EQ(encoding.num_graphs, 1);
   EXPECT_EQ(encoding.object_names, (std::vector< std::string >{"a", "b"}));
   ASSERT_EQ(encoding.node_names.at("entity"), (std::vector< std::string >{"a", "b"}));
   EXPECT_EQ(encoding.node_counts.at("entity"), 2);
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationCountsField)).values
      ),
      (std::vector< int64_t >{1, 1})
   );
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationArgsField)).values
      ),
      (std::vector< int64_t >{0, 0, 1})
   );
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(encoding.graph_fields.at("marker").values),
      (std::vector< int64_t >{17})
   );
}

TEST(FlatCompositionTest, BuiltInComponentsRejectMissingOrMismatchedInputFields)
{
   const auto spec = GraphFieldSpec{
      .dtype = GraphFieldDType::F32,
      .mode = GraphFieldMode::STACK,
      .dim = 1,
   };
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   plan.emplace_component< FlatRelationEmitterComponent >(
      "facts",
      std::vector< FlatCompositionRelationSpec >{{
         .key = predicate_relation_key("fact"),
         .layout = unary_layout(),
         .usage = RelationUsage::state,
      }}
   );
   plan.emplace_component< FlatFieldEmitterComponent >(
      "metadata", std::vector< FlatFieldEmitterComponent::FieldDeclaration >{{"score", spec}}
   );
   const auto compiled = plan.compile();

   FlatCompositionInput missing;
   EXPECT_THROW((void) compiled.encode(FlatInputView::from(missing)), std::invalid_argument);

   FlatCompositionInput wrong;
   wrong.fields = {
      {
         "score",
         NumericColumnData{std::vector< int64_t >{1}},
      },
   };
   EXPECT_THROW((void) compiled.encode(FlatInputView::from(wrong)), std::invalid_argument);
}

TEST(FlatCompositionTest, ComparesNativeEncodingsForExactParity)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   plan.emplace_component< FlatRelationEmitterComponent >(
      "facts",
      std::vector< FlatCompositionRelationSpec >{{
         .key = predicate_relation_key("fact"),
         .layout = unary_layout(),
         .usage = RelationUsage::state,
      }}
   );
   plan.emplace_component< FlatFieldEmitterComponent >(
      "metadata",
      std::vector< FlatFieldEmitterComponent::FieldDeclaration >{{
         "marker",
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         },
      }}
   );
   const auto compiled = plan.compile();
   FlatCompositionInput input;
   input.objects = {"a"};
   input.relations = {{compiled.schema().id_for(predicate_relation_key("fact")), {0}}};
   input.fields = {{"marker", NumericColumnData{std::vector< int64_t >{3}}}};

   const auto expected = compiled.encode(FlatInputView::from(input));
   const auto same = compare_flat_batch_encodings(expected, expected);
   EXPECT_TRUE(same.equal);
   EXPECT_TRUE(same.mismatch.empty());

   auto actual = expected;
   actual.graph_fields.at("marker").values = std::vector< int64_t >{4};
   const auto different = compare_flat_batch_encodings(expected, actual);
   EXPECT_FALSE(different.equal);
   EXPECT_EQ(different.mismatch, "graph_fields[marker]");
}

TEST(FlatCompositionTest, BuiltInRelationEmittersHonorExplicitRecordOwners)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   plan.emplace_component< FlatRelationEmitterComponent >(
      "facts",
      std::vector< FlatCompositionRelationSpec >{{
         .key = predicate_relation_key("fact"),
         .layout = unary_layout(),
         .usage = RelationUsage::state,
      }}
   );
   plan.emplace_component< FlatRelationEmitterComponent >(
      "parents",
      std::vector< FlatCompositionRelationSpec >{{
         .key = predicate_relation_key("parent"),
         .layout = unary_layout(),
         .usage = RelationUsage::parent,
      }}
   );
   const auto compiled = plan.compile();

   FlatCompositionInput input;
   input.objects = {"a"};
   input.relations = {
      {
         compiled.schema().id_for(predicate_relation_key("fact")),
         {0},
         "facts",
      },
      {
         compiled.schema().id_for(predicate_relation_key("parent")),
         {0},
         "parents",
      },
   };
   const auto encoding = compiled.encode(FlatInputView::from(input));
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationCountsField)).values
      ),
      (std::vector< int64_t >{1, 1})
   );
}

TEST(FlatCompositionTest, InputBuilderPrevalidatesCarrierIdentityAndArity)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   plan.emplace_component< FlatRelationEmitterComponent >(
      "facts",
      std::vector< FlatCompositionRelationSpec >{{
         .key = predicate_relation_key("fact"),
         .layout = unary_layout(),
         .usage = RelationUsage::state,
      }}
   );
   const auto compiled = plan.compile();
   FlatCompositionInputBuilder input(compiled.schema());
   input.add_object("a");
   EXPECT_THROW(input.add_object("a"), std::invalid_argument);
   EXPECT_THROW(
      input.add_relation(predicate_relation_key("fact"), std::array< int64_t, 2 >{0, 1}),
      std::invalid_argument
   );
   EXPECT_THROW(input.add_relation(99, std::array< int64_t, 1 >{0}), std::invalid_argument);
}

TEST(FlatCompositionTest, RejectsMultipleObjectNameMetadataOwners)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >("objects_a");
   plan.emplace_component< FlatObjectNodeComponent >("objects_b");
   plan.emplace_component< FlatRelationEmitterComponent >(
      "facts",
      std::vector< FlatCompositionRelationSpec >{{
         .key = predicate_relation_key("fact"),
         .layout = unary_layout(),
         .usage = RelationUsage::state,
      }}
   );
   EXPECT_THROW((void) plan.compile(), std::invalid_argument);
}

TEST(FlatCompositionTest, ResolvesRelationAliasesBeforeEmission)
{
   class AliasComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "alias"; }

      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, true);
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
         builder.register_relation_alias(
            predicate_relation_key("state_fact"), predicate_relation_key("fact")
         );
      }

      void plan_graph(const FlatInputView&, FlatNodePlanBuilder& builder) const override
      {
         (void) builder.add_node("entity", "a");
      }

      void emit(const FlatInputView&, FlatGraphContext& context) const override
      {
         const std::array args{int64_t{0}};
         context.emit(predicate_relation_key("state_fact"), args);
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< AliasComponent >();
   const auto compiled = plan.compile();
   FlatCompositionInput input;
   input.objects = {"a"};
   const auto encoding = compiled.encode(FlatInputView::from(input));
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationCountsField)).values
      ),
      (std::vector< int64_t >{1})
   );
}

TEST(FlatCompositionTest, RejectsFieldOwnershipCollision)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FactsComponent >();
   plan.emplace_component< ConflictingFieldComponent >();
   EXPECT_THROW((void) plan.compile(), std::invalid_argument);
}

TEST(FlatCompositionTest, RejectsProjectionWithUnknownNodeType)
{
   class InvalidProjection final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "invalid"; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity");
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
         builder.register_relation(
            predicate_relation_key("anchor"), unary_layout(), RelationUsage::parent
         );
         builder.add_projection(
            FlatRelationProjection{
               .source_relation = predicate_relation_key("fact"),
               .output_relation = predicate_relation_key("anchor"),
               .slots = {FlatSlotResolver::node_ref(FlatNodeRef{42, "a"})},
            }
         );
      }
   };
   FlatEncoderPlan plan;
   plan.emplace_component< InvalidProjection >();
   EXPECT_THROW((void) plan.compile(), std::invalid_argument);
}

TEST(FlatCompositionTest, RejectsEmptyPlans)
{
   FlatEncoderPlan plan;
   EXPECT_THROW((void) plan.compile(), std::invalid_argument);
}

TEST(FlatCompositionTest, RequiresConfiguredRelationArgumentOffsetNode)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FactsComponent >();
   FlatCompositionConfig config;
   config.relation_args_node_type = "vertex";
   EXPECT_THROW((void) plan.compile(config), std::invalid_argument);
}

TEST(FlatCompositionTest, RejectsUndeclaredFieldOffsetNode)
{
   class InvalidField final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "invalid_field"; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity");
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }
      void declare_fields(FlatFieldPlanBuilder& builder) const override
      {
         builder.register_field(
            "bad",
            GraphFieldSpec{
               .dtype = GraphFieldDType::I64,
               .mode = GraphFieldMode::CAT,
               .dim = 1,
               .inc = GraphFieldInc{
                  .kind = GraphFieldInc::Kind::NODE_OFFSET,
                  .node_type = "missing",
               },
            }
         );
      }
   };
   FlatEncoderPlan plan;
   plan.emplace_component< InvalidField >();
   EXPECT_THROW((void) plan.compile(), std::invalid_argument);
}

TEST(FlatCompositionTest, EmptyBatchStillCarriesCompiledSchemaMetadata)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FactsComponent >();
   auto compiled = plan.compile();

   const std::vector< FlatInputView > inputs;
   const auto encoding = compiled.encode_batch(inputs);
   EXPECT_EQ(encoding.num_graphs, 0);
   EXPECT_EQ(
      std::get< std::vector< std::string > >(
         encoding.graph_attrs.at(std::string(kRelationNamesAttr))
      ),
      (std::vector< std::string >{"fact"})
   );
   EXPECT_TRUE(encoding.graph_fields.contains(std::string(kRelationCountsField)));
}

TEST(FlatCompositionTest, PublishesExternalModeCapabilityContract)
{
   const auto contracts = flat_external_mode_contracts();
   ASSERT_EQ(contracts.size(), 6u);
   EXPECT_EQ(contracts.front().name, "concurrent_internal");
   EXPECT_EQ(contracts.back().name, "concurrent_internal_action_hybrid_tree");

   const auto& rooted = flat_external_mode_contract(
      FlatExternalMode::concurrent_internal_tree_rooted
   );
   EXPECT_NE(
      rooted.required_components
         & static_cast< uint32_t >(FlatExternalComponent::root_action_nodes),
      0U
   );
   EXPECT_NE(
      rooted.required_components & static_cast< uint32_t >(FlatExternalComponent::ground_actions),
      0U
   );

   const auto& comparison = flat_external_mode_contract(
      FlatExternalMode::concurrent_internal_comparison_tree
   );
   EXPECT_NE(
      comparison.required_components & static_cast< uint32_t >(FlatExternalComponent::shared_state),
      0U
   );
   EXPECT_EQ(
      comparison.required_components
         & static_cast< uint32_t >(FlatExternalComponent::ground_actions),
      0U
   );
}

TEST(FlatCompositionTest, ProjectionResolvesSourceAndConstantSlots)
{
   FlatNodeSchemaBuilder schema_builder;
   const auto entity_type = schema_builder.declare_node_type("entity");
   const auto schema = std::move(schema_builder).finalize();
   FlatNodePlanBuilder node_builder(schema);
   (void) node_builder.add_node(entity_type, "a");
   const auto nodes = std::move(node_builder).finish();

   const CompiledFlatRelationProjection projection{
      .source_relation_id = 0,
      .output_relation_id = 1,
      .slots = {FlatSlotResolver::source(1), FlatSlotResolver::constant_value(7)},
   };
   EXPECT_EQ(
      projection.project(std::array< int64_t, 2 >{2, 3}, nodes), (std::vector< int64_t >{3, 7})
   );
   const CompiledFlatRelationProjection invalid{
      .source_relation_id = 0,
      .output_relation_id = 1,
      .slots = {FlatSlotResolver::source(2)},
   };
   EXPECT_THROW(
      (void) invalid.project(std::array< int64_t, 2 >{2, 3}, nodes), std::invalid_argument
   );
}

}  // namespace
}  // namespace mifrost
