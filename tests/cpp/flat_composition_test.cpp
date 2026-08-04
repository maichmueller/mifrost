#include "mifrost/core/encoders/flat/flat_composition.hpp"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

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
   EXPECT_FALSE(encoding.columns.empty());
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

   FlatCompositionInput duplicate;
   duplicate.fields = {
      {
         "score",
         NumericColumnData{std::vector< float >{1.0F}},
      },
      {
         "score",
         NumericColumnData{std::vector< float >{2.0F}},
      },
   };
   EXPECT_THROW((void) compiled.encode(FlatInputView::from(duplicate)), std::invalid_argument);
}

TEST(FlatCompositionTest, BuiltInNodeRecordComponentPlansTypedRows)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   plan.emplace_component< FlatNodeRecordComponent >(
      "actions", "action", FlatNodeKind::action, 1, true
   );
   plan.emplace_component< FlatRelationEmitterComponent >(
      "facts",
      std::vector< FlatCompositionRelationSpec >{{
         .key = predicate_relation_key("fact"),
         .layout = unary_layout(),
         .usage = RelationUsage::state,
      }}
   );
   const auto compiled = plan.compile();
   ASSERT_EQ(compiled.node_features().size(), 2u);
   FlatCompositionInput input;
   input.objects = {"a"};
   input.nodes = {{"action", "move"}};
   const auto encoding = compiled.encode(FlatInputView::from(input));
   EXPECT_EQ(encoding.node_counts.at("entity"), 1);
   EXPECT_EQ(encoding.node_counts.at("action"), 1);
   EXPECT_EQ(encoding.columns.size(), 2u);
   EXPECT_EQ(encoding.node_names.at("action"), (std::vector< std::string >{"move"}));
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

TEST(FlatCompositionTest, ComposedCarrierMatchesMinimalSemanticRelationBaseline)
{
   SemanticFlatRelationEncoderEngine::Config legacy_config;
   legacy_config.goal_derivations.clear();
   legacy_config.pack_relation_args_relation_major = false;
   SemanticFlatRelationEncoderEngine legacy(
      std::vector< SemanticPredicateSpec >{{
         SemanticPredicateCategory::fluent,
         "at",
         1,
      }},
      {},
      legacy_config
   );
   SemanticFlatRelationInput semantic_input;
   semantic_input.objects = {"a", "b"};
   semantic_input.state_facts = {{0, {0}}};
   const auto expected = legacy.encode(semantic_input);

   const auto entity_offset = GraphFieldInc{
      .kind = GraphFieldInc::Kind::NODE_OFFSET,
      .node_type = std::string(kFlatEntityNodeType),
   };
   const auto scalar_i64 = GraphFieldSpec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::STACK,
      .dim = 1,
   };
   const auto vector_i64 = GraphFieldSpec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::CAT,
      .dim = 1,
   };
   const auto indexed_i64 = GraphFieldSpec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::CAT,
      .dim = 1,
      .inc = entity_offset,
   };

   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   plan.emplace_component< FlatRelationEmitterComponent >(
      "relations",
      std::vector< FlatCompositionRelationSpec >{
         {
            .key = predicate_relation_key("at"),
            .layout = make_predicate_tuple_layout(1, {}, false),
            .usage = RelationUsage::state,
         },
         {
            .key = predicate_relation_key("at", false, std::nullopt, std::nullopt, "[hist]"),
            .layout = make_predicate_tuple_layout(1, {FlatSlotRole::history_slot}, false),
            .usage = RelationUsage::history,
         },
         {
            .key = predicate_relation_key("at", true, std::nullopt, std::nullopt, "[hist]"),
            .layout = make_predicate_tuple_layout(1, {FlatSlotRole::history_slot}, false),
            .usage = RelationUsage::history,
         },
      }
   );
   plan.emplace_component< FlatFieldEmitterComponent >(
      "semantic_metadata",
      std::vector< FlatFieldEmitterComponent::FieldDeclaration >{
         {std::string(kNodeSizesField), scalar_i64},
         {std::string(kObjectSizesField), scalar_i64},
         {std::string(kObjectIndicesField), indexed_i64},
         {std::string(kEntityRoleIdsField), vector_i64},
         {std::string(kHistoryEntitySizesField), scalar_i64},
         {std::string(kHistoryEntityIndicesField), indexed_i64},
         {std::string(kHistoryEntityDtField), vector_i64},
         {std::string(kTargetEntitySizesField), scalar_i64},
         {std::string(kTargetEntityIndicesField), indexed_i64},
         {std::string(kTargetEntityGroupIdsField), vector_i64},
      }
   );

   FlatCompositionConfig composition_config;
   composition_config.pack_relation_args_relation_major = false;
   composition_config.graph_config.target_sources = std::vector< std::string >{};
   composition_config.graph_config.lgan_anchor_sources = std::vector< std::string >{};
   composition_config.graph_config.target_symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
   composition_config.graph_config.target_entity_group_names = {"action"};
   composition_config.graph_config.lgan_tn_edge_pos = defaults::lgan_tn_edge_pos;
   composition_config.graph_config.lgan_nn_edge_pos = defaults::lgan_nn_edge_pos;
   composition_config.graph_config.lgan_rr_edge_pos = defaults::lgan_rr_edge_pos;
   const auto compiled = plan.compile(composition_config);

   FlatCompositionInputBuilder input_builder(compiled.schema_plan());
   input_builder.add_object("a");
   input_builder.add_object("b");
   input_builder.add_relation(predicate_relation_key("at"), std::array< int64_t, 1 >{0});
   input_builder.set_field(std::string(kNodeSizesField), std::vector< int64_t >{2});
   input_builder.set_field(std::string(kObjectSizesField), std::vector< int64_t >{2});
   input_builder.set_field(std::string(kObjectIndicesField), std::vector< int64_t >{0, 1});
   input_builder.set_field(std::string(kEntityRoleIdsField), std::vector< int64_t >{0, 0});
   input_builder.set_field(std::string(kHistoryEntitySizesField), std::vector< int64_t >{0});
   input_builder.set_field(std::string(kHistoryEntityIndicesField), std::vector< int64_t >{});
   input_builder.set_field(std::string(kHistoryEntityDtField), std::vector< int64_t >{});
   input_builder.set_field(std::string(kTargetEntitySizesField), std::vector< int64_t >{0});
   input_builder.set_field(std::string(kTargetEntityIndicesField), std::vector< int64_t >{});
   input_builder.set_field(std::string(kTargetEntityGroupIdsField), std::vector< int64_t >{});
   const auto actual = compiled.encode(FlatInputView::from(std::move(input_builder).finish()));

   const auto parity = compare_flat_batch_encodings(expected, actual);
   ASSERT_TRUE(parity.equal) << parity.mismatch;
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

TEST(FlatCompositionTest, ComposedMetadataWriterEnforcesGraphAttributeOwnership)
{
   class MetadataComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "metadata"; }

      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, false);
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }

      void declare_metadata(FlatMetadataPlanBuilder& builder) const override
      {
         builder.claim_graph_attr("custom_metadata");
      }

      void write_metadata(const FlatGraphContext&, FlatMetadataWriter& writer) const override
      {
         writer.set_graph_attr("custom_metadata", std::string("native"));
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< MetadataComponent >();
   const auto compiled = plan.compile();
   FlatCompositionInput input;
   const auto encoding = compiled.encode(FlatInputView::from(input));
   EXPECT_EQ(std::get< std::string >(encoding.graph_attrs.at("custom_metadata")), "native");
}

TEST(FlatCompositionTest, ComposedMetadataWriterSupportsTypedGraphAttributeWrites)
{
   class MetadataComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "typed_metadata"; }

      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, false);
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }

      void declare_metadata(FlatMetadataPlanBuilder& builder) const override
      {
         builder.claim_graph_attr("count");
         builder.claim_graph_attr("label");
         builder.claim_graph_attr("ids");
         builder.claim_graph_attr("names");
      }

      void write_metadata(const FlatGraphContext&, FlatMetadataWriter& writer) const override
      {
         writer.set_graph_attr("count", int64_t{3});
         writer.set_graph_attr("label", std::string("typed"));
         writer.set_graph_attr("ids", std::vector< int64_t >{1, 2, 3});
         writer.set_graph_attr("names", std::vector< std::string >{"a", "b"});
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< MetadataComponent >();
   const auto encoding = plan.compile().encode(FlatInputView::from(FlatCompositionInput{}));
   EXPECT_EQ(std::get< int64_t >(encoding.graph_attrs.at("count")), 3);
   EXPECT_EQ(std::get< std::string >(encoding.graph_attrs.at("label")), "typed");
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(encoding.graph_attrs.at("ids")),
      (std::vector< int64_t >{1, 2, 3})
   );
   EXPECT_EQ(
      std::get< std::vector< std::string > >(encoding.graph_attrs.at("names")),
      (std::vector< std::string >{"a", "b"})
   );
}

TEST(FlatCompositionTest, RejectsReservedGraphAttributeOwnership)
{
   class ReservedMetadataComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "reserved_metadata"; }

      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, false);
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }

      void declare_metadata(FlatMetadataPlanBuilder& builder) const override
      {
         builder.claim_graph_attr(std::string(kRelationNamesAttr));
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< ReservedMetadataComponent >();
   EXPECT_THROW((void) plan.compile(), std::invalid_argument);
}

TEST(FlatCompositionTest, PreparationUsesGraphLocalScratchWithoutComponentMutation)
{
   struct PreparedNode {
      int64_t index = -1;
   };
   class PreparedComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "prepared"; }

      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, false);
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }

      void plan_graph(const FlatInputView&, FlatNodePlanBuilder& builder) const override
      {
         (void) builder.add_node("entity", "a");
      }

      void prepare_graph(const FlatInputView&, FlatGraphContext& context) const override
      {
         context.scratch.emplace< PreparedNode >(0);
      }

      void emit(const FlatInputView&, FlatGraphContext& context) const override
      {
         const auto id = context.relation_id(predicate_relation_key("fact"));
         const std::array args{context.scratch.get< PreparedNode >().index};
         context.emit(id, args);
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< PreparedComponent >();
   const auto compiled = plan.compile();
   const auto encoding = compiled.encode(FlatInputView::from(FlatCompositionInput{}));
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationArgsField)).values
      ),
      (std::vector< int64_t >{0})
   );
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
   FlatCompositionInputBuilder input_builder(compiled.schema_plan());
   input_builder.add_object("a");
   input_builder.add_relation(predicate_relation_key("state_fact"), std::array< int64_t, 1 >{0});
   auto input = std::move(input_builder).finish();
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
   EXPECT_TRUE(flat_external_mode_satisfied(
      FlatExternalMode::concurrent_internal,
      static_cast< uint32_t >(FlatExternalComponent::state_facts)
         | static_cast< uint32_t >(FlatExternalComponent::goal_facts)
         | static_cast< uint32_t >(FlatExternalComponent::transition_effects)
   ));
   const auto missing = flat_external_mode_missing_components(
      FlatExternalMode::concurrent_internal_tree,
      static_cast< uint32_t >(FlatExternalComponent::state_facts)
         | static_cast< uint32_t >(FlatExternalComponent::goal_facts)
   );
   EXPECT_EQ(
      missing,
      static_cast< uint32_t >(FlatExternalComponent::transition_effects)
         | static_cast< uint32_t >(FlatExternalComponent::parent_relations)
   );
}

TEST(FlatCompositionTest, ProjectionResolvesSourceAndConstantSlots)
{
   EXPECT_THROW((void) FlatNodePlan{}.schema(), std::logic_error);
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
