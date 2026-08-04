#include "mifrost/core/encoders/flat/flat_composition.hpp"

#include <gtest/gtest.h>

#include <array>
#include <future>
#include <memory>
#include <set>
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
      projection_ = builder.add_projection(
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
      context.emit_projection(projection_, source_args);
   }

  private:
   mutable FlatProjectionHandle projection_;
};

class PaddingProjectionComponent final: public FlatEmitterComponent {
  public:
   [[nodiscard]] std::string_view name() const noexcept override { return "padding_projection"; }

   void declare_schema(FlatSchemaPlanBuilder& builder) const override
   {
      (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, true);
      builder.register_relation(
         predicate_relation_key("fact"), unary_layout(), RelationUsage::state
      );
      builder.register_relation(
         predicate_relation_key("anchor"), unary_layout(), RelationUsage::parent
      );
      (void) builder.add_projection(
         FlatRelationProjection{
            .source_relation = predicate_relation_key("fact"),
            .output_relation = predicate_relation_key("anchor"),
            .slots = {FlatSlotResolver::source(0)},
         }
      );
   }
};

class ReorderingProjectionComponent final: public FlatEmitterComponent {
  public:
   [[nodiscard]] std::string_view name() const noexcept override { return "reordering"; }

   void add_leading_projection() { add_leading_projection_ = true; }

   void declare_schema(FlatSchemaPlanBuilder& builder) const override
   {
      const auto entity_type = builder.declare_node_type("entity", FlatNodeKind::object, 1, true);
      builder.register_relation(
         predicate_relation_key("fact"), unary_layout(), RelationUsage::state
      );
      builder.register_relation(
         predicate_relation_key("anchor"), unary_layout(), RelationUsage::parent
      );
      if(add_leading_projection_) {
         (void) builder.add_projection(
            FlatRelationProjection{
               .source_relation = predicate_relation_key("fact"),
               .output_relation = predicate_relation_key("anchor"),
               .slots = {FlatSlotResolver::source(0)},
            }
         );
      }
      primary_projection_ = builder.add_projection(
         FlatRelationProjection{
            .source_relation = predicate_relation_key("fact"),
            .output_relation = predicate_relation_key("anchor"),
            .slots = {FlatSlotResolver::node_ref(FlatNodeRef{entity_type, "a"})},
         }
      );
   }

   void plan_graph(const FlatInputView&, FlatNodePlanBuilder& builder) const override
   {
      (void) builder.add_node("entity", "a");
   }

   void emit(const FlatInputView&, FlatGraphContext& context) const override
   {
      context.emit_projection(primary_projection_, std::array< int64_t, 1 >{0});
   }

  private:
   bool add_leading_projection_ = false;
   mutable FlatProjectionHandle primary_projection_;
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
   auto compiled = std::make_shared< CompiledFlatPlan >(plan.compile(config));

   ASSERT_EQ(compiled->schema().size(), 2u);
   ASSERT_EQ(compiled->node_schema().id_for("entity"), 0);

   const DemoInput first{{"a", "b"}, 7};
   const DemoInput second{{"a"}, 9};
   const std::array inputs{FlatInputView::from(first), FlatInputView::from(second)};
   const auto encoding = compiled->encode_batch(inputs);
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

TEST(FlatCompositionTest, ProjectionHandlesRemainStableAcrossPlanRecompilation)
{
   auto projection = std::make_shared< ProjectionComponent >();
   FlatEncoderPlan first_plan;
   first_plan.emplace_component< FactsComponent >();
   first_plan.add_component(projection);
   const auto first = first_plan.compile();

   FlatEncoderPlan second_plan;
   second_plan.emplace_component< FactsComponent >();
   second_plan.emplace_component< PaddingProjectionComponent >();
   second_plan.add_component(projection);
   (void) second_plan.compile();

   const DemoInput input{{"a"}, 1};
   const auto encoding = first.encode(FlatInputView::from(input));
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationArgsField)).values
      ),
      (std::vector< int64_t >{0, 0})
   );
}

TEST(FlatCompositionTest, ProjectionHandlesSurviveLocalOrdinalChanges)
{
   auto projection = std::make_shared< ReorderingProjectionComponent >();
   FlatEncoderPlan first_plan;
   first_plan.add_component(projection);
   const auto first = first_plan.compile();

   projection->add_leading_projection();
   FlatEncoderPlan second_plan;
   second_plan.add_component(projection);
   (void) second_plan.compile();

   const DemoInput input{{"a"}, 1};
   const auto encoding = first.encode(FlatInputView::from(input));
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationArgsField)).values
      ),
      (std::vector< int64_t >{0})
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

TEST(FlatCompositionTest, FieldComponentRejectsDuplicateDeclarations)
{
   const GraphFieldSpec spec{
      .dtype = GraphFieldDType::I64,
      .mode = GraphFieldMode::STACK,
      .dim = 1,
   };
   EXPECT_THROW(
      (void) FlatFieldEmitterComponent(
         "fields",
         std::vector< FlatFieldEmitterComponent::FieldDeclaration >{
            {"marker", spec}, {"marker", spec}
         }
      ),
      std::invalid_argument
   );
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
   const auto composition_input = std::move(input_builder).finish();
   const auto actual = compiled.encode(FlatInputView::from(composition_input));

   const auto parity = compare_flat_batch_encodings(expected, actual);
   ASSERT_TRUE(parity.equal) << parity.mismatch;
}

TEST(FlatCompositionTest, SemanticRelationEngineUsesCompiledPlanAfterParity)
{
   SemanticFlatRelationEncoderEngine engine(
      std::vector< SemanticPredicateSpec >{{
         SemanticPredicateCategory::fluent,
         "at",
         1,
      }},
      {}
   );
   SemanticFlatRelationInput input;
   input.objects = {"a"};
   input.state_facts = {{0, {0}}};

   const auto encoding = engine.encode(input);

   EXPECT_EQ(encoding.num_graphs, 1);
}

TEST(FlatCompositionTest, SemanticRelationRejectsForeignTaskContext)
{
   const auto first_context = std::make_shared< SemanticTaskContext >(SemanticTaskContext{
      .predicates = {{SemanticPredicateCategory::fluent, "at", 1}},
      .objects = {"a"},
   });
   const auto second_context = std::make_shared< SemanticTaskContext >(*first_context);
   SemanticFlatRelationEncoderEngine engine(first_context);
   SemanticFlatRelationInput input;
   input.task_context = second_context;
   input.state_facts = {{0, {0}}};

   EXPECT_THROW((void) engine.encode(input), std::invalid_argument);
}

TEST(FlatCompositionTest, SemanticRelationCompositionPreservesPredicateVirtualNodeMetadata)
{
   SemanticFlatRelationEncoderEngine::Config config;
   config.use_predicate_virtual_nodes = true;
   SemanticFlatRelationEncoderEngine engine(
      std::vector< SemanticPredicateSpec >{{
         SemanticPredicateCategory::fluent,
         "at",
         1,
      }},
      {},
      config
   );
   SemanticFlatRelationInput input;
   input.objects = {"a"};
   input.state_facts = {{0, {0}}};

   const auto encoding = engine.encode(input);

   EXPECT_EQ(encoding.num_graphs, 1);
}

TEST(FlatCompositionTest, SemanticRelationBatchParityCoversRelationArgumentLayouts)
{
   for(const bool relation_major : {false, true}) {
      SemanticFlatRelationEncoderEngine::Config config;
      config.pack_relation_args_relation_major = relation_major;
      SemanticFlatRelationEncoderEngine engine(
         std::vector< SemanticPredicateSpec >{{
            SemanticPredicateCategory::fluent,
            "at",
            1,
         }},
         {},
         config
      );
      SemanticFlatRelationInput input;
      input.objects = {"a", "b"};
      input.state_facts = {{0, {0}}, {0, {1}}};

      const auto encoding = engine.encode_batch({input, input});

      EXPECT_EQ(encoding.num_graphs, 2);
   }
}

TEST(FlatCompositionTest, SemanticRelationCompositionPreservesLazyTargetNames)
{
   SemanticFlatRelationEncoderEngine::Config config;
   config.target_sources = {TargetSource::actions};
   SemanticFlatRelationEncoderEngine engine(
      std::vector< SemanticPredicateSpec >{{SemanticPredicateCategory::fluent, "at", 1}},
      std::vector< SemanticActionSpec >{{"move", 1}},
      config
   );
   SemanticFlatRelationInput input;
   input.objects = {"a"};
   input.state_facts = {{0, {0}}};
   input.actions = {{0, {0}}};

   const auto encoding = engine.encode(input);

   ASSERT_FALSE(encoding.lazy_target_name_strings.empty());
   EXPECT_EQ(encoding.lazy_target_name_strings.front(), "(move a)");
}

TEST(FlatCompositionTest, SemanticRelationCompositionPreservesEmptyTargetNames)
{
   SemanticFlatRelationEncoderEngine::Config config;
   config.target_sources = {TargetSource::actions};
   SemanticFlatRelationEncoderEngine engine(
      std::vector< SemanticPredicateSpec >{{SemanticPredicateCategory::fluent, "at", 1}},
      std::vector< SemanticActionSpec >{{"move", 1}},
      config
   );
   SemanticFlatRelationInput input;
   input.objects = {"a"};
   input.state_facts = {{0, {0}}};

   const auto encoding = engine.encode(input);

   ASSERT_TRUE(encoding.graph_attrs.contains(std::string(kTargetNamesAttr)));
   EXPECT_TRUE(
      std::get< std::vector< std::string > >(encoding.graph_attrs.at(std::string(kTargetNamesAttr)))
         .empty()
   );
}

TEST(FlatCompositionTest, SemanticRelationBatchCompositionOmitsEmptyTargetNamesWhenMixed)
{
   SemanticFlatRelationEncoderEngine::Config config;
   config.target_sources = {TargetSource::actions};
   SemanticFlatRelationEncoderEngine engine(
      std::vector< SemanticPredicateSpec >{{SemanticPredicateCategory::fluent, "at", 1}},
      std::vector< SemanticActionSpec >{{"move", 1}},
      config
   );
   SemanticFlatRelationInput without_action;
   without_action.objects = {"a"};
   without_action.state_facts = {{0, {0}}};
   SemanticFlatRelationInput with_action = without_action;
   with_action.actions = {{0, {0}}};

   const auto encoding = engine.encode_batch({without_action, with_action});

   EXPECT_FALSE(encoding.graph_attrs.contains(std::string(kTargetNamesAttr)));
   EXPECT_FALSE(encoding.lazy_target_name_strings.empty());
}

TEST(FlatCompositionTest, SemanticRelationDirectComponentsCoverSemanticMetadataLanes)
{
   SemanticFlatRelationEncoderEngine::Config config;
   config.include_lgan_edges = true;
   config.target_sources = {
      TargetSource::goals,
      TargetSource::actions,
      TargetSource::history,
   };
   config.lgan_anchor_sources = config.target_sources;
   SemanticFlatRelationEncoderEngine engine(
      std::vector< SemanticPredicateSpec >{{SemanticPredicateCategory::fluent, "at", 1}},
      std::vector< SemanticActionSpec >{{"move", 1}},
      config
   );
   SemanticFlatRelationInput input;
   input.objects = {"a"};
   input.state_facts = {{0, {0}}};
   input.goals = {{SemanticAtom{0, {0}}, true}};
   input.actions = {{0, {0}}};
   input.history = {{-1, {{{SemanticAtom{0, {0}}, true}}}}};

   const auto encoding = engine.encode(input);

   EXPECT_TRUE(encoding.node_names.contains("entity"));
   EXPECT_FALSE(encoding.lazy_target_name_strings.empty());
   EXPECT_TRUE(encoding.graph_fields.contains(std::string(kLGANTNSizesField)));
}

TEST(FlatCompositionTest, SemanticRelationParityMatrixUsesCompiledPlan)
{
   const std::vector< SemanticPredicateSpec > predicates = {
      {SemanticPredicateCategory::static_predicate, "ready", 0},
      {SemanticPredicateCategory::fluent, "at", 1},
      {SemanticPredicateCategory::derived, "clear", 1},
   };
   const std::vector< SemanticActionSpec > actions = {{"move", 1}};
   SemanticFlatRelationInput input;
   input.objects = {"a", "b"};
   input.state_facts = {{0, {}}, {1, {0}}, {2, {1}}};
   input.goals = {
      {{1, {1}}, true},
      {{2, {0}}, false},
   };
   input.subgoal_layers = {{{{2, {1}}, true}}};
   input.actions = {{0, {0}}, {0, {0}}};
   input.history = {
      SemanticHistoryEntry{
         .dt = -1,
         .literals =
            {
               SemanticLiteral{SemanticAtom{1, {0}}, true},
               SemanticLiteral{SemanticAtom{2, {1}}, false},
            },
      },
      SemanticHistoryEntry{
         .dt = -3,
         .literals = {SemanticLiteral{SemanticAtom{1, {1}}, true}},
      },
   };
   input.history_max_steps = 2;

   std::vector< SemanticFlatRelationEncoderEngine::Config > configs;
   configs.emplace_back();
   configs.back().max_goal_level = 1;
   configs.back().target_sources = {
      TargetSource::actions,
      TargetSource::goals,
      TargetSource::subgoals,
      TargetSource::history,
   };
   configs.back().lgan_anchor_sources = configs.back().target_sources;
   configs.back().include_lgan_edges = true;
   configs.back().support_literals = true;
   configs.back().goal_derivations = {
      GoalDerivation::plain,
      GoalDerivation::satisfied,
      GoalDerivation::unsatisfied,
   };
   configs.back().use_predicate_virtual_nodes = true;

   configs.emplace_back();
   configs.back().max_goal_level = 1;
   configs.back().export_node_names = false;
   configs.back().target_sources = {TargetSource::actions, TargetSource::history};
   configs.back().goal_derivations = {GoalDerivation::plain};

   for(auto config : configs) {
      for(const bool relation_major : {false, true}) {
         config.pack_relation_args_relation_major = relation_major;
         SemanticFlatRelationEncoderEngine engine(predicates, actions, config);
         const auto encoding = engine.encode_batch({input, input});

         EXPECT_EQ(encoding.num_graphs, 2);
      }
   }
}

TEST(FlatCompositionTest, SemanticRelationBuilderPathMatchesOneShotComposition)
{
   SemanticFlatRelationEncoderEngine::Config config;
   config.max_goal_level = 1;
   config.target_sources = {
      TargetSource::actions,
      TargetSource::goals,
      TargetSource::subgoals,
      TargetSource::history,
   };
   config.lgan_anchor_sources = config.target_sources;
   config.include_lgan_edges = true;
   config.support_literals = true;
   config.use_predicate_virtual_nodes = true;
   SemanticFlatRelationEncoderEngine engine(
      std::vector< SemanticPredicateSpec >{
         {SemanticPredicateCategory::static_predicate, "ready", 0},
         {SemanticPredicateCategory::fluent, "at", 1},
      },
      std::vector< SemanticActionSpec >{{"move", 1}},
      config
   );
   SemanticFlatRelationInput input;
   input.objects = {"a", "b"};
   input.state_facts = {{0, {}}, {1, {0}}};
   input.goals = {{{1, {1}}, true}};
   input.subgoal_layers = {{{{1, {0}}, false}}};
   input.actions = {{0, {0}}};
   input.history = {{-1, {{{1, {1}}, true}}}};

   const auto expected = engine.encode(input);
   BatchBuilder builder;
   engine.encode(input, builder);
   builder.next_graph();
   auto actual = builder.build();
   engine.finalize_batch_encoding(actual);

   const auto parity = compare_flat_batch_encodings(expected, actual);
   ASSERT_TRUE(parity.equal) << parity.mismatch;
   EXPECT_EQ(
      std::get< std::vector< std::string > >(actual.graph_attrs.at(std::string(kTargetGroupsAttr))),
      (std::vector< std::string >{"goal", "subgoal", "action", "history"})
   );
}

TEST(FlatCompositionTest, SemanticRelationCompiledPlanIsSafeForConcurrentEncodes)
{
   SemanticFlatRelationEncoderEngine::Config config;
   config.target_sources = {TargetSource::actions, TargetSource::goals};
   config.lgan_anchor_sources = config.target_sources;
   config.include_lgan_edges = true;
   config.use_predicate_virtual_nodes = true;
   SemanticFlatRelationEncoderEngine engine(
      std::vector< SemanticPredicateSpec >{{SemanticPredicateCategory::fluent, "at", 1}},
      std::vector< SemanticActionSpec >{{"move", 1}},
      config
   );
   SemanticFlatRelationInput input;
   input.objects = {"a", "b"};
   input.state_facts = {{0, {0}}};
   input.goals = {{{0, {1}}, true}};
   input.actions = {{0, {0}}, {0, {1}}};
   const auto expected = engine.encode(input);

   std::vector< std::future< BatchBuilder::BatchEncoding > > jobs;
   for(size_t index = 0; index < 8; ++index) {
      jobs.push_back(std::async(std::launch::async, [&engine, &input] {
         return engine.encode(input);
      }));
   }
   for(auto& job : jobs) {
      const auto parity = compare_flat_batch_encodings(expected, job.get());
      ASSERT_TRUE(parity.equal) << parity.mismatch;
   }
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

TEST(FlatCompositionTest, RejectsUnownedRelationsWithMultipleDeclaringEmitters)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   const auto relation = FlatCompositionRelationSpec{
      .key = predicate_relation_key("fact"),
      .layout = unary_layout(),
      .usage = RelationUsage::state,
   };
   plan.emplace_component< FlatRelationEmitterComponent >("facts", std::vector{relation});
   plan.emplace_component< FlatRelationEmitterComponent >("other", std::vector{relation});
   const auto compiled = plan.compile();

   FlatCompositionInput input;
   input.objects = {"a"};
   input.relations = {{
      compiled.schema().id_for(predicate_relation_key("fact")),
      {0},
   }};
   EXPECT_THROW((void) compiled.encode(FlatInputView::from(input)), std::invalid_argument);
}

TEST(FlatCompositionTest, RejectsNamedRelationsThatTheOwnerDidNotDeclare)
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

   FlatCompositionInput input;
   input.objects = {"a"};
   input.relations = {{
      compiled.schema().id_for(predicate_relation_key("fact")),
      {0},
      "missing",
   }};
   EXPECT_THROW((void) compiled.encode(FlatInputView::from(input)), std::invalid_argument);
}

TEST(FlatCompositionTest, BroadcastPolicyExplicitlyDuplicatesUnownedRelations)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >();
   const auto relation = FlatCompositionRelationSpec{
      .key = predicate_relation_key("fact"),
      .layout = unary_layout(),
      .usage = RelationUsage::state,
   };
   plan.emplace_component< FlatRelationEmitterComponent >("facts", std::vector{relation});
   plan.emplace_component< FlatRelationEmitterComponent >("other", std::vector{relation});
   FlatCompositionConfig config;
   config.unowned_relation_policy = FlatUnownedRelationPolicy::broadcast;
   const auto compiled = plan.compile(config);

   FlatCompositionInput input;
   input.objects = {"a"};
   input.relations = {{
      compiled.schema().id_for(predicate_relation_key("fact")),
      {0},
   }};
   const auto encoding = compiled.encode(FlatInputView::from(input));
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationCountsField)).values
      ),
      (std::vector< int64_t >{2})
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

TEST(FlatCompositionTest, LazyTargetNamesRequireDeclaredMetadataOwnership)
{
   class UnownedTargetNamesComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "unowned_names"; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity");
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }
      void write_metadata(const FlatGraphContext&, FlatMetadataWriter& writer) const override
      {
         const std::array names{std::string("target")};
         writer.add_lazy_target_names(names);
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< UnownedTargetNamesComponent >();
   const auto compiled = plan.compile();
   const FlatCompositionInput input;
   EXPECT_THROW((void) compiled.encode(FlatInputView::from(input)), std::invalid_argument);
}

TEST(FlatCompositionTest, ObjectNamesCanOnlyBeWrittenOncePerGraph)
{
   class DuplicateObjectNamesComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "duplicate_names"; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity");
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }
      void declare_metadata(FlatMetadataPlanBuilder& builder) const override
      {
         builder.claim_object_names();
      }
      void write_metadata(const FlatGraphContext&, FlatMetadataWriter& writer) const override
      {
         writer.set_object_names({"a"});
         writer.set_object_names({"b"});
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< DuplicateObjectNamesComponent >();
   const auto compiled = plan.compile();
   const FlatCompositionInput input;
   EXPECT_THROW((void) compiled.encode(FlatInputView::from(input)), std::invalid_argument);
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
   const FlatCompositionInput input;
   const auto encoding = plan.compile().encode(FlatInputView::from(input));
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
   const FlatCompositionInput input;
   const auto encoding = compiled.encode(FlatInputView::from(input));
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationArgsField)).values
      ),
      (std::vector< int64_t >{0})
   );
}

TEST(FlatCompositionTest, PreparedScratchIsAvailableToNodeFeatureWriters)
{
   struct PreparedNode {
      float value = 0.0F;
   };
   class PreparedFeatureComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "prepared_feature"; }

      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, false);
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }

      void declare_node_features(FlatNodeFeaturePlanBuilder& builder) const override
      {
         builder.register_feature("entity", "prepared", 1);
      }

      void plan_graph(const FlatInputView&, FlatNodePlanBuilder& builder) const override
      {
         (void) builder.add_node("entity", "a");
      }

      void prepare_graph(const FlatInputView&, FlatGraphContext& context) const override
      {
         context.scratch.emplace< PreparedNode >(7.0F);
      }

      void write_node_features(
         const FlatGraphContext& context,
         FlatNodeFeatureWriter& writer
      ) const override
      {
         const auto value = context.scratch.get< PreparedNode >().value;
         writer.set("entity", "prepared", std::span{&value, size_t{1}});
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< PreparedFeatureComponent >();
   const FlatCompositionInput input;
   const auto encoding = plan.compile().encode(FlatInputView::from(input));
   EXPECT_EQ(
      std::get< std::vector< float > >(encoding.columns.at("entity/prepared").data),
      (std::vector< float >{7.0F})
   );
}

TEST(FlatCompositionTest, DeclaredNodeFeaturesMustBeWrittenForEveryGraph)
{
   class MissingFeatureComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "missing_feature"; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity");
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }
      void declare_node_features(FlatNodeFeaturePlanBuilder& builder) const override
      {
         builder.register_feature("entity", "x", 1);
      }
      void plan_graph(const FlatInputView&, FlatNodePlanBuilder& builder) const override
      {
         (void) builder.add_node("entity", "a");
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< MissingFeatureComponent >();
   const auto compiled = plan.compile();
   const FlatCompositionInput input;
   EXPECT_THROW((void) compiled.encode(FlatInputView::from(input)), std::invalid_argument);
}

TEST(FlatCompositionTest, GraphMetadataMustRemainConstantAcrossBatch)
{
   struct MetadataInput {
      int64_t value = 0;
   };
   class MetadataComponent final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "varying_metadata"; }

      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         (void) builder.declare_node_type("entity", FlatNodeKind::object, 1, false);
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
      }

      void declare_metadata(FlatMetadataPlanBuilder& builder) const override
      {
         builder.claim_graph_attr("constant");
      }

      void
      write_metadata(const FlatGraphContext& context, FlatMetadataWriter& writer) const override
      {
         writer.set_graph_attr("constant", context.input.get< MetadataInput >().value);
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< MetadataComponent >();
   const auto compiled = plan.compile();
   const MetadataInput first{1};
   const MetadataInput second{2};
   const std::array inputs{FlatInputView::from(first), FlatInputView::from(second)};
   EXPECT_THROW((void) compiled.encode_batch(inputs), std::invalid_argument);

   BatchBuilder builder;
   compiled.encode(FlatInputView::from(first), builder);
   EXPECT_THROW(compiled.encode(FlatInputView::from(second), builder), std::invalid_argument);
}

TEST(FlatCompositionTest, RelationArgumentNodeTypeIsExportedConsistently)
{
   FlatEncoderPlan plan;
   plan.emplace_component< FlatObjectNodeComponent >(
      "vertices", "vertex", FlatNodeKind::object, 1, true
   );
   plan.emplace_component< FlatRelationEmitterComponent >(
      "facts",
      std::vector< FlatCompositionRelationSpec >{{
         .key = predicate_relation_key("fact"),
         .layout = unary_layout(),
         .usage = RelationUsage::state,
      }}
   );
   FlatCompositionConfig config;
   config.relation_args_node_type = "vertex";
   const FlatCompositionInput input;
   const auto encoding = plan.compile(config).encode(FlatInputView::from(input));
   EXPECT_EQ(
      std::get< std::string >(encoding.graph_attrs.at(std::string(kFlatEntityTypeAttr))), "vertex"
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
         (void) builder.add_projection(
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

TEST(FlatCompositionTest, RejectsOutOfRangeSourceNodeProjectionAtCompileTime)
{
   class InvalidSourceNodeProjection final: public FlatEmitterComponent {
     public:
      [[nodiscard]] std::string_view name() const noexcept override { return "invalid_source"; }
      void declare_schema(FlatSchemaPlanBuilder& builder) const override
      {
         const auto entity = builder.declare_node_type("entity");
         builder.register_relation(
            predicate_relation_key("fact"), unary_layout(), RelationUsage::state
         );
         builder.register_relation(
            predicate_relation_key("anchor"), unary_layout(), RelationUsage::parent
         );
         (void) builder.add_projection(
            FlatRelationProjection{
               .source_relation = predicate_relation_key("fact"),
               .output_relation = predicate_relation_key("anchor"),
               .slots = {FlatSlotResolver::source_node(1, entity)},
            }
         );
      }
   };

   FlatEncoderPlan plan;
   plan.emplace_component< InvalidSourceNodeProjection >();
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

TEST(FlatCompositionTest, ValidatesGenericCompositionCapabilities)
{
   const auto required = static_cast< uint32_t >(FlatCompositionCapability::state_facts)
                         | static_cast< uint32_t >(FlatCompositionCapability::goal_facts)
                         | static_cast< uint32_t >(FlatCompositionCapability::transition_effects)
                         | static_cast< uint32_t >(FlatCompositionCapability::parent_relations);
   const auto available = static_cast< uint32_t >(FlatCompositionCapability::state_facts)
                          | static_cast< uint32_t >(FlatCompositionCapability::goal_facts)
                          | static_cast< uint32_t >(FlatCompositionCapability::transition_effects);
   EXPECT_FALSE(flat_composition_capabilities_satisfied(required, available));
   const auto missing = flat_composition_missing_capabilities(required, available);
   EXPECT_EQ(missing, static_cast< uint32_t >(FlatCompositionCapability::parent_relations));
   EXPECT_TRUE(flat_composition_capabilities_satisfied(required, required));
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

TEST(FlatCompositionTest, ProjectionMapsSourceIdentityToPlannedNodeIndex)
{
   FlatNodeSchemaBuilder schema_builder;
   const auto entity_type = schema_builder.declare_node_type("entity");
   const auto schema = std::move(schema_builder).finalize();
   FlatNodePlanBuilder node_builder(schema);
   (void) node_builder.add_node_from_source(entity_type, 42, "b");
   (void) node_builder.add_node_from_source(entity_type, 7, "a");
   const auto nodes = std::move(node_builder).finish();

   const CompiledFlatRelationProjection projection{
      .source_relation_id = 0,
      .output_relation_id = 1,
      .slots = {FlatSlotResolver::source_node(0, entity_type)},
   };
   EXPECT_EQ(projection.project(std::array< int64_t, 1 >{7}, nodes), (std::vector< int64_t >{1}));
   EXPECT_THROW(
      (void) projection.project(std::array< int64_t, 1 >{1}, nodes), std::invalid_argument
   );
}

TEST(FlatCompositionTest, DistinctSourceIdentitiesCannotCollapseOntoOneNodeKey)
{
   FlatNodeSchemaBuilder schema_builder;
   const auto entity_type = schema_builder.declare_node_type("entity");
   const auto schema = std::move(schema_builder).finalize();
   FlatNodePlanBuilder node_builder(schema);
   (void) node_builder.add_node_from_source(entity_type, 1, "a");
   EXPECT_THROW(
      (void) node_builder.add_node_from_source(entity_type, 2, "a"), std::invalid_argument
   );
}

}  // namespace
}  // namespace mifrost
