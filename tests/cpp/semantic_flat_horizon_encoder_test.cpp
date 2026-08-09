#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "mifrost/core/encoders/flat/flat_composition.hpp"

namespace mifrost {
namespace {

using Node = SemanticTransitionDAG::Node;
using Edge = SemanticTransitionDAG::Edge;

std::vector< SemanticPredicateSpec > predicates()
{
   return {
      {SemanticPredicateCategory::fluent, "at", 1},
      {SemanticPredicateCategory::static_predicate, "ready", 0},
   };
}

std::vector< SemanticActionSpec > actions()
{
   return {{"move", 2}, {"finish", 0}};
}

SemanticFlatRelationInput state(int64_t object)
{
   SemanticFlatRelationInput value;
   value.objects = {"a", "b"};
   value.state_facts = {{0, {object}}, {1, {}}};
   value.goals = {{{0, {1}}, true}};
   return value;
}

std::vector< Node > sample_nodes()
{
   return {
      {
         .state = state(0),
         .index = 0,
         .depth = 0,
         .candidate_id = 0,
         .display_name = std::string("root"),
      },
      {
         .state = state(1),
         .index = 1,
         .depth = 1,
         .incoming_action = SemanticGroundAction{0, {0, 1}},
         .candidate_id = 101,
         .delta_literals = std::vector< SemanticLiteral >{{{0, {1}}, true}, {{0, {0}}, false}},
         .display_name = std::string("left"),
      },
      {
         .state = state(0),
         .index = 2,
         .depth = 1,
         .incoming_action = SemanticGroundAction{1, {}},
         .candidate_id = 202,
         .display_name = std::string("right"),
      },
   };
}

std::vector< Edge > sample_edges()
{
   return {{0, 1}, {0, 2}};
}

SemanticTransitionDAG make_dag()
{
   return SemanticTransitionDAG(predicates(), actions(), sample_nodes(), sample_edges());
}

SemanticTransitionDAG make_root_only_dag()
{
   auto nodes = sample_nodes();
   nodes.resize(1);
   return SemanticTransitionDAG(predicates(), actions(), std::move(nodes), {});
}

bool contains(const std::vector< std::string >& names, const std::string& name)
{
   return std::ranges::find(names, name) != names.end();
}

struct HorizonExtensionData {
   int64_t marker = 0;
   std::vector< std::string > node_names;
   std::vector< float > node_features;
};

class ConstantRelationNameAdapter final: public SemanticFlatHorizonRelationNameAdapter {
  public:
   [[nodiscard]] std::optional< std::string > map(std::string_view) const override
   {
      return "collapsed";
   }
};

class WrongArityRelationNameAdapter final: public SemanticFlatHorizonRelationNameAdapter {
  public:
   [[nodiscard]] std::optional< std::string > map(std::string_view source_name) const override
   {
      return std::string(source_name);
   }

   [[nodiscard]] std::optional< int64_t > target_arity(std::string_view) const override
   {
      return 99;
   }
};

class IdentityRelationNameAdapter final: public SemanticFlatHorizonRelationNameAdapter {
  public:
   [[nodiscard]] std::optional< std::string > map(std::string_view source_name) const override
   {
      return std::string(source_name);
   }
};

class FilterReadyRelationNameAdapter final: public SemanticFlatHorizonRelationNameAdapter {
  public:
   [[nodiscard]] std::optional< std::string > map(std::string_view source_name) const override
   {
      if(source_name == "ready") {
         return std::nullopt;
      }
      return std::string(source_name);
   }
};

class HorizonExtensionComponent final: public FlatEmitterComponent {
  public:
   [[nodiscard]] std::string_view name() const noexcept override { return "horizon_sdk_test"; }

   void declare_schema(FlatSchemaPlanBuilder& builder) const override
   {
      (void) builder.declare_node_type(
         "sdk_node", FlatNodeKind::auxiliary, 2, /*export_names=*/true
      );
      const auto output = opaque_relation_key("sdk_transition");
      builder.register_relation(
         output,
         make_nonpredicate_tuple_layout(0, {FlatSlotRole::state_slot}),
         RelationUsage::parent
      );
      builder.register_relation_alias(opaque_relation_key("sdk_transition_alias"), output);
      auto canonical_state = predicate_relation_key("at");
      canonical_state.state_anchored = true;
      projection_ = builder.add_projection(
         FlatRelationProjection{
            .source_relation = std::move(canonical_state),
            .output_relation = output,
            .slots = {FlatSlotResolver::source(1)},
         }
      );
   }

   void declare_fields(FlatFieldPlanBuilder& builder) const override
   {
      builder.register_field(
         "sdk_marker",
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
      );
   }

   void declare_node_features(FlatNodeFeaturePlanBuilder& builder) const override
   {
      builder.register_feature("sdk_node", "x", 2);
   }

   void declare_metadata(FlatMetadataPlanBuilder& builder) const override
   {
      builder.claim_graph_attr("sdk_metadata");
   }

   void plan_graph(const FlatInputView& input, FlatNodePlanBuilder& builder) const override
   {
      const auto& prepared = input.get< SemanticFlatHorizonPreparedGraph >();
      const auto& extension = prepared.annotations().get< HorizonExtensionData >("sdk");
      for(const auto& name : extension.node_names) {
         (void) builder.add_node("sdk_node", name);
         (void) builder.add_node(std::string(kFlatEntityNodeType), "sdk:" + name);
      }
   }

   void emit(const FlatInputView& input, FlatGraphContext& context) const override
   {
      const auto& prepared = input.get< SemanticFlatHorizonPreparedGraph >();
      EXPECT_EQ(prepared.source_graph().size(), prepared.state_entity_indices().size());
      EXPECT_FALSE(prepared.goals().empty());
      for(const auto position : prepared.target_positions()) {
         const std::array< int64_t, 2 > source_args = {0, position};
         context.emit_projection(projection_, source_args);
      }
      const auto& extension = prepared.annotations().get< HorizonExtensionData >("sdk");
      const auto entity_type = context.nodes.schema().id_for(kFlatEntityNodeType);
      const std::array< int64_t, 1 > extension_entity = {
         context.nodes.index(entity_type, "sdk:" + extension.node_names.front())
      };
      context.emit(opaque_relation_key("sdk_transition_alias"), extension_entity);
   }

   void write_node_features(
      const FlatGraphContext& context,
      FlatNodeFeatureWriter& writer
   ) const override
   {
      const auto& extension = context.input.get< SemanticFlatHorizonPreparedGraph >()
                                 .annotations()
                                 .get< HorizonExtensionData >("sdk");
      writer.set("sdk_node", "x", extension.node_features);
   }

   void write_fields(const FlatGraphContext& context, FlatFieldWriter& writer) const override
   {
      const auto& marker = context.input.get< SemanticFlatHorizonPreparedGraph >()
                              .annotations()
                              .get< HorizonExtensionData >("sdk")
                              .marker;
      writer.set("sdk_marker", std::span{&marker, size_t{1}});
   }

   void write_metadata(const FlatGraphContext&, FlatMetadataWriter& writer) const override
   {
      writer.set_graph_attr("sdk_metadata", std::string("canonical-plus-extension"));
   }

  private:
   mutable FlatProjectionHandle projection_;
};

SemanticFlatHorizonInput
extension_input(const SemanticTransitionDAG& dag, int64_t marker, std::vector< std::string > names)
{
   std::vector< float > features;
   features.reserve(names.size() * 2);
   for(size_t index = 0; index < names.size(); ++index) {
      features.push_back(static_cast< float >(marker));
      features.push_back(static_cast< float >(index));
   }
   SemanticFlatHorizonAnnotations annotations;
   annotations.emplace< HorizonExtensionData >(
      "sdk", HorizonExtensionData{marker, std::move(names), std::move(features)}
   );
   return SemanticFlatHorizonInput(dag, std::move(annotations));
}

TEST(SemanticFlatHorizonEncoderEngineTest, FullModeDefaultConfigEncodesWithoutThrowing)
{
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions());
   const auto& names = engine.get_relation_names();
   ASSERT_FALSE(names.empty());
   // Root-only variant and the split "[state]"-anchored candidate variant must both exist,
   // since the default root_policy (exclude) triggers split_full_state_relations() in full mode.
   EXPECT_TRUE(contains(names, "at"));
   EXPECT_TRUE(contains(names, "at[state]"));
   EXPECT_TRUE(contains(names, "[+]at[g]"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
   EXPECT_EQ(encoding.node_counts.at(std::string(kFlatEntityNodeType)), 5);
}

TEST(SemanticFlatHorizonEncoderEngineTest, RelationNameAdapterRejectsNonInjectiveOutputs)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.relation_name_adapter = std::make_shared< ConstantRelationNameAdapter >();
   EXPECT_THROW(
      (SemanticFlatHorizonEncoderEngine(predicates(), actions(), config)), std::invalid_argument
   );
}

TEST(SemanticFlatHorizonEncoderEngineTest, RelationNameAdapterRejectsTargetArityMismatch)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.relation_name_adapter = std::make_shared< WrongArityRelationNameAdapter >();
   try {
      (void) SemanticFlatHorizonEncoderEngine(predicates(), actions(), config);
      FAIL() << "expected target arity validation failure";
   } catch(const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find("encoded arity"), std::string::npos);
      EXPECT_NE(std::string(error.what()).find("declared arity"), std::string::npos);
   }
}

TEST(SemanticFlatHorizonEncoderEngineTest, RelationNameAdapterCanFilterCanonicalRelation)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.root_policy = RootPolicy::include;
   config.relation_name_adapter = std::make_shared< FilterReadyRelationNameAdapter >();
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   EXPECT_FALSE(contains(engine.get_relation_names(), "ready"));
   const auto encoding = engine.encode(make_dag());
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_fields.at(std::string(kRelationCountsField)).values
      )
         .size(),
      engine.get_relation_names().size()
   );
}

TEST(SemanticFlatHorizonEncoderEngineTest, IdentityRelationNameAdapterPreservesCanonicalOutput)
{
   SemanticFlatHorizonEncoderEngine::Config canonical_config;
   canonical_config.root_policy = RootPolicy::include;
   SemanticFlatHorizonEncoderEngine canonical(predicates(), actions(), canonical_config);

   auto adapted_config = canonical_config;
   adapted_config.relation_name_adapter = std::make_shared< IdentityRelationNameAdapter >();
   SemanticFlatHorizonEncoderEngine adapted(predicates(), actions(), adapted_config);

   EXPECT_EQ(canonical.get_relation_names(), adapted.get_relation_names());
   EXPECT_EQ(canonical.get_relation_arities(), adapted.get_relation_arities());
   const auto parity = compare_flat_batch_encodings(
      canonical.encode(make_dag()), adapted.encode(make_dag())
   );
   EXPECT_TRUE(parity.equal) << parity.mismatch;
}

TEST(SemanticFlatHorizonEncoderEngineTest, RelationSinkFindsOnlyExactRows)
{
   FlatRelationSink sink(2);
   const std::array< int64_t, 2 > first = {1, 2};
   const std::array< int64_t, 2 > second = {1, 3};
   const std::array< int64_t, 2 > absent = {2, 1};
   sink.emit(0, first);
   sink.emit(0, second);
   sink.emit(1, std::span< const int64_t >{});

   EXPECT_TRUE(sink.contains_exact(0, first));
   EXPECT_TRUE(sink.contains_exact(0, second));
   EXPECT_FALSE(sink.contains_exact(0, absent));
   EXPECT_TRUE(sink.contains_exact(1, std::span< const int64_t >{}));
}

TEST(SemanticFlatHorizonEncoderEngineTest, RootOnlyGraphPreservesEmptyTargetNames)
{
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions());

   const auto encoding = engine.encode(make_root_only_dag());

   ASSERT_TRUE(encoding.graph_attrs.contains(std::string(kTargetNamesAttr)));
   EXPECT_TRUE(
      std::get< std::vector< std::string > >(encoding.graph_attrs.at(std::string(kTargetNamesAttr)))
         .empty()
   );
}

TEST(SemanticFlatHorizonEncoderEngineTest, DeltaModeRegistersLiteralCandidateRelations)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.transition_mode = SemanticHorizonMode::delta;
   config.support_literals = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto& names = engine.get_relation_names();
   EXPECT_TRUE(contains(names, "[+]at"));
   EXPECT_TRUE(contains(names, "[-]at"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, BatchEncodingUsesCompiledPlanForRelationMajorPacking)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.pack_relation_args_relation_major = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto encoding = engine.encode_batch({make_dag(), make_dag()});

   EXPECT_EQ(encoding.num_graphs, 2);
   EXPECT_EQ(
      std::get< std::string >(encoding.graph_attrs.at(std::string(kRelationArgsLayoutAttr))),
      std::string(kRelationArgsRelationMajorLayout)
   );
}

TEST(SemanticFlatHorizonEncoderEngineTest, EmptyBatchUsesCompiledSchemaPath)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.pack_relation_args_relation_major = true;
   config.include_lgan_edges = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto encoding = engine.encode_batch({});

   EXPECT_EQ(encoding.num_graphs, 0);
   EXPECT_EQ(
      std::get< std::vector< std::string > >(
         encoding.graph_attrs.at(std::string(kRelationNamesAttr))
      ),
      engine.get_relation_names()
   );
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(
         encoding.graph_attrs.at(std::string(kRelationAritiesAttr))
      ),
      engine.get_relation_arities()
   );
   EXPECT_TRUE(encoding.graph_fields.contains(std::string(kRelationCountsField)));
}

TEST(SemanticFlatHorizonEncoderEngineTest, ParityMatrixUsesCompiledPlanAcrossHorizonPolicies)
{
   std::vector< SemanticFlatHorizonEncoderEngine::Config > configs;
   configs.emplace_back();
   configs.back().root_policy = RootPolicy::include;
   configs.back().support_literals = true;
   configs.back().include_lgan_edges = true;
   configs.back().enable_parent_relation = true;
   configs.back().enable_sibling_relation = true;
   configs.back().enable_cousin_relation = true;
   configs.back().use_predicate_virtual_nodes = true;
   configs.back().goal_derivations = {
      GoalDerivation::plain,
      GoalDerivation::satisfied,
      GoalDerivation::unsatisfied,
      GoalDerivation::added_satisfied,
      GoalDerivation::added_unsatisfied,
   };

   configs.emplace_back();
   configs.back().root_policy = RootPolicy::encode_only;
   configs.back().pack_relation_args_relation_major = true;

   configs.emplace_back();
   configs.back().root_policy = RootPolicy::exclude;
   configs.back().export_node_names = false;

   configs.emplace_back();
   configs.back().transition_mode = SemanticHorizonMode::delta;
   configs.back().root_policy = RootPolicy::exclude;
   configs.back().pack_relation_args_relation_major = true;

   configs.emplace_back();
   configs.back().transition_mode = SemanticHorizonMode::action;
   configs.back().ignore_actions = false;
   configs.back().root_policy = RootPolicy::include;

   for(const auto& config : configs) {
      SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);
      const auto encoding = engine.encode_batch({make_dag(), make_dag()});

      EXPECT_EQ(encoding.num_graphs, 2);
   }
}

TEST(SemanticFlatHorizonEncoderEngineTest, TopologyRelationsUseConfiguredNamesVerbatim)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.enable_parent_relation = true;
   config.enable_sibling_relation = true;
   config.enable_cousin_relation = true;
   config.parent_relation = "_custom_parent_";
   config.sibling_relation = "_custom_sibling_";
   config.cousin_relation = "_custom_cousin_";
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto& names = engine.get_relation_names();
   EXPECT_TRUE(contains(names, "_custom_parent_"));
   EXPECT_TRUE(contains(names, "_custom_sibling_"));
   EXPECT_TRUE(contains(names, "_custom_cousin_"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, RootPolicyIncludeSkipsSplitStateAnchoring)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.root_policy = RootPolicy::include;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto& names = engine.get_relation_names();
   EXPECT_TRUE(contains(names, "at"));
   // No split candidate relation should exist when the root isn't excluded.
   EXPECT_FALSE(contains(names, "at[state]"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, GoalSatisfactionDerivationsRegisterRootAndAnchoredForms)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.goal_derivations = {
      GoalDerivation::plain, GoalDerivation::satisfied, GoalDerivation::unsatisfied
   };
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);

   const auto& names = engine.get_relation_names();
   EXPECT_TRUE(contains(names, "[+]at[g][sat]"));
   EXPECT_TRUE(contains(names, "[+]at[g][sat][state]"));
   EXPECT_TRUE(contains(names, "[+]at[g][unsat]"));

   const auto encoding = engine.encode(make_dag());
   EXPECT_GT(encoding.num_graphs, 0);
}

TEST(SemanticFlatHorizonEncoderEngineTest, BuilderPathMatchesOneShotComposition)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.support_literals = true;
   config.include_lgan_edges = true;
   config.enable_parent_relation = true;
   config.enable_sibling_relation = true;
   config.enable_cousin_relation = true;
   config.use_predicate_virtual_nodes = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);
   const auto dag = make_dag();

   const auto expected = engine.encode(dag);
   BatchBuilder builder;
   engine.encode(dag, builder);
   builder.next_graph();
   auto actual = builder.build();
   engine.finalize_batch_encoding(actual);

   const auto parity = compare_flat_batch_encodings(expected, actual);
   ASSERT_TRUE(parity.equal) << parity.mismatch;
   EXPECT_EQ(
      std::get< std::string >(actual.graph_attrs.at(std::string(kParentRelationAttr))),
      config.parent_relation
   );
   EXPECT_EQ(
      std::get< std::vector< std::string > >(actual.graph_attrs.at(std::string(kTargetGroupsAttr))),
      (std::vector< std::string >{"state"})
   );
}

TEST(SemanticFlatHorizonEncoderEngineTest, CompiledPlanIsSafeForConcurrentEncodes)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.transition_mode = SemanticHorizonMode::delta;
   config.include_lgan_edges = true;
   config.enable_parent_relation = true;
   config.use_predicate_virtual_nodes = true;
   SemanticFlatHorizonEncoderEngine engine(predicates(), actions(), config);
   const auto dag = make_dag();
   const auto expected = engine.encode(dag);

   std::vector< std::future< BatchBuilder::BatchEncoding > > jobs;
   for(size_t index = 0; index < 8; ++index) {
      jobs.push_back(std::async(std::launch::async, [&engine, &dag] {
         return engine.encode(dag);
      }));
   }
   for(auto& job : jobs) {
      const auto parity = compare_flat_batch_encodings(expected, job.get());
      ASSERT_TRUE(parity.equal) << parity.mismatch;
   }
}

TEST(SemanticFlatHorizonEncoderEngineTest, PublicAssemblyExtendsCanonicalOnePassEncoding)
{
   SemanticFlatHorizonEncoderEngine::Config config;
   config.pack_relation_args_relation_major = true;
   SemanticFlatHorizonAssemblyBuilder builder(predicates(), actions(), config);
   builder.emplace_component< HorizonExtensionComponent >();
   auto compiled = std::move(builder).compile();
   SemanticFlatHorizonEncoderEngine engine(std::move(compiled));
   SemanticFlatHorizonEncoderEngine moved_again(predicates(), actions());
   moved_again = std::move(engine);
   const auto dag = make_dag();

   std::vector< SemanticFlatHorizonInput > inputs;
   inputs.push_back(extension_input(dag, 7, {"left_aux", "right_aux"}));
   inputs.push_back(extension_input(dag, 11, {"next_left_aux", "next_right_aux"}));
   const auto encoding = moved_again.encode_batch(std::span{inputs});

   EXPECT_EQ(encoding.num_graphs, 2);
   EXPECT_EQ(encoding.node_counts.at("sdk_node"), 4);
   EXPECT_EQ(
      encoding.node_names.at("sdk_node"),
      (std::vector< std::string >{"left_aux", "right_aux", "next_left_aux", "next_right_aux"})
   );
   EXPECT_EQ(
      std::get< std::vector< int64_t > >(encoding.graph_fields.at("sdk_marker").values),
      (std::vector< int64_t >{7, 11})
   );
   EXPECT_EQ(
      std::get< std::string >(encoding.graph_attrs.at("sdk_metadata")), "canonical-plus-extension"
   );
   ASSERT_TRUE(encoding.columns.contains("sdk_node/x"));
   EXPECT_EQ(
      std::get< std::vector< float > >(encoding.columns.at("sdk_node/x").data),
      (std::vector< float >{7.0F, 0.0F, 7.0F, 1.0F, 11.0F, 0.0F, 11.0F, 1.0F})
   );

   const auto relation = std::ranges::find(moved_again.get_relation_names(), "sdk_transition");
   ASSERT_NE(relation, moved_again.get_relation_names().end());
   EXPECT_FALSE(contains(moved_again.get_relation_names(), "sdk_transition_alias"));
   const auto relation_id = static_cast< size_t >(
      std::distance(moved_again.get_relation_names().begin(), relation)
   );
   const auto& counts = std::get< std::vector< int64_t > >(
      encoding.graph_fields.at(std::string(kRelationCountsField)).values
   );
   const auto relation_count = moved_again.get_relation_names().size();
   ASSERT_EQ(counts.size(), relation_count * 2);
   EXPECT_EQ(counts[relation_id], 3);
   EXPECT_EQ(counts[relation_count + relation_id], 3);

   const auto& args = std::get< std::vector< int64_t > >(
      encoding.graph_fields.at(std::string(kRelationArgsField)).values
   );
   size_t relation_offset = 0;
   for(size_t id = 0; id < relation_id; ++id) {
      relation_offset += static_cast< size_t >(counts[id] + counts[relation_count + id])
                         * static_cast< size_t >(moved_again.get_relation_arities()[id]);
   }
   ASSERT_GE(args.size(), relation_offset + 6);
   EXPECT_EQ(
      (std::vector< int64_t >(
         args.begin() + static_cast< std::ptrdiff_t >(relation_offset),
         args.begin() + static_cast< std::ptrdiff_t >(relation_offset + 6)
      )),
      (std::vector< int64_t >{3, 4, 5, 10, 11, 12})
   );
   EXPECT_EQ(
      std::get< std::string >(encoding.graph_attrs.at(std::string(kRelationArgsLayoutAttr))),
      std::string(kRelationArgsRelationMajorLayout)
   );

   const auto expected = moved_again.encode(inputs.front());
   std::vector< std::future< BatchBuilder::BatchEncoding > > jobs;
   for(size_t index = 0; index < 4; ++index) {
      jobs.push_back(std::async(std::launch::async, [&moved_again, &input = inputs.front()] {
         return moved_again.encode(input);
      }));
   }
   for(auto& job : jobs) {
      const auto parity = compare_flat_batch_encodings(expected, job.get());
      ASSERT_TRUE(parity.equal) << parity.mismatch;
   }
}

}  // namespace
}  // namespace mifrost
