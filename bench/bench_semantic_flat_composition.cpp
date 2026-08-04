#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

namespace {

mifrost::SemanticFlatRelationInput make_input()
{
   mifrost::SemanticFlatRelationInput input;
   input.objects = {"a", "b", "c", "d", "e", "f", "g", "h"};
   input.state_facts = {
      {0, {0}},
      {0, {1}},
      {1, {2}},
      {1, {3}},
      {2, {4}},
      {2, {5}},
   };
   input.goals = {
      {mifrost::SemanticAtom{0, {6}}, true},
      {mifrost::SemanticAtom{2, {7}}, false},
   };
   input.subgoal_layers = {
      {{mifrost::SemanticAtom{1, {0}}, true}},
      {{mifrost::SemanticAtom{2, {1}}, true}},
   };
   input.actions = {
      {0, {0, 1}},
      {0, {0, 1}},
      {0, {2, 3}},
   };
   input.history = {
      {-1, {{mifrost::SemanticAtom{1, {4}}, true}}},
      {-2, {{mifrost::SemanticAtom{2, {5}}, false}}},
   };
   input.history_max_steps = 2;
   return input;
}

mifrost::SemanticFlatRelationEncoderEngine make_engine()
{
   mifrost::SemanticFlatRelationEncoderEngine::Config config;
   config.max_goal_level = 2;
   config.support_literals = true;
   config.include_lgan_edges = true;
   config.target_sources = {
      mifrost::TargetSource::actions,
      mifrost::TargetSource::goals,
      mifrost::TargetSource::subgoals,
      mifrost::TargetSource::history,
   };
   config.lgan_anchor_sources = config.target_sources;
   config.goal_derivations = {
      mifrost::GoalDerivation::plain,
      mifrost::GoalDerivation::satisfied,
      mifrost::GoalDerivation::unsatisfied,
   };
   config.use_predicate_virtual_nodes = true;
   config.pack_relation_args_relation_major = true;
   return mifrost::SemanticFlatRelationEncoderEngine(
      std::vector< mifrost::SemanticPredicateSpec >{
         {mifrost::SemanticPredicateCategory::fluent, "at", 1},
         {mifrost::SemanticPredicateCategory::fluent, "clear", 1},
         {mifrost::SemanticPredicateCategory::derived, "ready", 1},
      },
      std::vector< mifrost::SemanticActionSpec >{{"move", 2}},
      config
   );
}

void BM_SemanticFlatEncodeComposed(benchmark::State& state)
{
   auto engine = make_engine();
   const auto input = make_input();
   for(auto _ : state) {
      const auto encoding = engine.encode(input);
      benchmark::DoNotOptimize(encoding.columns.size());
   }
}

void BM_SemanticFlatEncodeLegacy(benchmark::State& state)
{
   auto engine = make_engine();
   const auto input = make_input();
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      engine.encode(input, builder);
      builder.next_graph();
      auto encoding = builder.build();
      engine.finalize_batch_encoding(encoding);
      benchmark::DoNotOptimize(encoding.columns.size());
   }
}

}  // namespace

BENCHMARK(BM_SemanticFlatEncodeComposed);
BENCHMARK(BM_SemanticFlatEncodeLegacy);

int main(int argc, char** argv)
{
   benchmark::Initialize(&argc, argv);
   if(benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1;
   }
   benchmark::RunSpecifiedBenchmarks();
   return 0;
}
