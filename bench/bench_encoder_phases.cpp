/**
 * @file bench_encoder_phases.cpp
 * @brief Phase-separated benchmark for the borrowed-View encoder boundary.
 *
 * The other benchmarks measure whole encodes, which is enough to catch a gross
 * slowdown but not enough to show *where* the time goes -- and in particular
 * not enough to show whether an input lane is being materialized on the way
 * into the canonical algorithm. This one splits the direct Pymimir path into
 * the five stages the architecture describes and reports them separately:
 *
 *   1. backend View adaptation   (problem -> task context + compact ID tables)
 *   2. atom/action interning     (borrowed Views -> compact graph pools)
 *   3. family graph preparation  (pools -> entity/target/relation working state)
 *   4. graph emission            (working state -> BatchBuilder)
 *   5. batch finalization        (BatchBuilder -> BatchEncoding)
 *
 * Stages 2 and 3 are the ones a materialization regression would show up in:
 * if a lane starts being copied into an owning mirror, interning grows with
 * total occurrences rather than with unique identities.
 *
 * Every benchmark also reports counters -- unique atoms and actions, input
 * occurrences, and the resulting graph size -- so a timing change can be read
 * against the work actually being done rather than guessed at.
 *
 * Sizing is deliberately larger than the smoke benchmarks (see --repeat_state
 * and --duplicate_actions) so that quadratic interning, repeated literal
 * conversion, and duplicate action handling are all visible.
 */
#include <benchmark/benchmark.h>

#include <argparse/argparse.hpp>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <span>
#include <string>
#include <vector>

#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/semantic/views.hpp"
#include "mifrost/core/views/semantic_preparation.hpp"

namespace {

struct BenchConfig {
   std::string data_dir;
   std::string domain;
   std::string problem;
   int batch_size = 32;
   /** Repeat the state's action list to expose duplicate-occurrence handling. */
   int duplicate_actions = 4;
};

BenchConfig g_config;
std::vector< std::string > g_unparsed_args;

std::string default_data_dir()
{
   const char* env = std::getenv("MIFROST_DATA_DIR");
   if(env and *env) {
      return std::string(env);
   }
#ifdef MIFROST_DATA_DIR
   return std::string(MIFROST_DATA_DIR);
#else
   return std::string(".");
#endif
}

void parse_args(int& argc, char** argv)
{
   g_config.data_dir = default_data_dir();
   g_config.domain =
#ifdef MIFROST_BENCH_DEFAULT_DOMAIN
      std::string(MIFROST_BENCH_DEFAULT_DOMAIN);
#else
      std::string("blocks");
#endif
   g_config.problem =
#ifdef MIFROST_BENCH_DEFAULT_PROBLEM
      std::string(MIFROST_BENCH_DEFAULT_PROBLEM);
#else
      std::string("smedium");
#endif

   argparse::ArgumentParser parser("mifrost_bench_encoder_phases");
   parser.add_argument("--data_dir").default_value(g_config.data_dir);
   parser.add_argument("--domain").default_value(g_config.domain);
   parser.add_argument("--problem").default_value(g_config.problem);
   parser.add_argument("--batch_size").default_value(g_config.batch_size).scan< 'i', int >();
   parser.add_argument("--duplicate_actions")
      .default_value(g_config.duplicate_actions)
      .scan< 'i', int >();

   std::vector< std::string > unparsed;
   try {
      unparsed = parser.parse_known_args(argc, argv);
   } catch(const std::exception& ex) {
      std::cerr << ex.what() << "\n" << parser << "\n";
      std::exit(1);
   }

   g_config.data_dir = parser.get< std::string >("--data_dir");
   g_config.domain = parser.get< std::string >("--domain");
   g_config.problem = parser.get< std::string >("--problem");
   g_config.batch_size = parser.get< int >("--batch_size");
   g_config.duplicate_actions = parser.get< int >("--duplicate_actions");

   g_unparsed_args = std::move(unparsed);
   std::vector< char* > keep;
   keep.reserve(g_unparsed_args.size() + 1);
   keep.push_back(argv[0]);
   for(auto& arg : g_unparsed_args) {
      keep.push_back(const_cast< char* >(arg.c_str()));
   }
   for(size_t idx = 0; idx < keep.size(); ++idx) {
      argv[idx] = keep[idx];
   }
   argc = static_cast< int >(keep.size());
   argv[argc] = nullptr;
}

struct PhaseContext {
   mimir::formalism::Problem problem;
   mimir::search::State root;
   std::vector< mimir::formalism::GroundAction > actions;

   static mimir::search::State make_root(const mimir::formalism::Problem& problem)
   {
      mimir::search::LiftedGrounder grounder(problem);
      auto grounded = grounder.create_grounded_axiom_evaluator();
      auto axiom_eval = std::static_pointer_cast< mimir::search::IAxiomEvaluator >(
         std::move(grounded)
      );
      auto repo = mimir::search::StateRepositoryImpl::create(axiom_eval);
      return repo->get_or_create_initial_state().first;
   }

   explicit PhaseContext(const BenchConfig& cfg)
       : problem(
            mimir::formalism::ProblemImpl::create(
               std::filesystem::path(cfg.data_dir) / "pddl" / cfg.domain / "domain.pddl",
               std::filesystem::path(cfg.data_dir) / "pddl" / cfg.domain / (cfg.problem + ".pddl")
            )
         ),
         root(make_root(problem))
   {
   }
};

PhaseContext& context()
{
   static PhaseContext ctx(g_config);
   return ctx;
}

/** Report the work actually performed so timings can be read against it. */
void set_counters(
   benchmark::State& state,
   const mifrost::canonical::detail::ViewPreparation& prepared
)
{
   state.counters["unique_atoms"] = static_cast< double >(prepared.atom_pool.size());
   state.counters["state_facts"] = static_cast< double >(prepared.state_facts.size());
   state.counters["unique_actions"] = static_cast< double >(prepared.action_pool.size());
   state.counters["action_occurrences"] = static_cast< double >(
      prepared.action_occurrence_indices.size()
   );
   state.counters["goal_occurrences"] = static_cast< double >(prepared.goal_level_refs.size());
   state.counters["fact_membership"] = static_cast< double >(prepared.fact_lookup.size());
}

// --- Phase 1: backend View adaptation -------------------------------------

void BM_Phase1_ViewAdaptation(benchmark::State& state)
{
   auto& ctx = context();
   for(auto _ : state) {
      mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
      benchmark::DoNotOptimize(adapter.get_task_context().get());
   }
   state.SetLabel("problem -> task context + compact ID tables");
}

// --- Phase 2: interning borrowed Views into the compact pools --------------

void BM_Phase2_Interning(benchmark::State& state)
{
   auto& ctx = context();
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto task_context = adapter.get_task_context();
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{ctx.actions}
   );

   mifrost::canonical::detail::ViewPreparation last;
   for(auto _ : state) {
      auto prepared = mifrost::canonical::detail::make_flat_view_preparation(
         task_context, state_view, action_views
      );
      benchmark::DoNotOptimize(prepared.atom_pool.size());
      last = std::move(prepared);
   }
   set_counters(state, last);
   state.SetLabel("borrowed Views -> compact graph pools");
}

/**
 * The successor lane reads only objects and state facts. Comparing this against
 * BM_Phase2_Interning shows the cost of the lanes a path does not consume.
 */
void BM_Phase2_InterningStateOnly(benchmark::State& state)
{
   auto& ctx = context();
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto task_context = adapter.get_task_context();
   const auto state_view = adapter.make_state_view(ctx.root);

   mifrost::canonical::detail::ViewPreparation last;
   for(auto _ : state) {
      auto prepared = mifrost::canonical::detail::make_state_only_view_preparation(
         task_context, state_view
      );
      benchmark::DoNotOptimize(prepared.state_facts.size());
      last = std::move(prepared);
   }
   set_counters(state, last);
   state.SetLabel("state-only lane");
}

// --- Phases 3+4: family preparation and emission ---------------------------

void BM_Phase34_FlatPrepareAndEmit(benchmark::State& state)
{
   auto& ctx = context();
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto task_context = adapter.get_task_context();
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{ctx.actions}
   );
   const mifrost::SemanticFlatRelationEncoderEngine engine(task_context);
   const auto prepared = mifrost::canonical::detail::make_flat_view_preparation(
      task_context, state_view, action_views
   );

   int64_t nodes = 0;
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      engine.encode(state_view, action_views, builder);
      nodes = static_cast< int64_t >(builder.current_node_counts.size());
      benchmark::DoNotOptimize(nodes);
   }
   set_counters(state, prepared);
   state.counters["node_types"] = static_cast< double >(nodes);
   state.SetLabel("Flat: pools -> working state -> BatchBuilder");
}

void BM_Phase34_HGraphPrepareAndEmit(benchmark::State& state)
{
   auto& ctx = context();
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto task_context = adapter.get_task_context();
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{ctx.actions}
   );
   const mifrost::SemanticHGraphEncoderEngine engine(task_context);

   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      engine.encode(state_view, action_views, builder);
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
   state.SetLabel("HGraph: pools -> working state -> BatchBuilder");
}

// --- Phase 5: batch collation ---------------------------------------------

void BM_Phase5_BatchFinalization(benchmark::State& state)
{
   auto& ctx = context();
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const auto task_context = adapter.get_task_context();
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{ctx.actions}
   );
   const mifrost::SemanticFlatRelationEncoderEngine engine(task_context);
   const auto graphs = static_cast< size_t >(std::max(g_config.batch_size, 1));

   size_t columns = 0;
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      for(size_t index = 0; index < graphs; ++index) {
         engine.encode(state_view, action_views, builder);
         builder.next_graph();
      }
      auto encoding = builder.build();
      engine.finalize_batch_encoding(encoding);
      columns = encoding.columns.size();
      benchmark::DoNotOptimize(columns);
   }
   state.counters["graphs"] = static_cast< double >(graphs);
   state.counters["columns"] = static_cast< double >(columns);
   state.SetLabel("N graphs -> BatchEncoding");
}

// --- Direct vs compatibility, same algorithm -------------------------------

/**
 * The compatibility DTO must be *borrowed*, not copied into another carrier.
 * If a regression reintroduces a copy, this diverges from the direct path by
 * roughly the cost of one full lane duplication.
 *
 * This case reuses one already-built input, so it measures the encode alone.
 * It is NOT comparable to `BM_DirectViewEncode`, which adapts a live state on
 * every iteration -- see `BM_CompatibilityAdaptAndEncode` for the pair that is.
 */
void BM_CompatibilityInputEncode(benchmark::State& state)
{
   auto& ctx = context();
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context());
   const auto input = adapter.make_input(ctx.root);

   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      engine.encode(input, builder);
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
   state.counters["state_facts"] = static_cast< double >(input.state_facts.size());
   state.SetLabel("owning compatibility DTO, borrowed lanes (encode only)");
}

/**
 * The like-for-like counterpart to `BM_DirectViewEncode`.
 *
 * A caller encoding a live planning state pays for the adaptation too, so the
 * honest comparison between the two routes includes it on both sides. The gap
 * that remains is the direct path's compact-pool preparation (interning, fact
 * membership) against the owned DTO's plain lane vectors -- that is the real
 * price of not materializing an owning semantic mirror, and it is what a
 * backend batch or stream actually pays.
 */
void BM_CompatibilityAdaptAndEncode(benchmark::State& state)
{
   auto& ctx = context();
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context());

   for(auto _ : state) {
      const auto input = adapter.make_input(ctx.root);
      mifrost::BatchBuilder builder;
      engine.encode(input, builder);
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
   state.SetLabel("owning compatibility DTO, adapt + encode");
}

void BM_DirectViewEncode(benchmark::State& state)
{
   auto& ctx = context();
   const mifrost::pymimir::SemanticProblemAdapter adapter(*ctx.problem);
   const mifrost::SemanticFlatRelationEncoderEngine engine(adapter.get_task_context());
   const auto state_view = adapter.make_state_view(ctx.root);
   const auto action_views = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );

   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      engine.encode(state_view, action_views, builder);
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
   state.SetLabel("direct borrowed Views");
}

// --- Synthetic scaling -----------------------------------------------------
//
// The bundled PDDL fixtures are small (blocks/large has ten state facts), which
// is fine for parity but far too small to expose a growth-rate regression.
// These build synthetic inputs of a requested size so quadratic interning,
// repeated literal conversion, and duplicate action handling are all visible in
// the shape of the curve rather than in a single number.

struct SyntheticTask {
   std::shared_ptr< mifrost::SemanticTaskContext > context;
   std::vector< mifrost::SemanticAtom > state_facts;
   std::vector< mifrost::SemanticLiteral > goals;
   std::vector< mifrost::SemanticGroundAction > actions;
};

/**
 * @param size   Number of distinct state facts and goals.
 * @param repeat How many times each action is repeated, to exercise dedup.
 */
SyntheticTask make_synthetic_task(int64_t size, int64_t repeat)
{
   SyntheticTask task;
   task.context = std::make_shared< mifrost::SemanticTaskContext >();
   const auto objects = std::max< int64_t >(size, 1);
   task.context->objects.reserve(static_cast< size_t >(objects));
   for(int64_t index = 0; index < objects; ++index) {
      task.context->objects.push_back("o" + std::to_string(index));
   }
   for(int64_t index = 0; index < 4; ++index) {
      task.context->predicates.push_back(
         mifrost::SemanticPredicateSpec{
            mifrost::SemanticPredicateCategory::fluent, "p" + std::to_string(index), 2
         }
      );
   }
   task.context->actions.push_back(mifrost::SemanticActionSpec{"a0", 2});

   task.state_facts.reserve(static_cast< size_t >(size));
   task.goals.reserve(static_cast< size_t >(size));
   for(int64_t index = 0; index < size; ++index) {
      mifrost::SemanticAtom atom;
      atom.predicate = index % 4;
      atom.arguments.push_back(index % objects);
      atom.arguments.push_back((index * 7 + 1) % objects);
      task.state_facts.push_back(atom);
      task.goals.push_back(mifrost::SemanticLiteral{atom, index % 2 == 0});
   }

   const auto distinct_actions = std::max< int64_t >(size / 8, 1);
   task.actions.reserve(static_cast< size_t >(distinct_actions * std::max< int64_t >(repeat, 1)));
   for(int64_t pass = 0; pass < std::max< int64_t >(repeat, 1); ++pass) {
      for(int64_t index = 0; index < distinct_actions; ++index) {
         mifrost::SemanticGroundAction action;
         action.action = 0;
         action.arguments.push_back(index % objects);
         action.arguments.push_back((index * 3) % objects);
         task.actions.push_back(action);
      }
   }
   return task;
}

/**
 * Interning growth. Linear interning tracks the input size; a linear scan over
 * the pool makes this quadratic, which shows up immediately across the range.
 */
void BM_Scaling_Interning(benchmark::State& state)
{
   const auto size = state.range(0);
   const auto task = make_synthetic_task(size, 4);
   const mifrost::semantic::StateView state_view{
      mifrost::semantic::AtomsView{std::span{task.state_facts}},
      mifrost::semantic::AtomsView{std::span< const mifrost::SemanticAtom >{}},
   };
   const mifrost::semantic::LiteralsView goals{std::span{task.goals}};
   const mifrost::semantic::GroundActionsView actions{std::span{task.actions}};

   mifrost::canonical::detail::ViewPreparation last;
   for(auto _ : state) {
      auto prepared = mifrost::canonical::detail::make_flat_view_preparation(
         task.context, state_view, goals, actions
      );
      benchmark::DoNotOptimize(prepared.atom_pool.size());
      last = std::move(prepared);
   }
   set_counters(state, last);
   state.SetItemsProcessed(state.iterations() * size);
}

/** Whole-encode growth over the same synthetic inputs. */
void BM_Scaling_FlatDirectEncode(benchmark::State& state)
{
   const auto size = state.range(0);
   const auto task = make_synthetic_task(size, 4);
   const mifrost::semantic::StateView state_view{
      mifrost::semantic::AtomsView{std::span{task.state_facts}},
      mifrost::semantic::AtomsView{std::span< const mifrost::SemanticAtom >{}},
   };
   const mifrost::semantic::LiteralsView goals{std::span{task.goals}};
   const mifrost::semantic::GroundActionsView actions{std::span{task.actions}};
   const mifrost::SemanticFlatRelationEncoderEngine engine(task.context);

   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      engine.encode(state_view, goals, actions, builder);
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
   state.SetItemsProcessed(state.iterations() * size);
}

/**
 * The compatibility DTO over the same data. Direct and compatibility should
 * track each other; a persistent gap that grows with size means one of them
 * gained a copy.
 */
void BM_Scaling_FlatCompatibilityEncode(benchmark::State& state)
{
   const auto size = state.range(0);
   const auto task = make_synthetic_task(size, 4);
   mifrost::SemanticFlatRelationInput input;
   input.task_context = task.context;
   input.state_facts = task.state_facts;
   input.goals = task.goals;
   input.actions = task.actions;
   const mifrost::SemanticFlatRelationEncoderEngine engine(task.context);

   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      engine.encode(input, builder);
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
   state.SetItemsProcessed(state.iterations() * size);
}

/** Duplicate action occurrences: dedup cost must track distinct actions. */
void BM_Scaling_DuplicateActions(benchmark::State& state)
{
   const auto repeat = state.range(0);
   const auto task = make_synthetic_task(512, repeat);
   const mifrost::semantic::StateView state_view{
      mifrost::semantic::AtomsView{std::span{task.state_facts}},
      mifrost::semantic::AtomsView{std::span< const mifrost::SemanticAtom >{}},
   };
   const mifrost::semantic::GroundActionsView actions{std::span{task.actions}};

   mifrost::canonical::detail::ViewPreparation last;
   for(auto _ : state) {
      auto prepared = mifrost::canonical::detail::make_flat_view_preparation(
         task.context, state_view, actions
      );
      benchmark::DoNotOptimize(prepared.action_pool.size());
      last = std::move(prepared);
   }
   set_counters(state, last);
}

BENCHMARK(BM_Phase1_ViewAdaptation);
BENCHMARK(BM_Phase2_Interning);
BENCHMARK(BM_Phase2_InterningStateOnly);
BENCHMARK(BM_Phase34_FlatPrepareAndEmit);
BENCHMARK(BM_Phase34_HGraphPrepareAndEmit);
BENCHMARK(BM_Phase5_BatchFinalization);
BENCHMARK(BM_CompatibilityInputEncode);
BENCHMARK(BM_CompatibilityAdaptAndEncode);
BENCHMARK(BM_DirectViewEncode);
BENCHMARK(BM_Scaling_Interning)->RangeMultiplier(4)->Range(64, 16384);
BENCHMARK(BM_Scaling_FlatDirectEncode)->RangeMultiplier(4)->Range(64, 4096);
BENCHMARK(BM_Scaling_FlatCompatibilityEncode)->RangeMultiplier(4)->Range(64, 4096);
BENCHMARK(BM_Scaling_DuplicateActions)->RangeMultiplier(4)->Range(1, 64);

}  // namespace

int main(int argc, char** argv)
{
   parse_args(argc, argv);
   benchmark::Initialize(&argc, argv);
   if(benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1;
   }
   benchmark::RunSpecifiedBenchmarks();
   benchmark::Shutdown();
   return 0;
}
