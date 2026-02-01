#include <benchmark/benchmark.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <string>
#include <vector>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"

namespace {

struct BenchConfig {
   std::string data_dir;
   std::string domain;
   std::string problem;
   int batch_size = 32;
   int stream_size = 32;
};

BenchConfig g_config;
std::vector< std::string > g_unparsed_args;

std::string default_data_dir()
{
   const char* env = std::getenv("MIFROST_DATA_DIR");
   if(env && *env) {
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
      std::string("probBLOCKS-4-0");
#endif

   argparse::ArgumentParser parser("mifrost_bench_hgraph");
   parser.add_argument("--data_dir").default_value(g_config.data_dir);
   parser.add_argument("--domain").default_value(g_config.domain);
   parser.add_argument("--problem").default_value(g_config.problem);
   parser.add_argument("--batch_size").default_value(g_config.batch_size).scan< 'i', int >();
   parser.add_argument("--stream_size").default_value(g_config.stream_size).scan< 'i', int >();

   std::vector< std::string > unparsed;
   try {
      unparsed = parser.parse_known_args(argc, argv);
   } catch(const std::exception& ex) {
      std::cerr << ex.what() << "\n";
      std::cerr << parser << "\n";
      std::exit(1);
   }

   g_config.data_dir = parser.get< std::string >("--data_dir");
   g_config.domain = parser.get< std::string >("--domain");
   g_config.problem = parser.get< std::string >("--problem");
   g_config.batch_size = parser.get< int >("--batch_size");
   g_config.stream_size = parser.get< int >("--stream_size");

   g_unparsed_args = std::move(unparsed);
   std::vector< char* > keep;
   keep.reserve(unparsed.size() + 1);
   keep.push_back(argv[0]);
   for(auto& arg : g_unparsed_args) {
      keep.push_back(const_cast< char* >(arg.c_str()));
   }
   for(size_t i = 0; i < keep.size(); ++i) {
      argv[i] = keep[i];
   }
   argc = static_cast< int >(keep.size());
   argv[argc] = nullptr;
}

struct BenchContext {
   mimir::formalism::Problem problem;
   mimir::search::State root;
   mifrost::HGraphEncoderEngine engine;
   std::vector< mimir::search::State > batch_states;
   std::vector< mimir::search::State > stream_states;

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

   explicit BenchContext(const BenchConfig& cfg)
       : problem(
            mimir::formalism::ProblemImpl::create(
               std::filesystem::path(cfg.data_dir) / "pddl" / cfg.domain / "domain.pddl",
               std::filesystem::path(cfg.data_dir) / "pddl" / cfg.domain / (cfg.problem + ".pddl")
            )
         ),
         root(make_root(problem)),
         engine(problem->get_domain())
   {
      batch_states.assign(static_cast< size_t >(std::max(cfg.batch_size, 1)), root);
      stream_states.assign(static_cast< size_t >(std::max(cfg.stream_size, 1)), root);
   }
};

BenchContext& context()
{
   static BenchContext ctx(g_config);
   return ctx;
}

void BM_EncodeSingle(benchmark::State& state)
{
   auto& ctx = context();
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      ctx.engine.encode(ctx.root, builder);
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
}

void BM_EncodeBatch(benchmark::State& state)
{
   auto& ctx = context();
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      for(const auto& st : ctx.batch_states) {
         ctx.engine.encode(st, builder);
         builder.next_graph();
      }
      benchmark::DoNotOptimize(builder.current_graph_idx);
   }
}

void BM_EncodeStream(benchmark::State& state)
{
   auto& ctx = context();
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      for(const auto& st : ctx.stream_states) {
         ctx.engine.encode(st, builder);
         builder.next_graph();
      }
      benchmark::DoNotOptimize(builder.current_graph_idx);
   }
}

}  // namespace

BENCHMARK(BM_EncodeSingle);
BENCHMARK(BM_EncodeBatch);
BENCHMARK(BM_EncodeStream);

int main(int argc, char** argv)
{
   parse_args(argc, argv);
   benchmark::Initialize(&argc, argv);
   if(benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1;
   }
   benchmark::RunSpecifiedBenchmarks();
   return 0;
}
