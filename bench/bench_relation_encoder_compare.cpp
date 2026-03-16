#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <optional>
#include <string>
#include <vector>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/flat_relation_encoder.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/relation_formatter.hpp"

namespace {

struct BenchConfig {
   std::string data_dir;
   std::string domain;
   std::string problem;
   int batch_size = 32;
};

BenchConfig g_config;
std::vector< std::string > g_unparsed_args;

struct RelationSchemaView {
   std::vector< std::string > names;
   std::vector< int64_t > arities;
};

struct ParityCheckResult {
   bool ok = true;
   std::string detail = "ok";
};

struct ParityReport {
   ParityCheckResult shared_schema_names;
   ParityCheckResult shared_schema_arities;
   ParityCheckResult flat_only_schema_diff;
   ParityCheckResult hgraph_only_schema_diff;
   std::string flat_only_schema = "none";
   std::string flat_only_expected_schema = "none";
   std::string flat_only_unexpected_schema = "none";
   std::string flat_only_missing_expected_schema = "none";
   std::string hgraph_only_schema = "none";
   std::string hgraph_only_expected_schema = "none";
   std::string hgraph_only_unexpected_schema = "none";
   std::string hgraph_only_missing_expected_schema = "none";
   ParityCheckResult single_counts;
   ParityCheckResult single_slots;
   ParityCheckResult batch_counts;
   ParityCheckResult batch_slots;
};

mifrost::GoalInputs make_goal_inputs(const mimir::formalism::Problem& problem);

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

   argparse::ArgumentParser parser("mifrost_bench_relation_encoders");
   parser.add_argument("--data_dir").default_value(g_config.data_dir);
   parser.add_argument("--domain").default_value(g_config.domain);
   parser.add_argument("--problem").default_value(g_config.problem);
   parser.add_argument("--batch_size").default_value(g_config.batch_size).scan< 'i', int >();

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

struct BenchContext {
   mimir::formalism::Problem problem;
   mimir::search::State root;
   mifrost::HGraphEncoderEngine hgraph_engine;
   mifrost::FlatRelationEncoderEngine flat_engine;
   mifrost::GoalInputs goals;
   std::vector< mimir::search::State > batch_states;

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
         hgraph_engine(problem->get_domain()),
         flat_engine(problem->get_domain()),
         goals(make_goal_inputs(problem))
   {
      batch_states.assign(static_cast< size_t >(std::max(cfg.batch_size, 1)), root);
   }
};

BenchContext& context()
{
   static BenchContext ctx(g_config);
   return ctx;
}

RelationSchemaView relation_schema_from_dict(const mifrost::RelationDict& relation_dict)
{
   RelationSchemaView view;
   view.names.reserve(relation_dict.arity.size());
   view.arities.reserve(relation_dict.arity.size());
   for(const auto& [name, arity] : relation_dict.arity) {
      view.names.push_back(name);
      view.arities.push_back(arity);
   }
   return view;
}

mifrost::GoalInputs make_goal_inputs(const mimir::formalism::Problem& problem)
{
   mifrost::GoalInputs inputs;
   for(const auto& goal : problem->get_goal_literals< mimir::formalism::StaticTag >()) {
      inputs.static_goals.emplace_back(goal);
      inputs.static_goal_levels[goal] = 0;
   }
   for(const auto& goal : problem->get_goal_literals< mimir::formalism::FluentTag >()) {
      inputs.fluent_goals.emplace_back(goal);
      inputs.fluent_goal_levels[goal] = 0;
   }
   for(const auto& goal : problem->get_goal_literals< mimir::formalism::DerivedTag >()) {
      inputs.derived_goals.emplace_back(goal);
      inputs.derived_goal_levels[goal] = 0;
   }
   return inputs;
}

std::vector< int64_t > flat_relation_counts_total(
   const mifrost::BatchBuilder::BatchEncoding& encoding,
   size_t relation_count
)
{
   const auto field_it = encoding.graph_fields.find("relation_counts");
   if(field_it == encoding.graph_fields.end()) {
      return std::vector< int64_t >(relation_count, 0);
   }
   const auto& values = std::get< std::vector< int64_t > >(field_it->second.values);
   std::vector< int64_t > totals(relation_count, 0);
   if(relation_count == 0 || values.empty()) {
      return totals;
   }
   for(size_t idx = 0; idx < values.size(); ++idx) {
      totals[idx % relation_count] += values[idx];
   }
   return totals;
}

std::vector< int64_t > hgraph_relation_counts_total(
   const mifrost::BatchBuilder::BatchEncoding& encoding,
   const std::vector< std::string >& relation_names
)
{
   std::vector< int64_t > totals;
   totals.reserve(relation_names.size());
   for(const auto& relation_name : relation_names) {
      const auto it = encoding.node_counts.find(relation_name);
      totals.push_back(it == encoding.node_counts.end() ? 0 : it->second);
   }
   return totals;
}

std::vector< std::string > filter_names_in_order(
   const std::vector< std::string >& names,
   const mifrost::hash_set< std::string >& keep
)
{
   std::vector< std::string > out;
   out.reserve(names.size());
   for(const auto& name : names) {
      if(keep.contains(name)) {
         out.push_back(name);
      }
   }
   return out;
}

std::vector< int64_t > arities_for_names(
   const std::vector< std::string >& names,
   const std::vector< std::string >& ordered_schema_names,
   const std::vector< int64_t >& ordered_schema_arities
)
{
   mifrost::hash_map< std::string, int64_t > arity_by_name;
   arity_by_name.reserve(ordered_schema_names.size());
   for(size_t idx = 0; idx < ordered_schema_names.size(); ++idx) {
      arity_by_name.emplace(ordered_schema_names[idx], ordered_schema_arities[idx]);
   }

   std::vector< int64_t > out;
   out.reserve(names.size());
   for(const auto& name : names) {
      out.push_back(arity_by_name.at(name));
   }
   return out;
}

std::string summarize_name_diff(const std::vector< std::string >& names)
{
   if(names.empty()) {
      return "none";
   }
   constexpr size_t kMaxShown = 6;
   std::vector< std::string > shown;
   shown.reserve(std::min(kMaxShown, names.size()));
   for(size_t idx = 0; idx < names.size() && idx < kMaxShown; ++idx) {
      shown.push_back(names[idx]);
   }
   if(names.size() > kMaxShown) {
      return fmt::format("{} (+{} more)", fmt::join(shown, ","), names.size() - kMaxShown);
   }
   return fmt::format("{}", fmt::join(shown, ","));
}

template < typename Predicates >
void append_history_relation_names(std::vector< std::string >& out, const Predicates& predicates)
{
   for(const auto& predicate : predicates) {
      if(predicate->get_arity() == 0) {
         continue;
      }
      out.push_back(
         mifrost::RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, true, "[hist]"
         )
      );
      out.push_back(
         mifrost::RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, false, "[hist]"
         )
      );
   }
}

std::vector< std::string > expected_flat_only_schema(const mimir::formalism::Domain& domain)
{
   std::vector< std::string > names;
   names.reserve(domain->get_actions().size() + 32);
   for(const auto& action : domain->get_actions()) {
      names.push_back(mifrost::RelationFormatter::format_action_schema(*action));
   }
   append_history_relation_names(names, domain->get_predicates< mimir::formalism::StaticTag >());
   append_history_relation_names(names, domain->get_predicates< mimir::formalism::FluentTag >());
   append_history_relation_names(names, domain->get_predicates< mimir::formalism::DerivedTag >());
   std::ranges::sort(names);
   names.erase(std::ranges::unique(names).begin(), names.end());
   return names;
}

template < typename Predicates >
void append_zero_arity_relation_names(
   std::vector< std::string >& out,
   const Predicates& predicates,
   const mifrost::RelationDict& relation_dict
)
{
   for(const auto& predicate : predicates) {
      if(predicate->get_arity() != 0) {
         continue;
      }
      out.push_back(mifrost::RelationFormatter::format_predicate(predicate));
      for(int level = 0; level <= relation_dict.max_goal_level; ++level) {
         const mifrost::GoalLevel goal_level(level);
         for(bool polarity : {true, false}) {
            out.push_back(
               mifrost::RelationFormatter::format_predicate(
                  predicate, goal_level, std::nullopt, polarity
               )
            );
            for(const auto derivation :
                mifrost::goal_satisfaction_derivations(relation_dict.goal_derivations)) {
               out.push_back(
                  mifrost::RelationFormatter::format_predicate(
                     predicate, goal_level, derivation, polarity
                  )
               );
            }
         }
      }
      if(not relation_dict.support_literals) {
         continue;
      }
      for(bool polarity : {true, false}) {
         out.push_back(
            mifrost::RelationFormatter::format_predicate(
               predicate, std::nullopt, std::nullopt, polarity
            )
         );
         for(const auto derivation :
             mifrost::goal_satisfaction_derivations(relation_dict.goal_derivations)) {
            out.push_back(
               mifrost::RelationFormatter::format_predicate(
                  predicate, std::nullopt, derivation, polarity
               )
            );
         }
      }
   }
}

std::vector< std::string > expected_hgraph_only_schema(
   const mimir::formalism::Domain& domain,
   const mifrost::RelationDict& relation_dict
)
{
   std::vector< std::string > names;
   append_zero_arity_relation_names(
      names, domain->get_predicates< mimir::formalism::StaticTag >(), relation_dict
   );
   append_zero_arity_relation_names(
      names, domain->get_predicates< mimir::formalism::FluentTag >(), relation_dict
   );
   append_zero_arity_relation_names(
      names, domain->get_predicates< mimir::formalism::DerivedTag >(), relation_dict
   );
   std::ranges::sort(names);
   names.erase(std::ranges::unique(names).begin(), names.end());
   return names;
}

ParityCheckResult compare_name_diff(
   std::string_view label,
   const std::vector< std::string >& actual_names,
   const std::vector< std::string >& expected_names
)
{
   mifrost::hash_set< std::string > expected_set;
   expected_set.reserve(expected_names.size());
   for(const auto& name : expected_names) {
      expected_set.insert(name);
   }
   mifrost::hash_set< std::string > actual_set;
   actual_set.reserve(actual_names.size());
   for(const auto& name : actual_names) {
      actual_set.insert(name);
   }

   std::vector< std::string > unexpected;
   for(const auto& name : actual_names) {
      if(not expected_set.contains(name)) {
         unexpected.push_back(name);
      }
   }

   std::vector< std::string > missing;
   for(const auto& name : expected_names) {
      if(not actual_set.contains(name)) {
         missing.push_back(name);
      }
   }

   if(unexpected.empty() and missing.empty()) {
      return {.ok = true, .detail = "ok"};
   }
   return {
      .ok = false,
      .detail = fmt::format(
         "{} unexpected={} missing={}",
         label,
         summarize_name_diff(unexpected),
         summarize_name_diff(missing)
      ),
   };
}

std::vector< int64_t >
slot_counts(std::span< const int64_t > relation_counts, std::span< const int64_t > relation_arities)
{
   std::vector< int64_t > slots(relation_counts.size(), 0);
   for(size_t idx = 0; idx < relation_counts.size(); ++idx) {
      slots[idx] = relation_counts[idx] * relation_arities[idx];
   }
   return slots;
}

template < typename T >
ParityCheckResult compare_vectors(
   std::string_view label,
   const std::vector< T >& lhs,
   const std::vector< T >& rhs,
   const std::vector< std::string >& names = {}
)
{
   if(lhs.size() != rhs.size()) {
      return {
         .ok = false,
         .detail = fmt::format("{} size mismatch: {} vs {}", label, lhs.size(), rhs.size()),
      };
   }
   for(size_t idx = 0; idx < lhs.size(); ++idx) {
      if(lhs[idx] == rhs[idx]) {
         continue;
      }
      const std::string relation_name = idx < names.size() ? names[idx] : fmt::format("#{}", idx);
      return {
         .ok = false,
         .detail = fmt::format(
            "{} mismatch at {}: flat={} hetero={}", label, relation_name, lhs[idx], rhs[idx]
         ),
      };
   }
   return {.ok = true, .detail = "ok"};
}

mifrost::BatchBuilder::BatchEncoding encode_hgraph_single(BenchContext& ctx)
{
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("hetero");
   ctx.hgraph_engine.encode(ctx.root, ctx.goals, {}, builder);
   builder.next_graph();
   return builder.build();
}

mifrost::BatchBuilder::BatchEncoding encode_flat_single(BenchContext& ctx)
{
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("homo");
   ctx.flat_engine.encode(
      ctx.root, ctx.goals, std::span< const mimir::formalism::GroundAction >{}, builder
   );
   builder.next_graph();
   return builder.build();
}

mifrost::BatchBuilder::BatchEncoding encode_hgraph_batch(BenchContext& ctx)
{
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("hetero");
   for(const auto& st : ctx.batch_states) {
      ctx.hgraph_engine.encode(st, ctx.goals, {}, builder);
      builder.next_graph();
   }
   return builder.build();
}

mifrost::BatchBuilder::BatchEncoding encode_flat_batch(BenchContext& ctx)
{
   mifrost::BatchBuilder builder;
   builder.set_graph_kind("homo");
   for(const auto& st : ctx.batch_states) {
      ctx.flat_engine.encode(
         st, ctx.goals, std::span< const mimir::formalism::GroundAction >{}, builder
      );
      builder.next_graph();
   }
   return builder.build();
}

ParityReport build_parity_report(BenchContext& ctx)
{
   const auto hgraph_schema = relation_schema_from_dict(ctx.hgraph_engine.get_relation_dict());
   RelationSchemaView flat_schema{
      .names = ctx.flat_engine.get_relation_names(),
      .arities = ctx.flat_engine.get_relation_arities(),
   };
   mifrost::hash_set< std::string > hgraph_name_set;
   hgraph_name_set.reserve(hgraph_schema.names.size());
   for(const auto& name : hgraph_schema.names) {
      hgraph_name_set.insert(name);
   }
   mifrost::hash_set< std::string > flat_name_set;
   flat_name_set.reserve(flat_schema.names.size());
   for(const auto& name : flat_schema.names) {
      flat_name_set.insert(name);
   }

   std::vector< std::string > shared_names;
   shared_names.reserve(std::min(flat_schema.names.size(), hgraph_schema.names.size()));
   for(const auto& name : flat_schema.names) {
      if(hgraph_name_set.contains(name)) {
         shared_names.push_back(name);
      }
   }
   const auto hgraph_shared_names = filter_names_in_order(hgraph_schema.names, flat_name_set);
   const auto flat_shared_arities = arities_for_names(
      shared_names, flat_schema.names, flat_schema.arities
   );
   const auto hgraph_shared_arities = arities_for_names(
      shared_names, hgraph_schema.names, hgraph_schema.arities
   );

   std::vector< std::string > flat_only_names;
   for(const auto& name : flat_schema.names) {
      if(not hgraph_name_set.contains(name)) {
         flat_only_names.push_back(name);
      }
   }
   std::vector< std::string > hgraph_only_names;
   for(const auto& name : hgraph_schema.names) {
      if(not flat_name_set.contains(name)) {
         hgraph_only_names.push_back(name);
      }
   }
   const auto expected_flat_only_names = expected_flat_only_schema(ctx.problem->get_domain());
   const auto expected_hgraph_only_names = expected_hgraph_only_schema(
      ctx.problem->get_domain(), ctx.hgraph_engine.get_relation_dict()
   );

   const auto hgraph_single = encode_hgraph_single(ctx);
   const auto flat_single = encode_flat_single(ctx);
   const auto hgraph_batch = encode_hgraph_batch(ctx);
   const auto flat_batch = encode_flat_batch(ctx);

   const auto flat_single_all_counts = flat_relation_counts_total(
      flat_single, flat_schema.names.size()
   );
   const auto flat_batch_all_counts = flat_relation_counts_total(
      flat_batch, flat_schema.names.size()
   );
   mifrost::hash_map< std::string, int64_t > flat_single_count_by_name;
   mifrost::hash_map< std::string, int64_t > flat_batch_count_by_name;
   flat_single_count_by_name.reserve(flat_schema.names.size());
   flat_batch_count_by_name.reserve(flat_schema.names.size());
   for(size_t idx = 0; idx < flat_schema.names.size(); ++idx) {
      flat_single_count_by_name.emplace(flat_schema.names[idx], flat_single_all_counts[idx]);
      flat_batch_count_by_name.emplace(flat_schema.names[idx], flat_batch_all_counts[idx]);
   }
   std::vector< int64_t > flat_single_counts;
   std::vector< int64_t > flat_batch_counts;
   flat_single_counts.reserve(shared_names.size());
   flat_batch_counts.reserve(shared_names.size());
   for(const auto& name : shared_names) {
      flat_single_counts.push_back(flat_single_count_by_name.at(name));
      flat_batch_counts.push_back(flat_batch_count_by_name.at(name));
   }
   const auto hgraph_single_counts = hgraph_relation_counts_total(hgraph_single, shared_names);
   const auto hgraph_batch_counts = hgraph_relation_counts_total(hgraph_batch, shared_names);

   return ParityReport{
      .shared_schema_names = compare_vectors(
         "shared schema names", shared_names, hgraph_shared_names
      ),
      .shared_schema_arities = compare_vectors(
         "shared schema arities", flat_shared_arities, hgraph_shared_arities, shared_names
      ),
      .flat_only_schema_diff = compare_name_diff(
         "flat-only schema diff", flat_only_names, expected_flat_only_names
      ),
      .hgraph_only_schema_diff = compare_name_diff(
         "hgraph-only schema diff", hgraph_only_names, expected_hgraph_only_names
      ),
      .flat_only_schema = summarize_name_diff(flat_only_names),
      .flat_only_expected_schema = summarize_name_diff(expected_flat_only_names),
      .flat_only_unexpected_schema = summarize_name_diff([&] {
         std::vector< std::string > names;
         for(const auto& name : flat_only_names) {
            if(not std::ranges::contains(expected_flat_only_names, name)) {
               names.push_back(name);
            }
         }
         return names;
      }()),
      .flat_only_missing_expected_schema = summarize_name_diff([&] {
         std::vector< std::string > names;
         for(const auto& name : expected_flat_only_names) {
            if(not std::ranges::contains(flat_only_names, name)) {
               names.push_back(name);
            }
         }
         return names;
      }()),
      .hgraph_only_schema = summarize_name_diff(hgraph_only_names),
      .hgraph_only_expected_schema = summarize_name_diff(expected_hgraph_only_names),
      .hgraph_only_unexpected_schema = summarize_name_diff([&] {
         std::vector< std::string > names;
         for(const auto& name : hgraph_only_names) {
            if(not std::ranges::contains(expected_hgraph_only_names, name)) {
               names.push_back(name);
            }
         }
         return names;
      }()),
      .hgraph_only_missing_expected_schema = summarize_name_diff([&] {
         std::vector< std::string > names;
         for(const auto& name : expected_hgraph_only_names) {
            if(not std::ranges::contains(hgraph_only_names, name)) {
               names.push_back(name);
            }
         }
         return names;
      }()),
      .single_counts = compare_vectors(
         "single counts", flat_single_counts, hgraph_single_counts, shared_names
      ),
      .single_slots = compare_vectors(
         "single slots",
         slot_counts(flat_single_counts, flat_shared_arities),
         slot_counts(hgraph_single_counts, flat_shared_arities),
         shared_names
      ),
      .batch_counts = compare_vectors(
         "batch counts", flat_batch_counts, hgraph_batch_counts, shared_names
      ),
      .batch_slots = compare_vectors(
         "batch slots",
         slot_counts(flat_batch_counts, flat_shared_arities),
         slot_counts(hgraph_batch_counts, flat_shared_arities),
         shared_names
      ),
   };
}

void publish_parity_report(const ParityReport& report)
{
   const auto publish = [](std::string_view key, const ParityCheckResult& result) {
      benchmark::AddCustomContext(std::string(key), result.detail);
   };
   publish("parity_shared_schema_names", report.shared_schema_names);
   publish("parity_shared_schema_arities", report.shared_schema_arities);
   publish("parity_schema_flat_only", report.flat_only_schema_diff);
   publish("parity_schema_hgraph_only", report.hgraph_only_schema_diff);
   benchmark::AddCustomContext("schema_flat_only", report.flat_only_schema);
   benchmark::AddCustomContext("schema_flat_only_expected", report.flat_only_expected_schema);
   benchmark::AddCustomContext("schema_flat_only_unexpected", report.flat_only_unexpected_schema);
   benchmark::AddCustomContext(
      "schema_flat_only_missing_expected", report.flat_only_missing_expected_schema
   );
   benchmark::AddCustomContext("schema_hgraph_only", report.hgraph_only_schema);
   benchmark::AddCustomContext("schema_hgraph_only_expected", report.hgraph_only_expected_schema);
   benchmark::AddCustomContext(
      "schema_hgraph_only_unexpected", report.hgraph_only_unexpected_schema
   );
   benchmark::AddCustomContext(
      "schema_hgraph_only_missing_expected", report.hgraph_only_missing_expected_schema
   );
   publish("parity_single_counts", report.single_counts);
   publish("parity_single_slots", report.single_slots);
   publish("parity_batch_counts", report.batch_counts);
   publish("parity_batch_slots", report.batch_slots);
}

void BM_HGraphEncodeSingle(benchmark::State& state)
{
   auto& ctx = context();
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      ctx.hgraph_engine.encode(ctx.root, ctx.goals, {}, builder);
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
}

void BM_FlatEncodeSingle(benchmark::State& state)
{
   auto& ctx = context();
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      ctx.flat_engine.encode(
         ctx.root, ctx.goals, std::span< const mimir::formalism::GroundAction >{}, builder
      );
      benchmark::DoNotOptimize(builder.current_node_counts.size());
   }
}

void BM_HGraphEncodeBatch(benchmark::State& state)
{
   auto& ctx = context();
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      for(const auto& st : ctx.batch_states) {
         ctx.hgraph_engine.encode(st, ctx.goals, {}, builder);
         builder.next_graph();
      }
      benchmark::DoNotOptimize(builder.current_graph_idx);
   }
}

void BM_FlatEncodeBatch(benchmark::State& state)
{
   auto& ctx = context();
   for(auto _ : state) {
      mifrost::BatchBuilder builder;
      for(const auto& st : ctx.batch_states) {
         ctx.flat_engine.encode(
            st, ctx.goals, std::span< const mimir::formalism::GroundAction >{}, builder
         );
         builder.next_graph();
      }
      benchmark::DoNotOptimize(builder.current_graph_idx);
   }
}

}  // namespace

BENCHMARK(BM_HGraphEncodeSingle);
BENCHMARK(BM_FlatEncodeSingle);
BENCHMARK(BM_HGraphEncodeBatch);
BENCHMARK(BM_FlatEncodeBatch);

int main(int argc, char** argv)
{
   parse_args(argc, argv);
   benchmark::Initialize(&argc, argv);
   if(benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1;
   }
   publish_parity_report(build_parity_report(context()));
   benchmark::RunSpecifiedBenchmarks();
   return 0;
}
