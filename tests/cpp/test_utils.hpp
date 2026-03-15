#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/formatter.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/goal_inputs.hpp"

namespace mifrost_test {

struct DomainCase {
   const char* domain;
   const char* problem;
};

inline constexpr DomainCase kSmallDomains[] = {
   {"blocks", "smedium"},
   {"gripper", "gripper_b-5"},
   {"spanner", "medium"},
   {"delivery", "instance_2x2_p-2_0"},
};

inline constexpr DomainCase kParityDomains[] = {
   {"blocks", "smedium"},
   {"gripper", "gripper_b-5"},
   {"spanner", "medium"},
   {"delivery", "instance_2x2_p-2_0"},
};

inline std::string sanitize_test_name(std::string name)
{
   for(char& ch : name) {
      if(! std::isalnum(static_cast< unsigned char >(ch))) {
         ch = '_';
      }
   }
   return name;
}

inline std::string case_name(const DomainCase& domain_case)
{
   return sanitize_test_name(
      std::string(domain_case.domain) + "_" + std::string(domain_case.problem)
   );
}

struct Context {
   mimir::formalism::Problem problem;
   mimir::search::StateRepository repo;
   mimir::search::State root;
   mimir::ContinuousCost root_metric = 0.0;
   mimir::formalism::GroundActionList actions;
};

inline std::filesystem::path data_dir()
{
   if(const char* env = std::getenv("MIFROST_DATA_DIR"); env && *env) {
      return std::filesystem::path(env);
   }
#ifdef MIFROST_DATA_DIR
   return std::filesystem::path(MIFROST_DATA_DIR);
#else
   return std::filesystem::current_path() / "data";
#endif
}

inline mimir::formalism::Problem load_problem(const std::string& domain, const std::string& problem)
{
   const auto base = data_dir() / "pddl" / domain;
   const auto domain_path = base / "domain.pddl";
   const auto problem_path = base / (problem + ".pddl");
   return mimir::formalism::ProblemImpl::create(domain_path, problem_path);
}

inline Context make_context(const std::string& domain, const std::string& problem)
{
   auto problem_obj = load_problem(domain, problem);

   mimir::search::LiftedGrounder grounder(problem_obj);
   auto actions = grounder.create_ground_actions();

   auto grounded = grounder.create_grounded_axiom_evaluator();
   auto axiom_eval = std::static_pointer_cast< mimir::search::IAxiomEvaluator >(
      std::move(grounded)
   );
   auto repo = mimir::search::StateRepositoryImpl::create(axiom_eval);

   auto [root_state, metric] = repo->get_or_create_initial_state();
   return Context{problem_obj, repo, root_state, metric, actions};
}

inline mifrost::GoalInputs make_goal_inputs(const mimir::formalism::Problem& problem)
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

inline std::pair< mimir::search::State, mimir::formalism::GroundAction > find_successor(
   const Context& ctx
)
{
   for(const auto& action : ctx.actions) {
      auto [succ_state, _metric] = ctx.repo->get_or_create_successor_state(
         ctx.root, action, ctx.root_metric
      );
      if(succ_state.get_index() != ctx.root.get_index()) {
         return {succ_state, action};
      }
   }
   throw std::runtime_error("No applicable successor action found in test domain.");
}

using IndexMap = std::unordered_map< std::string, std::unordered_map< std::string, int64_t > >;
using EdgePairs = std::vector< std::pair< int64_t, int64_t > >;

inline IndexMap build_index_map(const mifrost::BatchBuilder& builder)
{
   IndexMap index_map;
   for(const auto& [node_type, names] : builder.node_names) {
      auto& map = index_map[node_type];
      for(size_t i = 0; i < names.size(); ++i) {
         map.emplace(names[i], static_cast< int64_t >(i));
      }
   }
   return index_map;
}

inline EdgePairs edge_pairs_for(const mifrost::BatchBuilder& builder, const std::string& edge_type)
{
   const std::string src_key = edge_type + "/edge_index_0";
   const std::string dst_key = edge_type + "/edge_index_1";
   auto src_it = builder.columns.find(src_key);
   auto dst_it = builder.columns.find(dst_key);
   if(src_it == builder.columns.end() || dst_it == builder.columns.end()) {
      return {};
   }
   const auto& src_col = std::get< mifrost::BatchBuilder::LongCol >(src_it->second.data);
   const auto& dst_col = std::get< mifrost::BatchBuilder::LongCol >(dst_it->second.data);
   EdgePairs pairs;
   pairs.reserve(src_col.size());
   for(size_t i = 0; i < src_col.size(); ++i) {
      pairs.emplace_back(src_col[i], dst_col[i]);
   }
   return pairs;
}

inline void sort_edge_pairs(EdgePairs& pairs)
{
   std::sort(pairs.begin(), pairs.end());
}

inline std::vector< std::string > materialize_target_name_states(
   std::span< const mimir::search::State > states
)
{
   std::vector< std::string > names;
   names.reserve(states.size());
   for(const auto& state : states) {
      std::ostringstream stream;
      stream << state;
      names.push_back(stream.str());
   }
   return names;
}

template < typename AttrMap >
inline auto graph_attrs_without_target_names(const AttrMap& graph_attrs)
{
   auto filtered = graph_attrs;
   filtered.erase("target_names");
   return filtered;
}

template < typename EncodingLike >
inline std::vector< std::string > target_names_for(const EncodingLike& encoding)
{
   std::vector< std::string > out;
   if(const auto it = encoding.graph_attrs.find("target_names"); it != encoding.graph_attrs.end()) {
      const auto* names = std::get_if< std::vector< std::string > >(&it->second);
      if(names == nullptr) {
         throw std::runtime_error("target_names graph attr must be a string vector");
      }
      out = *names;
   }
   if(not encoding.lazy_target_name_states.empty()) {
      auto lazy_names = materialize_target_name_states(std::span(encoding.lazy_target_name_states));
      out.insert(
         out.end(),
         std::make_move_iterator(lazy_names.begin()),
         std::make_move_iterator(lazy_names.end())
      );
   }
   return out;
}

}  // namespace mifrost_test
