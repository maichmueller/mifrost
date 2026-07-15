/**
 * @file flat_horizon_encoder.cpp
 * @brief Main implementation file for the flat horizon encoder.
 *
 * This file holds the horizon-specific emit flow on top of the shared flat
 * schema, node-table, and tuple helpers. Root handling differs from the plain
 * flat relation encoder and stays documented here.
 */
#include "flat_horizon_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mimir/formalism/problem.hpp>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>

#include "flat_encoder_common.hpp"
#include "flat_goal_helpers.hpp"
#include "flat_horizon_context.hpp"
#include "flat_lgan.hpp"
#include "flat_tuple_args.hpp"
#include "mifrost/backends/pymimir/deferred_state_names.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

namespace {

constexpr std::string_view kHiddenRootCarrierName = "_root_state_";
constexpr std::string_view kCandidateRelationSuffix = "[state]";

template < typename... Args >
std::string state_anchored_relation_name(const std::string_view name, Args&&... args)
{
   return RelationFormatter::format_predicate(
      name, std::forward< Args >(args)..., kCandidateRelationSuffix
   );
}

bool split_full_state_relations(const FlatHorizonEncoderEngine::Config& config)
{
   return config.transition_mode == FlatHorizonEncoderEngine::Mode::full
          && root_uses_split_state_relations(config.root_policy);
}

template < typename LiteralTag >
uint32_t fact_tag_id()
{
   if constexpr(std::is_same_v< LiteralTag, mimir::formalism::StaticTag >) {
      return 1U;
   }
   if constexpr(std::is_same_v< LiteralTag, mimir::formalism::FluentTag >) {
      return 2U;
   }
   return 3U;
}

uint64_t pack_u32_u32(uint32_t hi, uint32_t lo)
{
   return (static_cast< uint64_t >(hi) << 32) | static_cast< uint64_t >(lo);
}

struct FlatHorizonBatchProfile {
   using Clock = std::chrono::steady_clock;

   bool enabled = false;
   double prepare_builder_s = 0.0;
   double goal_inputs_s = 0.0;
   double default_dag_s = 0.0;
   double encode_impl_s = 0.0;
   double make_context_s = 0.0;
   double root_emit_s = 0.0;
   double root_delta_setup_s = 0.0;
   double candidate_loop_s = 0.0;
   double delta_fallback_s = 0.0;
   double topology_relations_s = 0.0;
   double finalize_builder_s = 0.0;
   double lgan_s = 0.0;
   int64_t graphs = 0;
   int64_t dag_nodes = 0;
   int64_t candidate_nodes = 0;
   int64_t entity_rows = 0;
   int64_t relation_instances = 0;
   int64_t provided_delta_nodes = 0;
   int64_t fallback_delta_nodes = 0;
};

thread_local FlatHorizonBatchProfile* g_flat_horizon_batch_profile = nullptr;

bool flat_horizon_batch_profile_enabled()
{
   static const bool enabled = [] {
      const char* value = std::getenv("MIFROST_PROFILE_FLAT_HORIZON");
      return value != nullptr && std::string_view(value) != "0"
             && std::string_view(value) != "false";
   }();
   return enabled;
}

struct ScopedProfileTimer {
   using Clock = FlatHorizonBatchProfile::Clock;

   double* accum = nullptr;
   Clock::time_point start{};

   explicit ScopedProfileTimer(double* accum_ptr) : accum(accum_ptr)
   {
      if(accum != nullptr) {
         start = Clock::now();
      }
   }

   ~ScopedProfileTimer()
   {
      if(accum != nullptr) {
         *accum += std::chrono::duration< double >(Clock::now() - start).count();
      }
   }
};

void print_flat_horizon_batch_profile(const FlatHorizonBatchProfile& profile)
{
   if(not profile.enabled) {
      return;
   }
   std::fprintf(
      stderr,
      "[mifrost.flat_horizon] graphs=%lld dag_nodes=%lld candidates=%lld "
      "entities=%lld relation_instances=%lld provided_delta=%lld fallback_delta=%lld\n"
      "  prepare_builder=%.6fs goal_inputs=%.6fs default_dag=%.6fs encode_impl=%.6fs\n"
      "  make_context=%.6fs root_emit=%.6fs root_delta_setup=%.6fs candidate_loop=%.6fs\n"
      "  delta_fallback=%.6fs topology_relations=%.6fs finalize_builder=%.6fs lgan=%.6fs\n",
      static_cast< long long >(profile.graphs),
      static_cast< long long >(profile.dag_nodes),
      static_cast< long long >(profile.candidate_nodes),
      static_cast< long long >(profile.entity_rows),
      static_cast< long long >(profile.relation_instances),
      static_cast< long long >(profile.provided_delta_nodes),
      static_cast< long long >(profile.fallback_delta_nodes),
      profile.prepare_builder_s,
      profile.goal_inputs_s,
      profile.default_dag_s,
      profile.encode_impl_s,
      profile.make_context_s,
      profile.root_emit_s,
      profile.root_delta_setup_s,
      profile.candidate_loop_s,
      profile.delta_fallback_s,
      profile.topology_relations_s,
      profile.finalize_builder_s,
      profile.lgan_s
   );
}

FlatHorizonEncoderEngine::Config normalize_config(FlatHorizonEncoderEngine::Config config)
{
   if(config.transition_mode == FlatHorizonEncoderEngine::Mode::delta) {
      config.support_literals = true;
   }
   return config;
}

template < typename AtomTag >
std::vector< int64_t > logical_arg_rows_for_atom(
   const FlatHorizonEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAtom< AtomTag >& atom,
   std::optional< int64_t >
)
{
   return flat_logical_arg_rows_for_atom(
      context, atom, "Flat horizon encoder encountered object not present in entity table: "
   );
}

template < typename AtomTag >
std::vector< int64_t > local_arg_rows_for_atom(
   const FlatHorizonEncoderEngine& engine,
   FlatHorizonEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAtom< AtomTag >& atom,
   std::optional< int64_t > state_entity_index
)
{
   std::vector< int64_t > auxiliary_args;
   if(state_entity_index.has_value()) {
      auxiliary_args.push_back(*state_entity_index);
   }
   return build_flat_atom_tuple_args(
      context,
      atom,
      std::span{auxiliary_args},
      engine.get_config().use_predicate_virtual_nodes,
      "Flat horizon encoder encountered object not present in entity table: "
   );
}

std::vector< int64_t > local_arg_rows_for_action(
   const FlatHorizonEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAction& action,
   std::optional< int64_t > state_entity_index
)
{
   std::vector< int64_t > auxiliary_args;
   if(state_entity_index.has_value()) {
      auxiliary_args.push_back(*state_entity_index);
   }
   return build_flat_action_tuple_args(
      context,
      action,
      std::span{auxiliary_args},
      "Flat horizon encoder encountered action object not present in entity table: "
   );
}

}  // namespace

FlatHorizonEncoderEngine::FlatHorizonEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : FlatHorizonEncoderEngine(domain, Config{})
{
}

FlatHorizonEncoderEngine::FlatHorizonEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : domain_(domain), config_(normalize_config(std::move(config)))
{
   initialize_from_domain();
}

FlatHorizonEncoderEngine::FlatHorizonEncoderEngine(mimir::formalism::Domain domain)
    : FlatHorizonEncoderEngine(std::move(domain), Config{})
{
}

FlatHorizonEncoderEngine::FlatHorizonEncoderEngine(mimir::formalism::Domain domain, Config config)
    : domain_holder_(std::move(domain)),
      domain_(*domain_holder_),
      config_(normalize_config(std::move(config)))
{
   initialize_from_domain();
}

FlatHorizonEncoderEngine::~FlatHorizonEncoderEngine() = default;

void FlatHorizonEncoderEngine::initialize_from_domain()
{
   if(config_.transition_mode == Mode::action and config_.ignore_actions) {
      throw std::invalid_argument("Action flat horizon encoding requires ignore_actions=false.");
   }

   predicate_specs_.clear();
   regular_predicate_specs_.clear();
   action_specs_.clear();

   RelationDictConfig rel_config;
   rel_config.max_goal_level = static_cast< int >(config_.max_goal_level);
   rel_config.support_literals = config_.support_literals;
   rel_config.goal_derivations = config_.goal_derivations;

   std::vector< mimir::formalism::Action > actions;
   if(not config_.ignore_actions) {
      actions.assign(domain_.get_actions().begin(), domain_.get_actions().end());
   }

   const auto top_type_predicates = rel_config.top_type_predicates;
   auto collect_predicates = [&]< typename Tag >(Tag) {
      for(const auto predicate : domain_.get_predicates< Tag >()) {
         PredicateSpec spec{
            .name = RelationFormatter::format_predicate(predicate),
            .arity = static_cast< int >(predicate->get_arity()),
         };
         predicate_specs_.push_back(spec);
         if(not top_type_predicates.contains(predicate->get_name())) {
            regular_predicate_specs_.push_back(spec);
         }
      }
   };
   collect_predicates(mimir::formalism::StaticTag{});
   collect_predicates(mimir::formalism::FluentTag{});
   collect_predicates(mimir::formalism::DerivedTag{});

   auto predicate_order = [](const PredicateSpec& lhs, const PredicateSpec& rhs) {
      return lhs.name < rhs.name;
   };
   std::ranges::sort(predicate_specs_, predicate_order);
   std::ranges::sort(regular_predicate_specs_, predicate_order);

   for(const auto& action : actions) {
      const auto layout = make_nonpredicate_tuple_layout(
         action->get_arity(), {FlatSlotRole::state_slot}
      );
      action_specs_.push_back(
         PredicateSpec{
            .name = RelationFormatter::format_action_schema(*action),
            .arity = layout.encoded_arity(),
         }
      );
   }
   std::ranges::sort(action_specs_, predicate_order);

   target_entity_group_names_ = {std::string(target_source_group_name(TargetSource::states))};
   target_metadata_group_names_ = target_entity_group_names_;

   FlatRelationSchemaRegistry schema_registry;
   const bool root_relations_use_state_slot = root_in_state_relations(config_.root_policy);
   const bool split_candidate_relations = split_full_state_relations(config_);
   auto register_relation =
      [&](const std::string& name, const FlatTupleLayout& layout, const std::string& source) {
         schema_registry.add_or_validate(name, layout, source);
      };
   auto predicate_layout = [&](int logical_arity, bool include_state_slot) {
      return include_state_slot
                ? make_predicate_tuple_layout(
                     logical_arity, {FlatSlotRole::state_slot}, config_.use_predicate_virtual_nodes
                  )
                : make_predicate_tuple_layout(
                     logical_arity, {}, config_.use_predicate_virtual_nodes
                  );
   };
   auto add_root_only_relation =
      [&](const std::string& base_name, int base_arity, const std::string& source) {
         register_relation(
            base_name, predicate_layout(base_arity, root_relations_use_state_slot), source
         );
      };
   auto add_full_state_relation =
      [&](const std::string& base_name, int base_arity, const std::string& source) {
         add_root_only_relation(base_name, base_arity, source);
         if(split_candidate_relations) {
            register_relation(
               state_anchored_relation_name(base_name), predicate_layout(base_arity, true), source
            );
         }
      };
   auto add_delta_candidate_relation =
      [&](const std::string& base_name, int base_arity, const std::string& source) {
         register_relation(base_name, predicate_layout(base_arity, true), source);
      };

   for(const auto& spec : predicate_specs_) {
      add_full_state_relation(spec.name, spec.arity, "state");
   }
   for(const auto& spec : regular_predicate_specs_) {
      if(includes_plain_goal_derivation(config_.goal_derivations)) {
         for(size_t level = 0; level <= config_.max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               add_root_only_relation(
                  RelationFormatter::format_predicate(
                     spec.name, goal_level, std::nullopt, polarity
                  ),
                  spec.arity,
                  "goal"
               );
            }
         }
      }
      if(config_.support_literals) {
         for(bool polarity : {true, false}) {
            const auto relation_name = RelationFormatter::format_predicate(
               spec.name, std::nullopt, std::nullopt, polarity
            );
            if(config_.transition_mode == Mode::delta) {
               add_delta_candidate_relation(relation_name, spec.arity, "state");
            } else {
               add_root_only_relation(relation_name, spec.arity, "state");
            }
         }
      }
      for(const auto derivation : goal_satisfaction_derivations(config_.goal_derivations)) {
         const bool emitted_on_root = derivation == GoalDerivation::satisfied
                                      || derivation == GoalDerivation::unsatisfied;
         const bool emitted_on_delta_candidate = derivation == GoalDerivation::added_satisfied
                                                 || derivation == GoalDerivation::added_unsatisfied;
         const bool emitted_on_full_candidate = derivation == GoalDerivation::satisfied
                                                || derivation == GoalDerivation::unsatisfied;
         if(emitted_on_root) {
            for(size_t level = 0; level <= config_.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(bool polarity : {true, false}) {
                  add_root_only_relation(
                     RelationFormatter::format_predicate(
                        spec.name, goal_level, derivation, polarity
                     ),
                     spec.arity,
                     "goal_satisfaction"
                  );
               }
            }
            if(config_.support_literals) {
               for(bool polarity : {true, false}) {
                  add_root_only_relation(
                     RelationFormatter::format_predicate(
                        spec.name, std::nullopt, derivation, polarity
                     ),
                     spec.arity,
                     "goal_satisfaction"
                  );
               }
            }
         }
         if(config_.transition_mode == Mode::full && emitted_on_full_candidate
            && split_candidate_relations) {
            for(size_t level = 0; level <= config_.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(bool polarity : {true, false}) {
                  register_relation(
                     state_anchored_relation_name(
                        RelationFormatter::format_predicate(
                           spec.name, goal_level, derivation, polarity
                        )
                     ),
                     predicate_layout(spec.arity, true),
                     "goal_satisfaction"
                  );
               }
            }
            if(config_.support_literals) {
               for(bool polarity : {true, false}) {
                  register_relation(
                     state_anchored_relation_name(
                        RelationFormatter::format_predicate(
                           spec.name, std::nullopt, derivation, polarity
                        )
                     ),
                     predicate_layout(spec.arity, true),
                     "goal_satisfaction"
                  );
               }
            }
         }
         if(config_.transition_mode == Mode::delta && emitted_on_delta_candidate) {
            for(size_t level = 0; level <= config_.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(bool polarity : {true, false}) {
                  add_delta_candidate_relation(
                     RelationFormatter::format_predicate(
                        spec.name, goal_level, derivation, polarity
                     ),
                     spec.arity,
                     "goal_satisfaction"
                  );
               }
            }
            if(config_.support_literals) {
               for(bool polarity : {true, false}) {
                  add_delta_candidate_relation(
                     RelationFormatter::format_predicate(
                        spec.name, std::nullopt, derivation, polarity
                     ),
                     spec.arity,
                     "goal_satisfaction"
                  );
               }
            }
         }
      }
   }
   for(const auto& spec : action_specs_) {
      register_relation(
         spec.name,
         make_nonpredicate_tuple_layout(spec.arity - 1, {FlatSlotRole::state_slot}),
         "action"
      );
   }
   if(config_.enable_parent_relation) {
      register_relation(
         config_.parent_relation,
         make_nonpredicate_tuple_layout(0, {FlatSlotRole::state_slot, FlatSlotRole::state_slot}),
         "parent"
      );
   }
   if(config_.enable_sibling_relation) {
      register_relation(
         config_.sibling_relation,
         make_nonpredicate_tuple_layout(0, {FlatSlotRole::state_slot, FlatSlotRole::state_slot}),
         "sibling"
      );
   }
   if(config_.enable_cousin_relation) {
      register_relation(
         config_.cousin_relation,
         make_nonpredicate_tuple_layout(0, {FlatSlotRole::state_slot, FlatSlotRole::state_slot}),
         "cousin"
      );
   }

   const auto metadata = build_flat_relation_schema_metadata(
      schema_registry,
      static_cast< int >(config_.max_goal_level),
      config_.support_literals,
      config_.goal_derivations,
      "FlatHorizonEncoderEngine did not derive any relation types for this domain/config"
   );
   relation_dict_ = metadata.relation_dict;
   relation_names_ = metadata.relation_names;
   relation_arities_ = metadata.relation_arities;
   relation_sources_ = metadata.relation_sources;
   relation_logical_arities_ = metadata.relation_logical_arities;
   relation_encoded_arities_ = metadata.relation_encoded_arities;
   relation_slot_roles_ = metadata.relation_slot_roles;
   relation_slot_role_offsets_ = metadata.relation_slot_role_offsets;
   slot_role_names_ = metadata.slot_role_names;
   relation_name_to_id_ = metadata.relation_name_to_id;
}

void FlatHorizonEncoderEngine::prepare_builder(BatchBuilder& builder) const
{
   const FlatRelationSchemaMetadata metadata{
      .relation_dict = relation_dict_,
      .relation_names = relation_names_,
      .relation_arities = relation_arities_,
      .relation_sources = relation_sources_,
      .relation_logical_arities = relation_logical_arities_,
      .relation_encoded_arities = relation_encoded_arities_,
      .relation_slot_roles = relation_slot_roles_,
      .relation_slot_role_offsets = relation_slot_role_offsets_,
      .slot_role_names = slot_role_names_,
      .relation_name_to_id = relation_name_to_id_,
   };
   set_flat_graph_attrs(
      builder,
      metadata,
      FlatBuilderGraphConfig{
         .include_lgan_edges = config_.include_lgan_edges,
         .use_predicate_virtual_nodes = config_.use_predicate_virtual_nodes,
         .target_symbol_prefix = config_.target_symbol_prefix,
         .target_entity_group_names = target_entity_group_names_,
         .lgan_tn_edge_pos = config_.lgan_tn_edge_pos,
         .lgan_nn_edge_pos = config_.lgan_nn_edge_pos,
         .lgan_rr_edge_pos = config_.lgan_rr_edge_pos,
         .pack_relation_args_relation_major = config_.pack_relation_args_relation_major,
      }
   );

   register_flat_entity_fields(builder);
   register_flat_target_entity_fields(builder);
   builder.register_field(
      std::string(kTargetSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   const TargetMetadataEmitConfig target_emit_config{
      .position_node_type_id = std::string(kFlatEntityNodeType),
      .symbol_prefix = config_.target_symbol_prefix,
      .include_depth = true,
      .include_group = true,
      .include_names = false,
      .groups = target_metadata_group_names_,
      .parent_relation = config_.parent_relation,
   };
   register_target_fields(builder, target_emit_config);
   builder.set_graph_attr(std::string(kTargetGroupsAttr), target_metadata_group_names_);
   builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   builder.set_graph_attr(std::string(kParentRelationAttr), config_.parent_relation);

   register_flat_relation_instance_fields(builder, static_cast< int >(relation_names_.size()));
   if(config_.include_lgan_edges) {
      register_flat_lgan_fields(builder);
   }
}

FlatHorizonEncoderEngine::EncodingContext FlatHorizonEncoderEngine::make_context(
   const mimir::search::State& root,
   const TransitionDAG& dag
) const
{
   return build_flat_horizon_encoding_context(
      root,
      dag,
      FlatHorizonContextBuildConfig{
         .root_policy = config_.root_policy,
         .export_node_names = config_.export_node_names,
         .predicate_symbol_capacity = predicate_specs_.size(),
         .target_symbol_prefix = config_.target_symbol_prefix,
         .hidden_root_carrier_name = std::string(kHiddenRootCarrierName),
      }
   );
}

int FlatHorizonEncoderEngine::relation_id_for(const std::string& name) const
{
   const auto it = relation_name_to_id_.find(name);
   if(it == relation_name_to_id_.end()) {
      throw std::invalid_argument("Unknown flat horizon relation name '" + name + "'");
   }
   return it->second;
}

int64_t FlatHorizonEncoderEngine::state_entity_index_for(
   const EncodingContext& context,
   int64_t node_index
) const
{
   const auto it = context.state_entity_index_by_node_index.find(node_index);
   if(it == context.state_entity_index_by_node_index.end()) {
      throw std::invalid_argument(
         "Flat horizon encoder encountered missing state target entity for node index "
         + std::to_string(node_index)
      );
   }
   return it->second;
}

void FlatHorizonEncoderEngine::encode(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   encode_impl(root, dag, goals, builder);
}

void FlatHorizonEncoderEngine::encode_impl(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals,
   BatchBuilder& builder,
   std::vector< mimir::search::State >* batch_target_name_states,
   bool prepare_builder_once
)
{
   auto* profile = g_flat_horizon_batch_profile;
   // Summary:
   // 1. Build the per-graph state, target, and relation context for the DAG.
   // 2. Emit root facts, candidate-state facts, goal views, and topology relations.
   // 3. Write node rows, target rows, tuple data, and optional LGAN edges.
   // Phase 1: prepare shared schema state and build the per-graph horizon context.
   if(prepare_builder_once) {
      ScopedProfileTimer timer(profile != nullptr ? &profile->prepare_builder_s : nullptr);
      prepare_builder(builder);
   }
   auto context = [&]() {
      ScopedProfileTimer timer(profile != nullptr ? &profile->make_context_s : nullptr);
      return make_context(root, dag);
   }();
   FlatRelationSink sink(relation_names_.size(), config_.include_lgan_edges);

   // Phase 2: define the local emit helpers used by the different horizon passes.
   auto emit_state_facts = [&]< typename Tag >(
                              const auto& atoms,
                              std::optional< int64_t > state_entity_index,
                              hash_set< uint64_t >& fact_keys
                           ) {
      for(const auto& atom : atoms) {
         const int raw_arity = static_cast< int >(atom->get_predicate()->get_arity());
         if(config_.ignore_zero_arity_relations and raw_arity == 0) {
            continue;
         }
         const std::string base_relation_name = RelationFormatter::format_predicate(
            atom->get_predicate()
         );
         const auto relation_id = relation_id_for(
            state_entity_index.has_value() && split_full_state_relations(config_)
               ? state_anchored_relation_name(base_relation_name)
               : base_relation_name
         );
         const auto args = local_arg_rows_for_atom(*this, context, atom, state_entity_index);
         sink.emit(relation_id, args);
         fact_keys.insert(
            pack_u32_u32(static_cast< uint32_t >(atom->get_index()), fact_tag_id< Tag >())
         );
      }
   };

   auto emit_state_for_candidate = [&](
                                      const mimir::search::State& state,
                                      int node_index,
                                      bool include_static,
                                      bool include_state_anchor
                                   ) -> hash_set< uint64_t > {
      hash_set< uint64_t > fact_keys;
      const std::optional< int64_t > state_entity_index = include_state_anchor
                                                             ? std::optional< int64_t >(
                                                                  state_entity_index_for(
                                                                     context, node_index
                                                                  )
                                                               )
                                                             : std::nullopt;
      const auto& problem = state.get_problem();
      const auto& repos = problem.get_repositories();

      if(include_static) {
         for(const auto& literal : problem.get_initial_literals< mimir::formalism::StaticTag >()) {
            if(not literal->get_polarity()) {
               continue;
            }
            std::array atoms{literal->get_atom()};
            emit_state_facts.template operator()< mimir::formalism::StaticTag >(
               atoms, state_entity_index, fact_keys
            );
         }
      }

      const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         state.get_atoms< mimir::formalism::FluentTag >()
      );
      emit_state_facts.template operator()< mimir::formalism::FluentTag >(
         fluent_atoms, state_entity_index, fact_keys
      );

      const auto derived_atoms = repos
                                    .get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
                                       state.get_atoms< mimir::formalism::DerivedTag >()
                                    );
      emit_state_facts.template operator()< mimir::formalism::DerivedTag >(
         derived_atoms, state_entity_index, fact_keys
      );

      return fact_keys;
   };

   auto emit_goal_literals =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const auto& goal_levels,
         int node_index,
         bool include_state_anchor
      ) {
         const std::optional< int64_t > state_entity_index = include_state_anchor
                                                                ? std::optional< int64_t >(
                                                                     state_entity_index_for(
                                                                        context, node_index
                                                                     )
                                                                  )
                                                                : std::nullopt;
         for(const auto& literal : literals) {
            const auto predicate = literal->get_atom()->get_predicate();
            const int raw_arity = static_cast< int >(predicate->get_arity());
            if(config_.ignore_zero_arity_relations and raw_arity == 0) {
               continue;
            }
            const auto level = goal_level_for(goal_levels, literal);
            std::string relation_name;
            if(level.has_value()) {
               relation_name = RelationFormatter::format_predicate(
                  predicate, GoalLevel(*level), std::nullopt, literal->get_polarity()
               );
            } else {
               relation_name = RelationFormatter::format_predicate(
                  predicate, std::nullopt, std::nullopt, literal->get_polarity()
               );
            }
            if(state_entity_index.has_value() && split_full_state_relations(config_)) {
               relation_name = state_anchored_relation_name(relation_name);
            }
            const auto relation_id = relation_id_for(relation_name);
            const auto args = local_arg_rows_for_atom(
               *this, context, literal->get_atom(), state_entity_index
            );
            sink.emit(relation_id, args);
         }
      };

   auto emit_goal_satisfaction =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const auto& goal_levels,
         const hash_set< uint64_t >& fact_keys,
         int node_index,
         bool include_state_anchor
      ) {
         const std::optional< int64_t > state_entity_index = include_state_anchor
                                                                ? std::optional< int64_t >(
                                                                     state_entity_index_for(
                                                                        context, node_index
                                                                     )
                                                                  )
                                                                : std::nullopt;
         for(const auto& literal : literals) {
            const auto predicate = literal->get_atom()->get_predicate();
            const int raw_arity = static_cast< int >(predicate->get_arity());
            if(config_.ignore_zero_arity_relations and raw_arity == 0) {
               continue;
            }
            const uint64_t fact_key = pack_u32_u32(
               static_cast< uint32_t >(literal->get_atom()->get_index()), fact_tag_id< GoalTag >()
            );
            const bool satisfied = fact_keys.contains(fact_key) == literal->get_polarity();
            const GoalDerivation satisfaction = satisfied ? GoalDerivation::satisfied
                                                          : GoalDerivation::unsatisfied;
            if(not config_.goal_derivations.contains(satisfaction)) {
               continue;
            }
            const auto level = goal_level_for(goal_levels, literal);
            std::string relation_name;
            if(level.has_value()) {
               relation_name = RelationFormatter::format_predicate(
                  predicate, GoalLevel(*level), satisfaction, literal->get_polarity()
               );
            } else {
               relation_name = RelationFormatter::format_predicate(
                  predicate, std::nullopt, satisfaction, literal->get_polarity()
               );
            }
            if(state_entity_index.has_value() && split_full_state_relations(config_)) {
               relation_name = state_anchored_relation_name(relation_name);
            }
            const auto relation_id = relation_id_for(relation_name);
            const auto args = local_arg_rows_for_atom(
               *this, context, literal->get_atom(), state_entity_index
            );
            sink.emit(relation_id, args);
         }
      };

   auto emit_delta_literal =
      [&]< typename AtomTag >(
         const mimir::formalism::GroundAtom< AtomTag >& atom, bool polarity, int node_index
      ) {
         const int raw_arity = static_cast< int >(atom->get_predicate()->get_arity());
         if(config_.ignore_zero_arity_relations and raw_arity == 0) {
            return;
         }
         const auto state_entity_index = state_entity_index_for(context, node_index);
         std::string relation_name = RelationFormatter::format_predicate(
            atom->get_predicate(), std::nullopt, std::nullopt, polarity
         );
         if(split_full_state_relations(config_)) {
            relation_name = state_anchored_relation_name(relation_name);
         }
         const auto relation_id = relation_id_for(relation_name);
         const auto args = local_arg_rows_for_atom(*this, context, atom, state_entity_index);
         sink.emit(relation_id, args);
      };

   auto emit_delta_goal_satisfaction =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const auto& goal_levels,
         const hash_set< int >& added_set,
         const hash_set< int >& removed_set,
         int node_index
      ) {
         const int64_t state_entity_index = state_entity_index_for(context, node_index);
         for(const auto& goal : literals) {
            const auto atom = goal->get_atom();
            const int raw_arity = static_cast< int >(atom->get_predicate()->get_arity());
            if(config_.ignore_zero_arity_relations and raw_arity == 0) {
               continue;
            }
            const int idx = atom->get_index();
            const bool added_match = added_set.contains(idx);
            const bool removed_match = removed_set.contains(idx);
            const auto sat = delta_goal_satisfaction_derivation(
               goal->get_polarity(), added_match, removed_match
            );
            if(not sat.has_value() or not config_.goal_derivations.contains(*sat)) {
               continue;
            }
            const auto level = goal_level_for(goal_levels, goal);
            std::string relation_name;
            if(level.has_value()) {
               relation_name = RelationFormatter::format_predicate(
                  atom->get_predicate(), GoalLevel(*level), *sat, goal->get_polarity()
               );
            } else {
               relation_name = RelationFormatter::format_predicate(
                  atom->get_predicate(), std::nullopt, *sat, goal->get_polarity()
               );
            }
            if(split_full_state_relations(config_)) {
               relation_name = state_anchored_relation_name(relation_name);
            }
            const auto relation_id = relation_id_for(relation_name);
            const auto args = local_arg_rows_for_atom(*this, context, atom, state_entity_index);
            sink.emit(relation_id, args);
         }
      };

   auto emit_action = [&](const mimir::formalism::GroundAction& action, int node_index) {
      const int64_t state_entity_index = state_entity_index_for(context, node_index);
      const auto relation_id = relation_id_for(
         RelationFormatter::format_action_schema(*action->get_action())
      );
      const auto args = local_arg_rows_for_action(context, action, state_entity_index);
      sink.emit(relation_id, args);
   };

   // Phase 3: encode the root state and any goal facts that live on the root.
   const auto root_fact_keys = [&]() {
      ScopedProfileTimer timer(profile != nullptr ? &profile->root_emit_s : nullptr);
      const auto fact_keys = emit_state_for_candidate(
         root,
         dag.root_index(),
         config_.include_static,
         /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
      );
      if(includes_plain_goal_derivation(config_.goal_derivations)) {
         emit_goal_literals(
            std::span{goals.static_goals},
            goals.static_goal_levels,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
         emit_goal_literals(
            std::span{goals.fluent_goals},
            goals.fluent_goal_levels,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
         emit_goal_literals(
            std::span{goals.derived_goals},
            goals.derived_goal_levels,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
      }
      if(has_non_plain_goal_derivations(config_.goal_derivations)) {
         emit_goal_satisfaction(
            std::span{goals.static_goals},
            goals.static_goal_levels,
            fact_keys,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
         emit_goal_satisfaction(
            std::span{goals.fluent_goals},
            goals.fluent_goal_levels,
            fact_keys,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
         emit_goal_satisfaction(
            std::span{goals.derived_goals},
            goals.derived_goal_levels,
            fact_keys,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
      }
      return fact_keys;
   }();

   const bool encode_actions = (not config_.ignore_actions)
                               or (config_.transition_mode == Mode::action);

   // Phase 4: encode every non-root candidate state and its per-node relations.
   hash_set< int > root_fluent_indices;
   hash_set< int > root_derived_indices;
   const auto emit_provided_delta_literals = [&](
                                                const TransitionDAG::Node& node,
                                                hash_set< int >& added_fluents,
                                                hash_set< int >& removed_fluents,
                                                hash_set< int >& added_derived,
                                                hash_set< int >& removed_derived
                                             ) {
      if(not node.delta_literals.has_value()) {
         return false;
      }
      for(const auto& literal_variant : *node.delta_literals) {
         std::visit(
            [&]< typename Tag >(const mimir::formalism::GroundLiteral< Tag >& literal) {
               const auto atom = literal->get_atom();
               if(config_.ignore_zero_arity_relations and atom->get_predicate()->get_arity() == 0) {
                  return;
               }
               emit_delta_literal(atom, literal->get_polarity(), node.index);
               if constexpr(std::is_same_v< Tag, mimir::formalism::FluentTag >) {
                  if(literal->get_polarity()) {
                     added_fluents.insert(atom->get_index());
                  } else {
                     removed_fluents.insert(atom->get_index());
                  }
               } else if constexpr(std::is_same_v< Tag, mimir::formalism::DerivedTag >) {
                  if(literal->get_polarity()) {
                     added_derived.insert(atom->get_index());
                  } else {
                     removed_derived.insert(atom->get_index());
                  }
               }
            },
            literal_variant
         );
      }
      return true;
   };
   if(config_.transition_mode == Mode::delta) {
      ScopedProfileTimer timer(profile != nullptr ? &profile->root_delta_setup_s : nullptr);
      const auto& repos = root.get_problem().get_repositories();
      const auto root_fluents = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         root.get_atoms< mimir::formalism::FluentTag >()
      );
      for(const auto& atom : root_fluents) {
         if(config_.ignore_zero_arity_relations and atom->get_predicate()->get_arity() == 0) {
            continue;
         }
         root_fluent_indices.insert(atom->get_index());
      }
      const auto root_derived = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
         root.get_atoms< mimir::formalism::DerivedTag >()
      );
      for(const auto& atom : root_derived) {
         if(config_.ignore_zero_arity_relations and atom->get_predicate()->get_arity() == 0) {
            continue;
         }
         root_derived_indices.insert(atom->get_index());
      }
   }

   {
      ScopedProfileTimer timer(profile != nullptr ? &profile->candidate_loop_s : nullptr);
      for(const auto& node : dag.nodes()) {
         if(node.index == dag.root_index()) {
            continue;
         }

         if(config_.transition_mode == Mode::full) {
            const auto succ_fact_keys = emit_state_for_candidate(
               node.state, node.index, false, /*include_state_anchor=*/true
            );
            if(encode_actions and node.action.has_value()) {
               emit_action(*node.action, node.index);
            }
            if(has_non_plain_goal_derivations(config_.goal_derivations)) {
               emit_goal_satisfaction.template operator()< mimir::formalism::StaticTag >(
                  std::span{goals.static_goals},
                  goals.static_goal_levels,
                  succ_fact_keys,
                  node.index,
                  /*include_state_anchor=*/true
               );
               emit_goal_satisfaction.template operator()< mimir::formalism::FluentTag >(
                  std::span{goals.fluent_goals},
                  goals.fluent_goal_levels,
                  succ_fact_keys,
                  node.index,
                  /*include_state_anchor=*/true
               );
               emit_goal_satisfaction.template operator()< mimir::formalism::DerivedTag >(
                  std::span{goals.derived_goals},
                  goals.derived_goal_levels,
                  succ_fact_keys,
                  node.index,
                  /*include_state_anchor=*/true
               );
            }
         } else if(config_.transition_mode == Mode::delta) {
            hash_set< int > added_fluents;
            hash_set< int > removed_fluents;
            hash_set< int > added_derived;
            hash_set< int > removed_derived;
            const bool used_provided_delta = emit_provided_delta_literals(
               node, added_fluents, removed_fluents, added_derived, removed_derived
            );
            if(profile != nullptr) {
               if(used_provided_delta) {
                  ++profile->provided_delta_nodes;
               } else {
                  ++profile->fallback_delta_nodes;
               }
            }

            if(not used_provided_delta) {
               ScopedProfileTimer delta_timer(
                  profile != nullptr ? &profile->delta_fallback_s : nullptr
               );
               const auto& repos = node.state.get_problem().get_repositories();
               const auto
                  succ_fluents = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
                     node.state.get_atoms< mimir::formalism::FluentTag >()
                  );
               const auto succ_derived = repos.get_ground_atoms_from_indices<
                  mimir::formalism::DerivedTag >(
                  node.state.get_atoms< mimir::formalism::DerivedTag >()
               );

               hash_set< int > succ_fluent_indices;
               for(const auto& atom : succ_fluents) {
                  if(config_.ignore_zero_arity_relations
                     and atom->get_predicate()->get_arity() == 0) {
                     continue;
                  }
                  succ_fluent_indices.insert(atom->get_index());
                  if(not root_fluent_indices.contains(atom->get_index())) {
                     added_fluents.insert(atom->get_index());
                     emit_delta_literal(atom, true, node.index);
                  }
               }
               for(const auto idx : root_fluent_indices) {
                  if(not succ_fluent_indices.contains(idx)) {
                     removed_fluents.insert(idx);
                     const auto atom = repos.get_ground_atom< mimir::formalism::FluentTag >(idx);
                     emit_delta_literal(atom, false, node.index);
                  }
               }

               hash_set< int > succ_derived_indices;
               for(const auto& atom : succ_derived) {
                  if(config_.ignore_zero_arity_relations
                     and atom->get_predicate()->get_arity() == 0) {
                     continue;
                  }
                  succ_derived_indices.insert(atom->get_index());
                  if(not root_derived_indices.contains(atom->get_index())) {
                     added_derived.insert(atom->get_index());
                     emit_delta_literal(atom, true, node.index);
                  }
               }
               for(const auto idx : root_derived_indices) {
                  if(not succ_derived_indices.contains(idx)) {
                     removed_derived.insert(idx);
                     const auto atom = repos.get_ground_atom< mimir::formalism::DerivedTag >(idx);
                     emit_delta_literal(atom, false, node.index);
                  }
               }
            }

            if(encode_actions and node.action.has_value()) {
               emit_action(*node.action, node.index);
            }

            if(has_non_plain_goal_derivations(config_.goal_derivations)) {
               emit_delta_goal_satisfaction(
                  std::span{goals.static_goals},
                  goals.static_goal_levels,
                  hash_set< int >{},
                  hash_set< int >{},
                  node.index
               );
               emit_delta_goal_satisfaction(
                  std::span{goals.fluent_goals},
                  goals.fluent_goal_levels,
                  added_fluents,
                  removed_fluents,
                  node.index
               );
               emit_delta_goal_satisfaction(
                  std::span{goals.derived_goals},
                  goals.derived_goal_levels,
                  added_derived,
                  removed_derived,
                  node.index
               );
            }
         } else {
            if(encode_actions and node.action.has_value()) {
               emit_action(*node.action, node.index);
            }
         }
      }
   }

   // Phase 5: add graph topology relations such as parent, sibling, and cousin links.
   {
      ScopedProfileTimer timer(profile != nullptr ? &profile->topology_relations_s : nullptr);
      const int root_index = dag.root_index();
      const bool exclude_root_topology = config_.root_policy == RootPolicy::exclude;
      if(config_.enable_parent_relation) {
         const int relation_id = relation_id_for(config_.parent_relation);
         for(const auto& [parent_idx, child_idx] : dag.transitions()) {
            if(exclude_root_topology && parent_idx == root_index) {
               continue;
            }
            const std::array< int64_t, 2 > args = {
               state_entity_index_for(context, parent_idx),
               state_entity_index_for(context, child_idx),
            };
            sink.emit(relation_id, args);
         }
      }

      if(config_.enable_sibling_relation or config_.enable_cousin_relation) {
         hash_map< int, std::vector< int > > parent_to_children;
         for(const auto& [parent_idx, child_idx] : dag.transitions()) {
            if(exclude_root_topology && parent_idx == root_index) {
               continue;
            }
            parent_to_children[parent_idx].push_back(child_idx);
         }

         auto emit_directed_pair_relation =
            [&](const std::string& relation_name, int src, int dst) {
               const int relation_id = relation_id_for(relation_name);
               const std::array< int64_t, 2 > args = {
                  state_entity_index_for(context, src),
                  state_entity_index_for(context, dst),
               };
               sink.emit(relation_id, args);
            };

         std::set< std::pair< int, int > > siblings_seen;
         if(config_.enable_sibling_relation) {
            for(auto& children : parent_to_children | std::views::values) {
               std::ranges::sort(children);
               for(size_t i = 0; i < children.size(); ++i) {
                  for(size_t j = i + 1; j < children.size(); ++j) {
                     const int a = children[i];
                     const int b = children[j];
                     const auto pair = std::pair{a, b};
                     if(siblings_seen.contains(pair)) {
                        continue;
                     }
                     siblings_seen.insert(pair);
                     emit_directed_pair_relation(config_.sibling_relation, a, b);
                     emit_directed_pair_relation(config_.sibling_relation, b, a);
                  }
               }
            }
         }

         if(config_.enable_cousin_relation) {
            std::set< std::pair< int, int > > cousins_seen;
            for(const auto& parents : parent_to_children | std::views::values) {
               std::vector< int > par = parents;
               std::ranges::sort(par);
               for(size_t i = 0; i < par.size(); ++i) {
                  for(size_t j = i + 1; j < par.size(); ++j) {
                     const int pu = par[i];
                     const int pv = par[j];
                     const auto cu_it = parent_to_children.find(pu);
                     const auto cv_it = parent_to_children.find(pv);
                     if(cu_it == parent_to_children.end() or cv_it == parent_to_children.end()) {
                        continue;
                     }
                     const auto& cu = cu_it->second;
                     const auto& cv = cv_it->second;
                     for(int u : cu) {
                        for(int v : cv) {
                           if(u == v) {
                              continue;
                           }
                           const int a = std::min(u, v);
                           const int b = std::max(u, v);
                           const auto pair = std::pair{a, b};
                           if(cousins_seen.contains(pair) or siblings_seen.contains(pair)) {
                              continue;
                           }
                           cousins_seen.insert(pair);
                           emit_directed_pair_relation(config_.cousin_relation, u, v);
                           emit_directed_pair_relation(config_.cousin_relation, v, u);
                        }
                     }
                  }
               }
            }
         }
      }
   }

   {
      ScopedProfileTimer timer(profile != nullptr ? &profile->finalize_builder_s : nullptr);
      // Phase 6: export node rows, target rows, and emitted relation tuples.
      std::vector< float > zeros(context.entity_names.size(), 0.0f);
      builder.add_node_features(
         std::string(kFlatEntityNodeType),
         "x",
         std::span< const float >(zeros.data(), zeros.size()),
         1
      );
      if(config_.export_node_names) {
         builder.set_node_names(std::string(kFlatEntityNodeType), context.entity_names);
         builder.set_object_names(context.object_names);
      }

      const int64_t node_size = static_cast< int64_t >(context.entity_names.size());
      const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
      const int64_t target_entity_size = static_cast< int64_t >(
         context.target_entity_indices.size()
      );
      const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
      builder.set_field(std::string(kNodeSizesField), std::span< const int64_t >(&node_size, 1));
      builder.set_field(
         std::string(kObjectSizesField), std::span< const int64_t >(&object_size, 1)
      );
      builder.set_field(
         std::string(kObjectIndicesField),
         std::span< const int64_t >(context.object_indices.data(), context.object_indices.size())
      );
      builder.set_field(
         std::string(kEntityRoleIdsField),
         std::span< const int64_t >(context.entity_role_ids.data(), context.entity_role_ids.size())
      );
      builder.set_field(
         std::string(kTargetEntitySizesField), std::span< const int64_t >(&target_entity_size, 1)
      );
      builder.set_field(
         std::string(kTargetEntityIndicesField),
         std::span< const int64_t >(
            context.target_entity_indices.data(), context.target_entity_indices.size()
         )
      );
      builder.set_field(
         std::string(kTargetEntityGroupIdsField),
         std::span< const int64_t >(
            context.target_entity_group_ids.data(), context.target_entity_group_ids.size()
         )
      );
      builder.set_field(
         std::string(kTargetSizesField), std::span< const int64_t >(&target_size, 1)
      );

      const TargetMetadataEmitConfig target_emit_config{
         .position_node_type_id = std::string(kFlatEntityNodeType),
         .symbol_prefix = config_.target_symbol_prefix,
         .include_depth = true,
         .include_group = true,
         .include_names = false,
         .groups = target_metadata_group_names_,
         .parent_relation = config_.parent_relation,
      };
      set_target_fields(builder, context.target_columns, target_emit_config);
      set_target_graph_attrs(builder, context.target_columns, target_emit_config);
      if(config_.export_node_names) {
         if(batch_target_name_states != nullptr) {
            if(not context.target_name_states.empty()) {
               batch_target_name_states->insert(
                  batch_target_name_states->end(),
                  context.target_name_states.begin(),
                  context.target_name_states.end()
               );
            }
         } else {
            if(context.target_name_states.empty()) {
               builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
            } else {
               pymimir_backend::add_deferred_state_names(
                  builder, std::span(context.target_name_states)
               );
            }
         }
      }

      builder.set_field(
         std::string(kRelationCountsField),
         std::span< const int64_t >(sink.relation_counts().data(), sink.relation_counts().size())
      );
      const int64_t relation_instance_size = sink.relation_instance_count();
      builder.set_field(
         std::string(kRelationInstanceSizesField),
         std::span< const int64_t >(&relation_instance_size, 1)
      );
      builder.set_field(
         std::string(kRelationArgsField),
         std::span< const int64_t >(sink.relation_args().data(), sink.relation_args().size())
      );
   }
   if(profile != nullptr) {
      ++profile->graphs;
      profile->dag_nodes += static_cast< int64_t >(dag.nodes().size());
      profile->candidate_nodes += static_cast< int64_t >(
         std::max< int64_t >(0, static_cast< int64_t >(dag.nodes().size()) - 1)
      );
      profile->entity_rows += static_cast< int64_t >(context.entity_names.size());
      profile->relation_instances += sink.relation_instance_count();
   }
   if(config_.include_lgan_edges) {
      ScopedProfileTimer timer(profile != nullptr ? &profile->lgan_s : nullptr);
      // Phase 7: derive LGAN helper edges from the final tuple sink and target rows.
      if(context.target_columns.positions.empty()) {
         throw std::invalid_argument(
            "FlatHorizonEncoder include_lgan_edges=true requires surviving candidate state "
            "rows, but none were encoded. Ensure the horizon DAG exposes at least one "
            "selectable candidate state."
         );
      }
      const auto lgan = build_flat_lgan(sink, std::span{context.target_columns.positions});
      const int64_t tn_size = static_cast< int64_t >(lgan.tn_relation_indices.size());
      const int64_t nn_size = static_cast< int64_t >(lgan.nn_relation_indices.size());
      const int64_t rr_size = static_cast< int64_t >(lgan.rr_src_relation_indices.size());
      builder.set_field(std::string(kLGANTNSizesField), std::span< const int64_t >(&tn_size, 1));
      builder.set_field(
         std::string(kLGANTNRelationIndicesField),
         std::span< const int64_t >(
            lgan.tn_relation_indices.data(), lgan.tn_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANTNEntityIndicesField),
         std::span< const int64_t >(lgan.tn_entity_indices.data(), lgan.tn_entity_indices.size())
      );
      builder.set_field(std::string(kLGANNNSizesField), std::span< const int64_t >(&nn_size, 1));
      builder.set_field(
         std::string(kLGANNNRelationIndicesField),
         std::span< const int64_t >(
            lgan.nn_relation_indices.data(), lgan.nn_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANNNEntityIndicesField),
         std::span< const int64_t >(lgan.nn_entity_indices.data(), lgan.nn_entity_indices.size())
      );
      builder.set_field(std::string(kLGANRRSizesField), std::span< const int64_t >(&rr_size, 1));
      builder.set_field(
         std::string(kLGANRRSrcRelationIndicesField),
         std::span< const int64_t >(
            lgan.rr_src_relation_indices.data(), lgan.rr_src_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANRRDstRelationIndicesField),
         std::span< const int64_t >(
            lgan.rr_dst_relation_indices.data(), lgan.rr_dst_relation_indices.size()
         )
      );
   }
}

BatchBuilder::BatchEncoding FlatHorizonEncoderEngine::encode_batch(
   const batch_input::parsed::HorizonBatchInputs& inputs
)
{
   // Summary:
   // 1. Read one batch item at a time and normalize goals and the optional DAG.
   // 2. Encode each item into the shared builder as one flat graph.
   // 3. Write batch-level target metadata after all graphs are appended.
   FlatHorizonBatchProfile profile;
   profile.enabled = flat_horizon_batch_profile_enabled();
   struct ProfileGuard {
      FlatHorizonBatchProfile* previous = nullptr;
      explicit ProfileGuard(FlatHorizonBatchProfile* current)
          : previous(g_flat_horizon_batch_profile)
      {
         g_flat_horizon_batch_profile = current;
      }
      ~ProfileGuard() { g_flat_horizon_batch_profile = previous; }
   } guard(profile.enabled ? &profile : nullptr);

   BatchBuilder builder;
   builder.set_graph_kind("flat");
   {
      ScopedProfileTimer timer(profile.enabled ? &profile.prepare_builder_s : nullptr);
      prepare_builder(builder);
   }

   const size_t state_count = inputs.roots.states.size();
   std::vector< mimir::search::State > batch_target_name_states;
   for(size_t idx = 0; idx < state_count; ++idx) {
      // Phase 1: collect the root state, optional DAG, and optional goal payloads.
      const auto& root_entry = inputs.roots.states[idx];
      const auto& dag_entry = inputs.dags.at(idx);
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);

      GoalInputs goal_inputs;
      {
         // Phase 2: normalize the batch goal inputs for this root state.
         ScopedProfileTimer timer(profile.enabled ? &profile.goal_inputs_s : nullptr);
         if(goals_entry.has_value()) {
            const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                      : nullptr;
            goal_inputs = batch_input::compose_goal_inputs(*goals_entry, layers_ptr);
         } else {
            goal_inputs = batch_input::default_goal_inputs_for_batch_state(root_entry);
            if(subgoal_layers_entry.has_value()) {
               size_t level = 1;
               for(const auto& layer : *subgoal_layers_entry) {
                  goal_inputs.extend(layer, level);
                  ++level;
               }
            }
         }
      }

      std::optional< TransitionDAG > default_dag = std::nullopt;
      const TransitionDAG* dag_ptr = nullptr;
      if(dag_entry.has_value()) {
         dag_ptr = &(*dag_entry);
      } else {
         // Build a one-node DAG when the batch input does not supply a horizon graph.
         ScopedProfileTimer timer(profile.enabled ? &profile.default_dag_s : nullptr);
         default_dag.emplace(root_entry.state);
         dag_ptr = &(*default_dag);
      }
      {
         // Phase 3: encode one horizon graph and move the builder to the next graph.
         ScopedProfileTimer timer(profile.enabled ? &profile.encode_impl_s : nullptr);
         encode_impl(
            root_entry.state,
            *dag_ptr,
            goal_inputs,
            builder,
            &batch_target_name_states,
            /*prepare_builder_once=*/false
         );
      }
      builder.next_graph();
   }

   // Phase 4: write batch-level target metadata once all graphs are present.
   if(config_.export_node_names) {
      if(batch_target_name_states.empty()) {
         builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
      } else {
         pymimir_backend::add_deferred_state_names(builder, std::span(batch_target_name_states));
      }
   }
   builder.set_graph_attr(std::string(kTargetGroupsAttr), target_metadata_group_names_);
   builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   builder.set_graph_attr(std::string(kParentRelationAttr), config_.parent_relation);
   print_flat_horizon_batch_profile(profile);
   auto encoding = builder.build();
   finalize_batch_encoding(encoding);
   return encoding;
}

void FlatHorizonEncoderEngine::finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
{
   if(config_.pack_relation_args_relation_major) {
      pack_flat_relation_args_relation_major(encoding, std::span{relation_arities_});
   }
}

}  // namespace mifrost
