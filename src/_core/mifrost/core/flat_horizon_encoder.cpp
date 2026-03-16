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

#include "flat_lgan.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

namespace {

constexpr std::string_view kEntityNodeType = "entity";
constexpr std::string_view kFlatEntityTypeAttr = "entity_node_type";
constexpr std::string_view kIncludeLGANEdgesAttr = "include_lgan_edges";
constexpr std::string_view kRelationNamesAttr = "relation_names";
constexpr std::string_view kRelationAritiesAttr = "relation_arities";
constexpr std::string_view kRelationSourcesAttr = "relation_sources";
constexpr std::string_view kNodeSizesField = "node_sizes";
constexpr std::string_view kObjectSizesField = "object_sizes";
constexpr std::string_view kObjectIndicesField = "object_indices";
constexpr std::string_view kTargetEntitySizesField = "target_entity_sizes";
constexpr std::string_view kTargetEntityIndicesField = "target_entity_indices";
constexpr std::string_view kTargetEntityGroupIdsField = "target_entity_group_ids";
constexpr std::string_view kTargetEntityGroupsAttr = "target_entity_groups";
constexpr std::string_view kTargetSizesField = "target_sizes";
constexpr std::string_view kRelationInstanceSizesField = "relation_instance_sizes";
constexpr std::string_view kRelationCountsField = "relation_counts";
constexpr std::string_view kRelationArgsField = "relation_args";
constexpr std::string_view kLGANTNSizesField = "lgan_tn_sizes";
constexpr std::string_view kLGANTNRelationIndicesField = "lgan_tn_relation_indices";
constexpr std::string_view kLGANTNEntityIndicesField = "lgan_tn_entity_indices";
constexpr std::string_view kLGANNNSizesField = "lgan_nn_sizes";
constexpr std::string_view kLGANNNRelationIndicesField = "lgan_nn_relation_indices";
constexpr std::string_view kLGANNNEntityIndicesField = "lgan_nn_entity_indices";
constexpr std::string_view kLGANRRSizesField = "lgan_rr_sizes";
constexpr std::string_view kLGANRRSrcRelationIndicesField = "lgan_rr_src_relation_indices";
constexpr std::string_view kLGANRRDstRelationIndicesField = "lgan_rr_dst_relation_indices";
constexpr std::string_view kLGANTNEdgePosAttr = "lgan_tn_edge_pos";
constexpr std::string_view kLGANNNEdgePosAttr = "lgan_nn_edge_pos";
constexpr std::string_view kLGANRREdgePosAttr = "lgan_rr_edge_pos";
constexpr std::string_view kHiddenRootCarrierName = "_root_state_";
constexpr std::string_view kCandidateRelationSuffix = "[state]";

template < typename... Args >
std::string state_anchored_relation_name(const std::string_view name, Args&&... args)
{
   return RelationFormatter::format_predicate(
      name, std::forward< Args >(args)..., kCandidateRelationSuffix
   );
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

template < typename GoalLevelsMap, typename LiteralTag >
std::optional< size_t > goal_level_for(
   const GoalLevelsMap& goal_levels,
   const mimir::formalism::GroundLiteral< LiteralTag >& literal
)
{
   if(const auto it = goal_levels.find(literal); it != goal_levels.end()) {
      return it->second;
   }
   return std::nullopt;
}

template < typename AtomTag >
std::vector< int64_t > local_arg_rows_for_atom(
   const FlatHorizonEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAtom< AtomTag >& atom,
   std::optional< int64_t > state_entity_index
)
{
   std::vector< int64_t > args;
   args.reserve(atom->get_objects().size() + (state_entity_index.has_value() ? 1U : 0U));
   if(state_entity_index.has_value()) {
      args.push_back(*state_entity_index);
   }
   for(const auto& obj : atom->get_objects()) {
      const auto it = context.entity_index_by_object_id.find(
         static_cast< int64_t >(obj->get_index())
      );
      if(it == context.entity_index_by_object_id.end()) {
         throw std::invalid_argument(
            "Flat horizon encoder encountered object not present in entity table: "
            + RelationFormatter::format_object(*obj)
         );
      }
      args.push_back(it->second);
   }
   return args;
}

std::vector< int64_t > local_arg_rows_for_action(
   const FlatHorizonEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAction& action,
   std::optional< int64_t > state_entity_index
)
{
   std::vector< int64_t > args;
   args.reserve(action->get_objects().size() + (state_entity_index.has_value() ? 1U : 0U));
   if(state_entity_index.has_value()) {
      args.push_back(*state_entity_index);
   }
   for(const auto& obj : action->get_objects()) {
      const auto it = context.entity_index_by_object_id.find(
         static_cast< int64_t >(obj->get_index())
      );
      if(it == context.entity_index_by_object_id.end()) {
         throw std::invalid_argument(
            "Flat horizon encoder encountered action object not present in entity table: "
            + RelationFormatter::format_object(*obj)
         );
      }
      args.push_back(it->second);
   }
   return args;
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
   actions.assign(domain_.get_actions().begin(), domain_.get_actions().end());

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
      action_specs_.push_back(
         PredicateSpec{
            .name = RelationFormatter::format_action_schema(*action),
            .arity = static_cast< int >(action->get_arity()) + 1,
         }
      );
   }
   std::ranges::sort(action_specs_, predicate_order);

   relation_dict_ = RelationDict(
      domain_, actions, rel_config, root_in_state_relations(config_.root_policy) ? 1 : 0, 1
   );
   if(config_.enable_parent_relation) {
      relation_dict_.arity[config_.parent_relation] = 2;
   }
   if(config_.enable_sibling_relation) {
      relation_dict_.arity[config_.sibling_relation] = 2;
   }
   if(config_.enable_cousin_relation) {
      relation_dict_.arity[config_.cousin_relation] = 2;
   }

   target_entity_group_names_ = {std::string(target_source_group_name(TargetSource::states))};
   target_metadata_group_names_ = target_entity_group_names_;

   std::map< std::string, std::string > relation_sources_by_name;
   auto add_predicate_relation =
      [&](const std::string& base_name, int base_arity, std::string source) {
         relation_dict_.arity[base_name] = root_in_state_relations(config_.root_policy)
                                              ? base_arity + 1
                                              : base_arity;
         relation_sources_by_name[base_name] = source;
         if(root_uses_split_state_relations(config_.root_policy)) {
            const auto candidate_name = state_anchored_relation_name(base_name);
            relation_dict_.arity[candidate_name] = base_arity + 1;
            relation_sources_by_name[candidate_name] = source;
         }
      };

   for(const auto& spec : predicate_specs_) {
      add_predicate_relation(spec.name, spec.arity, "state");
   }
   for(const auto& spec : regular_predicate_specs_) {
      if(includes_plain_goal_derivation(config_.goal_derivations)) {
         for(size_t level = 0; level <= config_.max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               add_predicate_relation(
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
            add_predicate_relation(
               RelationFormatter::format_predicate(spec.name, std::nullopt, std::nullopt, polarity),
               spec.arity,
               "state"
            );
         }
      }
      for(const auto derivation : goal_satisfaction_derivations(config_.goal_derivations)) {
         for(size_t level = 0; level <= config_.max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               add_predicate_relation(
                  RelationFormatter::format_predicate(spec.name, goal_level, derivation, polarity),
                  spec.arity,
                  "goal_satisfaction"
               );
            }
         }
         if(config_.support_literals) {
            for(bool polarity : {true, false}) {
               add_predicate_relation(
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
   for(const auto& spec : action_specs_) {
      relation_sources_by_name.emplace(spec.name, "action");
   }
   if(config_.enable_parent_relation) {
      relation_sources_by_name[config_.parent_relation] = "parent";
   }
   if(config_.enable_sibling_relation) {
      relation_sources_by_name[config_.sibling_relation] = "sibling";
   }
   if(config_.enable_cousin_relation) {
      relation_sources_by_name[config_.cousin_relation] = "cousin";
   }

   relation_names_.clear();
   relation_arities_.clear();
   relation_sources_.clear();
   relation_name_to_id_.clear();
   relation_names_.reserve(relation_dict_.arity.size());
   relation_arities_.reserve(relation_dict_.arity.size());
   relation_sources_.reserve(relation_dict_.arity.size());
   relation_name_to_id_.reserve(relation_dict_.arity.size());
   for(const auto& [name, arity] : relation_dict_.arity) {
      relation_name_to_id_.emplace(name, static_cast< int >(relation_names_.size()));
      relation_names_.push_back(name);
      relation_arities_.push_back(arity);
      const auto source_it = relation_sources_by_name.find(name);
      relation_sources_.push_back(
         source_it != relation_sources_by_name.end() ? source_it->second : "state"
      );
   }
}

void FlatHorizonEncoderEngine::prepare_builder(BatchBuilder& builder) const
{
   builder.set_graph_kind("flat");
   builder.set_schema_flag("flat_relations", true);
   builder.set_graph_attr(std::string(kFlatEntityTypeAttr), std::string(kEntityNodeType));
   builder.set_graph_attr(
      std::string(kIncludeLGANEdgesAttr), static_cast< int64_t >(config_.include_lgan_edges)
   );
   builder.set_graph_attr(std::string(kRelationNamesAttr), relation_names_);
   builder.set_graph_attr(std::string(kRelationAritiesAttr), relation_arities_);
   builder.set_graph_attr(std::string(kRelationSourcesAttr), relation_sources_);
   builder.set_graph_attr(std::string(kTargetEntityGroupsAttr), target_entity_group_names_);
   builder.set_graph_attr(std::string(kLGANTNEdgePosAttr), config_.lgan_tn_edge_pos);
   builder.set_graph_attr(std::string(kLGANNNEdgePosAttr), config_.lgan_nn_edge_pos);
   builder.set_graph_attr(std::string(kLGANRREdgePosAttr), config_.lgan_rr_edge_pos);

   builder.register_field(
      std::string(kNodeSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kObjectSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kObjectIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::NODE_OFFSET,
            .node_type = std::string(kEntityNodeType),
         },
      }
   );
   builder.register_field(
      std::string(kTargetEntitySizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kTargetEntityIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::NODE_OFFSET,
            .node_type = std::string(kEntityNodeType),
         },
      }
   );
   builder.register_field(
      std::string(kTargetEntityGroupIdsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kTargetSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   const TargetMetadataEmitConfig target_emit_config{
      .position_node_type_id = std::string(kEntityNodeType),
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

   builder.register_field(
      std::string(kRelationCountsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = static_cast< int >(relation_names_.size()),
      }
   );
   builder.register_field(
      std::string(kRelationInstanceSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kRelationArgsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::NODE_OFFSET,
            .node_type = std::string(kEntityNodeType),
         },
      }
   );
   if(config_.include_lgan_edges) {
      builder.register_field(
         std::string(kLGANTNSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
      builder.register_field(
         std::string(kLGANTNRelationIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::FIELD_OFFSET,
               .field_key = std::string(kRelationInstanceSizesField),
            },
         }
      );
      builder.register_field(
         std::string(kLGANTNEntityIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::NODE_OFFSET,
               .node_type = std::string(kEntityNodeType),
            },
         }
      );
      builder.register_field(
         std::string(kLGANNNSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
      builder.register_field(
         std::string(kLGANNNRelationIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::FIELD_OFFSET,
               .field_key = std::string(kRelationInstanceSizesField),
            },
         }
      );
      builder.register_field(
         std::string(kLGANNNEntityIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::NODE_OFFSET,
               .node_type = std::string(kEntityNodeType),
            },
         }
      );
      builder.register_field(
         std::string(kLGANRRSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
      builder.register_field(
         std::string(kLGANRRSrcRelationIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::FIELD_OFFSET,
               .field_key = std::string(kRelationInstanceSizesField),
            },
         }
      );
      builder.register_field(
         std::string(kLGANRRDstRelationIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::FIELD_OFFSET,
               .field_key = std::string(kRelationInstanceSizesField),
            },
         }
      );
   }
}

FlatHorizonEncoderEngine::EncodingContext FlatHorizonEncoderEngine::make_context(
   const mimir::search::State& root,
   const TransitionDAG& dag
) const
{
   EncodingContext context;
   const auto& objects = root.get_problem().get_problem_and_domain_objects();
   std::vector< mimir::formalism::Object > ordered(objects.begin(), objects.end());
   std::ranges::sort(ordered, [](const auto& lhs, const auto& rhs) {
      return lhs->get_index() < rhs->get_index();
   });

   const auto& nodes = dag.nodes();
   context.entity_names.reserve(ordered.size() + nodes.size());
   context.object_names.reserve(ordered.size());
   context.object_indices.reserve(ordered.size());
   context.entity_index_by_object_id.reserve(ordered.size());
   context.state_entity_index_by_node_index.reserve(nodes.size());
   context.target_entity_indices.reserve(
      (not root_in_target_metadata(config_.root_policy) and not nodes.empty()) ? nodes.size() - 1
                                                                               : nodes.size()
   );
   context.target_entity_group_ids.reserve(context.target_entity_indices.capacity());
   context.target_name_states.reserve(nodes.size());
   context.target_columns.reserve(
      (not root_in_target_metadata(config_.root_policy) and not nodes.empty()) ? nodes.size() - 1
                                                                               : nodes.size(),
      /*include_depth=*/true,
      /*include_group=*/true
   );

   for(size_t idx = 0; idx < ordered.size(); ++idx) {
      const auto& obj = ordered[idx];
      const int64_t local_index = static_cast< int64_t >(idx);
      context.entity_index_by_object_id.emplace(
         static_cast< int64_t >(obj->get_index()), local_index
      );
      const std::string object_name = RelationFormatter::format_object(*obj);
      context.entity_names.push_back(object_name);
      context.object_names.push_back(object_name);
      context.object_indices.push_back(local_index);
   }

   hash_map< int64_t, int64_t > target_positions_by_index;
   target_positions_by_index.reserve(nodes.size());
   for(const auto& node : nodes) {
      const int64_t local_index = static_cast< int64_t >(context.entity_names.size());
      const bool include_in_target_metadata = not(
         not root_in_target_metadata(config_.root_policy) and node.index == dag.root_index()
      );
      const bool include_in_public_carrier = root_in_public_carrier(config_.root_policy)
                                             || node.index != dag.root_index();
      const std::string node_name = include_in_public_carrier ? target_node_name(node.index)
                                                              : std::string(kHiddenRootCarrierName);
      context.entity_names.push_back(node_name);
      context.state_entity_index_by_node_index.emplace(node.index, local_index);
      if(include_in_target_metadata) {
         context.target_entity_indices.push_back(local_index);
         context.target_entity_group_ids.push_back(0);
         target_positions_by_index.emplace(node.index, local_index);
      }
   }

   const auto rows = collect_transition_dag_target_candidate_rows(
      dag,
      target_positions_by_index,
      config_.root_policy,
      int64_t{0},
      /*include_names=*/false
   );
   append_target_candidate_rows(
      context.target_columns,
      rows,
      TargetCandidateAppendConfig{
         .include_depth = true,
         .include_group = true,
         .missing_candidate_id_prefix = "missing candidate_id for target node index ",
         .duplicate_candidate_id_prefix = "duplicate candidate_id ",
      }
   );
   if(config_.export_node_names) {
      for(const auto& node : nodes) {
         if(not root_in_target_metadata(config_.root_policy) and node.index == dag.root_index()) {
            continue;
         }
         if(not target_positions_by_index.contains(node.index)) {
            continue;
         }
         context.target_name_states.push_back(node.state);
      }
   }

   return context;
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

std::string FlatHorizonEncoderEngine::target_node_name(int idx) const
{
   return fmt::format("{}{}", config_.target_symbol_prefix, idx);
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
   if(prepare_builder_once) {
      ScopedProfileTimer timer(profile != nullptr ? &profile->prepare_builder_s : nullptr);
      prepare_builder(builder);
   }
   const auto context = [&]() {
      ScopedProfileTimer timer(profile != nullptr ? &profile->make_context_s : nullptr);
      return make_context(root, dag);
   }();
   FlatRelationSink sink(relation_names_.size(), config_.include_lgan_edges);

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
            state_entity_index.has_value() && root_uses_split_state_relations(config_.root_policy)
               ? state_anchored_relation_name(base_relation_name)
               : base_relation_name
         );
         const auto args = local_arg_rows_for_atom(context, atom, state_entity_index);
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
            if(state_entity_index.has_value()
               && root_uses_split_state_relations(config_.root_policy)) {
               relation_name = state_anchored_relation_name(relation_name);
            }
            const auto relation_id = relation_id_for(relation_name);
            const auto args = local_arg_rows_for_atom(
               context, literal->get_atom(), state_entity_index
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
            if(state_entity_index.has_value()
               && root_uses_split_state_relations(config_.root_policy)) {
               relation_name = state_anchored_relation_name(relation_name);
            }
            const auto relation_id = relation_id_for(relation_name);
            const auto args = local_arg_rows_for_atom(
               context, literal->get_atom(), state_entity_index
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
         if(root_uses_split_state_relations(config_.root_policy)) {
            relation_name = state_anchored_relation_name(relation_name);
         }
         const auto relation_id = relation_id_for(relation_name);
         const auto args = local_arg_rows_for_atom(context, atom, state_entity_index);
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
            std::optional< GoalDerivation > sat = std::nullopt;
            if(added_match == goal->get_polarity()) {
               sat = GoalDerivation::added_satisfied;
            } else if(removed_match != goal->get_polarity()) {
               sat = GoalDerivation::added_unsatisfied;
            }
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
            if(root_uses_split_state_relations(config_.root_policy)) {
               relation_name = state_anchored_relation_name(relation_name);
            }
            const auto relation_id = relation_id_for(relation_name);
            const auto args = local_arg_rows_for_atom(context, atom, state_entity_index);
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

   const auto root_fact_keys = [&]() {
      ScopedProfileTimer timer(profile != nullptr ? &profile->root_emit_s : nullptr);
      const auto fact_keys = emit_state_for_candidate(
         root,
         dag.root_index(),
         config_.include_static,
         /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
      );
      if(includes_plain_goal_derivation(config_.goal_derivations)) {
         emit_goal_literals.template operator()< mimir::formalism::StaticTag >(
            std::span{goals.static_goals},
            goals.static_goal_levels,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
         emit_goal_literals.template operator()< mimir::formalism::FluentTag >(
            std::span{goals.fluent_goals},
            goals.fluent_goal_levels,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
         emit_goal_literals.template operator()< mimir::formalism::DerivedTag >(
            std::span{goals.derived_goals},
            goals.derived_goal_levels,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
      }
      if(has_non_plain_goal_derivations(config_.goal_derivations)) {
         emit_goal_satisfaction.template operator()< mimir::formalism::StaticTag >(
            std::span{goals.static_goals},
            goals.static_goal_levels,
            fact_keys,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
         emit_goal_satisfaction.template operator()< mimir::formalism::FluentTag >(
            std::span{goals.fluent_goals},
            goals.fluent_goal_levels,
            fact_keys,
            dag.root_index(),
            /*include_state_anchor=*/root_in_state_relations(config_.root_policy)
         );
         emit_goal_satisfaction.template operator()< mimir::formalism::DerivedTag >(
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
               emit_delta_goal_satisfaction.template operator()< mimir::formalism::StaticTag >(
                  std::span{goals.static_goals},
                  goals.static_goal_levels,
                  hash_set< int >{},
                  hash_set< int >{},
                  node.index
               );
               emit_delta_goal_satisfaction.template operator()< mimir::formalism::FluentTag >(
                  std::span{goals.fluent_goals},
                  goals.fluent_goal_levels,
                  added_fluents,
                  removed_fluents,
                  node.index
               );
               emit_delta_goal_satisfaction.template operator()< mimir::formalism::DerivedTag >(
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

   {
      ScopedProfileTimer timer(profile != nullptr ? &profile->topology_relations_s : nullptr);
      if(config_.enable_parent_relation) {
         const int relation_id = relation_id_for(config_.parent_relation);
         for(const auto& [parent_idx, child_idx] : dag.transitions()) {
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
            for(auto& [_, children] : parent_to_children) {
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
            for(const auto& [_, parents] : parent_to_children) {
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
      std::vector< float > zeros(context.entity_names.size(), 0.0f);
      builder.add_node_features(
         std::string(kEntityNodeType), "x", std::span< const float >(zeros.data(), zeros.size()), 1
      );
      if(config_.export_node_names) {
         builder.set_node_names(std::string(kEntityNodeType), context.entity_names);
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
         .position_node_type_id = std::string(kEntityNodeType),
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
               builder.add_lazy_target_names(std::span(context.target_name_states));
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
      const auto& root_entry = inputs.roots.states[idx];
      const auto& dag_entry = inputs.dags.at(idx);
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);

      GoalInputs goal_inputs;
      {
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
         ScopedProfileTimer timer(profile.enabled ? &profile.default_dag_s : nullptr);
         default_dag.emplace(root_entry.state);
         dag_ptr = &(*default_dag);
      }
      {
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

   if(config_.export_node_names) {
      if(batch_target_name_states.empty()) {
         builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
      } else {
         builder.add_lazy_target_names(std::span(batch_target_name_states));
      }
   }
   builder.set_graph_attr(std::string(kTargetGroupsAttr), target_metadata_group_names_);
   builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   builder.set_graph_attr(std::string(kParentRelationAttr), config_.parent_relation);
   print_flat_horizon_batch_profile(profile);
   return builder.build();
}

}  // namespace mifrost
