/**
 * @file successor_hgraph_encoder.cpp
 * @brief Pymimir immediate-successor encoder implementation.
 */
#include "successor_hgraph_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>

#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

namespace {

SuccessorHGraphEncoderEngine::Config normalize_successor_config(
   SuccessorHGraphEncoderEngine::Config config
)
{
   if(config.successor_mode == SuccessorHGraphEncoderEngine::Mode::delta
      and not config.support_literals) {
      config.support_literals = true;
   }
   return config;
}

}  // namespace

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain
)
    : HGraphEncoderEngine(domain), successor_config_()
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : HGraphEncoderEngine(domain, normalize_successor_config(config)),
      successor_config_(normalize_successor_config(std::move(config)))
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(mimir::formalism::Domain domain)
    : HGraphEncoderEngine(domain), successor_config_()
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   mimir::formalism::Domain domain,
   Config config
)
    : HGraphEncoderEngine(domain, normalize_successor_config(config)),
      successor_config_(normalize_successor_config(std::move(config)))
{
}

void SuccessorHGraphEncoderEngine::encode(
   const mimir::search::State& current,
   const mimir::search::State& successor,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   encode_impl(current, successor, goals, builder);
}

void SuccessorHGraphEncoderEngine::encode_impl(
   const mimir::search::State& current,
   const mimir::search::State& successor,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   // Summary:
   // 1. Encode shared objects and the current-state facts.
   // 2. Encode the successor view as full facts or delta literals.
   // 3. Add goal views, helper edges, and finalize the hetero graph.
   auto& workspace = init_hetero_workspace(builder);

   // Phase 1: add the shared object nodes once for the current/successor pair.
   encode_objects(current, builder, workspace.node_indices, workspace.node_names);

   // Phase 2: encode the current-state facts and keep them for goal-satisfaction checks.
   const auto cur_fact_keys = encode_facts(
      current,
      builder,
      workspace.node_indices,
      workspace.node_names,
      workspace.relation_to_symbols,
      workspace.symbol_to_relations
   );

   // Phase 3: encode the successor view, either as a full state or as delta literals.
   hash_set< uint64_t > suc_fact_keys;
   const auto& problem = successor.get_problem();
   const auto& repos = problem.get_repositories();

   auto handle_suc_atom = [&](auto atom, std::optional< bool > polarity = std::nullopt) {
      using Tag = typename std::remove_pointer_t< decltype(atom) >::Type;
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
         return;
      }
      std::string node_type;
      if(polarity.has_value()) {
         node_type = RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, *polarity, successor_config_.successor_suffix
         );
      } else {
         node_type = RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, std::nullopt, successor_config_.successor_suffix
         );
      }
      const int64_t relation_key = static_cast< int64_t >(atom->get_index());
      std::string node_name;
      if(config_.export_node_names) {
         const std::string atom_str = RelationFormatter::format_atom< Tag >(
            atom, successor_config_.successor_suffix
         );
         node_name = polarity.has_value()
                        ? fmt::format(
                             "{}{}", RelationFormatter::polarity_prefix(*polarity), atom_str
                          )
                        : atom_str;
      }
      const auto relation_idx = get_or_add_relation_node_i64(
         node_type, relation_key, builder, workspace.node_indices, workspace.node_names, node_name
      );

      std::vector< int64_t > object_symbol_ids;
      if(predicate->get_arity() == 0) {
         const auto nullary_idx = get_or_add_symbol_special_node(
            config_.nullary_object_name, config_.nullary_object_name, builder, workspace.node_names
         );
         (void) nullary_idx;
         object_symbol_ids.emplace_back(
            get_or_assign_special_symbol_id(config_.nullary_object_name)
         );
      } else {
         for(const auto& obj : atom->get_objects()) {
            const auto obj_idx = get_or_add_symbol_object_node(obj, builder, workspace.node_names);
            (void) obj_idx;
            object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
         }
      }

      for(size_t pos = 0; pos < object_symbol_ids.size(); ++pos) {
         const auto obj_idx = workspace.symbol_indices.at(object_symbol_ids[pos]);
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const auto rel_ref = relation_ref_for(node_type, relation_idx);
      track_relation_symbols_if_enabled(
         rel_ref,
         std::span{object_symbol_ids},
         std::span< const int64_t >{},
         workspace.relation_to_symbols,
         workspace.symbol_to_relations
      );

      if(not polarity.has_value() or *polarity) {
         uint32_t tag_id = 0;
         if constexpr(std::is_same_v< Tag, mimir::formalism::StaticTag >) {
            tag_id = 1;
         } else if constexpr(std::is_same_v< Tag, mimir::formalism::FluentTag >) {
            tag_id = 2;
         } else {
            tag_id = 3;
         }
         suc_fact_keys.insert(pack_u32_u32(static_cast< uint32_t >(atom->get_index()), tag_id));
      }
   };

   if(successor_config_.successor_mode == Mode::full) {
      const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         successor.get_atoms< mimir::formalism::FluentTag >()
      );
      for(const auto& atom : fluent_atoms) {
         handle_suc_atom(atom, std::nullopt);
      }
      const auto derived_atoms = repos
                                    .get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
                                       successor.get_atoms< mimir::formalism::DerivedTag >()
                                    );
      for(const auto& atom : derived_atoms) {
         handle_suc_atom(atom, std::nullopt);
      }
   } else {
      // Delta mode
      auto handle_delta = [&](auto cur_indices, auto suc_indices, auto tag_ptr) {
         using Tag = std::remove_pointer_t< decltype(tag_ptr) >;
         std::vector< uint32_t > added;
         std::vector< uint32_t > removed;
         std::set_difference(
            suc_indices.begin(),
            suc_indices.end(),
            cur_indices.begin(),
            cur_indices.end(),
            std::back_inserter(added)
         );
         std::set_difference(
            cur_indices.begin(),
            cur_indices.end(),
            suc_indices.begin(),
            suc_indices.end(),
            std::back_inserter(removed)
         );

         for(uint32_t idx : added) {
            handle_suc_atom(repos.get_ground_atom< Tag >(idx), true);
         }
         for(uint32_t idx : removed) {
            handle_suc_atom(repos.get_ground_atom< Tag >(idx), false);
         }
      };

      handle_delta(
         current.get_atoms< mimir::formalism::FluentTag >(),
         successor.get_atoms< mimir::formalism::FluentTag >(),
         (mimir::formalism::FluentTag*) nullptr
      );
      handle_delta(
         current.get_atoms< mimir::formalism::DerivedTag >(),
         successor.get_atoms< mimir::formalism::DerivedTag >(),
         (mimir::formalism::DerivedTag*) nullptr
      );
   }

   // Phase 4: goal literals are always attached to the current state.
   encode_goal_inputs(goals, builder, workspace);

   if(successor_config_.successor_mode == Mode::full) {
      // Phase 5: optionally add goal satisfaction for the current and successor states.
      encode_goal_satisfaction_inputs(goals, cur_fact_keys, builder, workspace);

      if(successor_config_.include_successor_goal_satisfaction) {
         encode_goal_satisfaction_inputs(
            goals, suc_fact_keys, builder, workspace, successor_config_.successor_suffix
         );
      }
   }

   // Phase 6: derive helper edges and finalize the graph-level metadata.
   maybe_add_lgan_edges(builder, workspace);
   finalize_hetero_encoding(builder, workspace);
}

BatchBuilder::BatchEncoding SuccessorHGraphEncoderEngine::encode_batch(
   const batch_input::parsed::SuccessorBatchInputs& inputs
)
{
   // Summary:
   // 1. Read one current/successor pair at a time and normalize goals.
   // 2. Encode the pair into the shared hetero batch builder.
   // 3. Advance the builder to the next graph slot.
   BatchBuilder builder;
   builder.set_graph_kind("hetero");

   const size_t state_count = inputs.states.states.size();
   for(size_t idx = 0; idx < state_count; ++idx) {
      // Phase 1: collect the current state, successor state, and optional goal payloads.
      const auto& state_entry = inputs.states.states[idx];
      const auto& successor_entry = inputs.successors.at(idx);
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);

      // Phase 2: normalize goals into one GoalInputs object.
      GoalInputs goal_inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         goal_inputs = batch_input::compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         goal_inputs = batch_input::default_goal_inputs_for_batch_state(state_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               goal_inputs.extend(layer, level);
               ++level;
            }
         }
      }

      // Phase 3: encode one current/successor pair and move to the next graph.
      encode(state_entry.state, successor_entry->state, goal_inputs, builder);
      builder.next_graph();
   }

   return builder.build();
}

}  // namespace mifrost
