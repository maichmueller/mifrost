#include "successor_hgraph_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>

namespace mifrost {

namespace {

SuccessorHGraphEncoderEngine::Config normalize_successor_config(
   SuccessorHGraphEncoderEngine::Config config
)
{
   if(config.successor_mode == SuccessorHGraphEncoderEngine::Mode::Delta
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
   auto& workspace = init_hetero_workspace(builder);

   // 1. Encode objects
   encode_objects(current, builder, workspace.node_indices, workspace.node_names);

   // 2. Encode current facts
   const auto cur_fact_keys = encode_facts(
      current,
      builder,
      workspace.node_indices,
      workspace.node_names,
      workspace.relation_to_symbols,
      workspace.symbol_to_relations
   );

   // 3. Encode successor facts
   hash_set< std::string > suc_fact_keys;
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
      const std::string atom_str = RelationFormatter::format_atom< Tag >(
         atom, successor_config_.successor_suffix
      );
      const std::string node_key = polarity.has_value()
                                      ? fmt::format(
                                           "{}{}",
                                           RelationFormatter::polarity_prefix(*polarity),
                                           atom_str
                                        )
                                      : atom_str;
      const auto relation_idx = get_or_add_node(
         node_type,
         node_key,
         builder,
         workspace.node_indices,
         workspace.node_names,
         config_.export_node_names
      );

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         object_keys.emplace_back(config_.nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(symbol_node_key(obj));
         }
      }

      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto& obj_key = object_keys[pos];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id,
            obj_key,
            builder,
            workspace.node_indices,
            workspace.node_names,
            config_.export_node_names
         );
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      track_relation_symbols_if_enabled(
         rel_key,
         std::span{object_keys},
         {},
         workspace.relation_to_symbols,
         workspace.symbol_to_relations
      );

      if(not polarity.has_value() || *polarity) {
         suc_fact_keys.insert(RelationFormatter::format_atom< Tag >(atom));
      }
   };

   if(successor_config_.successor_mode == Mode::Full) {
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

   // 4. Encode goals for current (always)
   encode_goal_inputs(goals, builder, workspace);

   if(successor_config_.successor_mode == Mode::Full) {
      // 5. Encode goal satisfaction for current
      encode_goal_satisfaction_inputs(goals, cur_fact_keys, builder, workspace);

      if(successor_config_.include_successor_goal_satisfaction) {
         encode_goal_satisfaction_inputs(
            goals, suc_fact_keys, builder, workspace, successor_config_.successor_suffix
         );
      }
   }

   // 7. LGAN edges
   maybe_add_lgan_edges(builder, workspace);
   finalize_hetero_encoding(builder, workspace);
}

}  // namespace mifrost
