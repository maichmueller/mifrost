/**
 * @file color_encoder.cpp
 * @brief Homogeneous color encoder implementation.
 */
#include "color_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <mimir/formalism/problem.hpp>
#include <stdexcept>

#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

namespace {
struct ColorBuffers {
   hash_map< std::string, int64_t > node_indices;
   std::vector< std::string > node_names;
   std::vector< float > node_colors;
   std::vector< int64_t > edge_src;
   std::vector< int64_t > edge_dst;
   std::vector< float > edge_colors;
   hash_map< std::string, int > colormap;
   int next_color = 1;
   hash_set< std::string > predicate_self_edges;
};

template < typename MapT, typename KeyT >
std::optional< GoalLevel > find_goal_level(const MapT& map, const KeyT& key)
{
   auto it = map.find(key);
   if(it != map.end()) {
      return GoalLevel{it->second};
   }
   return std::nullopt;
}

}  // namespace

ColorEncoderEngine::ColorEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : ColorEncoderEngine(domain, Config{})
{
}

ColorEncoderEngine::ColorEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config)
    : domain_(domain), config_(std::move(config))
{
}

ColorEncoderEngine::ColorEncoderEngine(mimir::formalism::Domain domain)
    : ColorEncoderEngine(std::move(domain), Config{})
{
}

ColorEncoderEngine::ColorEncoderEngine(mimir::formalism::Domain domain, Config config)
    : domain_holder_(std::move(domain)), domain_(**domain_holder_), config_(std::move(config))
{
}

void ColorEncoderEngine::encode_state_impl(const mimir::search::State& state, BatchBuilder& builder)
{
   GoalInputs inputs;
   const auto& problem = state.get_problem();
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::StaticTag >()) {
      inputs.static_goals.emplace_back(goal);
      inputs.static_goal_levels[goal] = 0;
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::FluentTag >()) {
      inputs.fluent_goals.emplace_back(goal);
      inputs.fluent_goal_levels[goal] = 0;
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::DerivedTag >()) {
      inputs.derived_goals.emplace_back(goal);
      inputs.derived_goal_levels[goal] = 0;
   }
   encode_impl(state, inputs, {}, builder);
}

void ColorEncoderEngine::encode_impl(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   if(not actions.empty()) {
      throw std::invalid_argument("ColorEncoderEngine does not support action encoding");
   }

   // Summary:
   // 1. Build the local color map and the node and edge buffers.
   // 2. Encode state facts first, then goal literals.
   // 3. Flush the buffered nodes, edges, and features into the homo builder.
   // Phase 1: start one homo graph and set the schema flags used by downstream code.
   builder.set_graph_kind("homo");
   builder.set_schema_flag("edge_features", config_.edge_features);
   builder.set_schema_flag("predicate_nodes", config_.enable_global_predicate_nodes);

   ColorBuffers buffers;

   auto color_for = [&](const std::string& key) {
      auto [it, inserted] = buffers.colormap.try_emplace(key, buffers.next_color);
      if(inserted) {
         ++buffers.next_color;
      }
      return it->second;
   };

   auto ensure_node = [&](const std::string& name, std::optional< float > color) {
      auto [it, inserted] = buffers.node_indices.try_emplace(
         name, static_cast< int64_t >(buffers.node_names.size())
      );
      if(inserted) {
         buffers.node_names.push_back(name);
         if(not config_.edge_features) {
            buffers.node_colors.push_back(color.value_or(0.0f));
         }
      } else if(not config_.edge_features and color.has_value()) {
         buffers.node_colors[it->second] = *color;
      }
      return it->second;
   };

   auto add_edge = [&](int64_t src, int64_t dst, std::optional< float > color) {
      buffers.edge_src.push_back(src);
      buffers.edge_dst.push_back(dst);
      if(config_.edge_features) {
         buffers.edge_colors.push_back(color.value_or(0.0f));
      }
   };

   auto add_predicate_self_edge = [&](const std::string& predicate_node, int64_t idx) {
      if(not config_.edge_features) {
         return;
      }
      if(buffers.predicate_self_edges.contains(predicate_node)) {
         return;
      }
      buffers.predicate_self_edges.insert(predicate_node);
      add_edge(idx, idx, static_cast< float >(color_for(predicate_node)));
   };

   // Phase 2: define the local builders for nodes, colors, and edge patterns.
   auto encode_atom = [&](const auto& atom, std::optional< int > goal_level) {
      const auto predicate = atom->get_predicate();
      const auto arity = predicate->get_arity();

      std::optional< std::string > predicate_node;
      std::optional< int64_t > predicate_idx;
      if(config_.enable_global_predicate_nodes) {
         if(goal_level.has_value()) {
            const GoalLevel level(static_cast< std::size_t >(*goal_level));
            predicate_node = RelationFormatter::format_predicate(
               predicate, level, std::nullopt, std::nullopt, ""
            );
         } else {
            predicate_node = RelationFormatter::format_predicate(
               predicate, std::nullopt, std::nullopt, std::nullopt, ""
            );
         }
         predicate_idx = ensure_node(*predicate_node, 0.0f);
         add_predicate_self_edge(*predicate_node, *predicate_idx);
      }

      const std::string base_name = RelationFormatter::format_atom(atom);

      if(arity == 0) {
         const auto color = static_cast< float >(color_for(base_name));
         if(config_.edge_features) {
            const auto idx = ensure_node(base_name, std::nullopt);
            add_edge(idx, idx, color);
         } else {
            ensure_node(base_name, color);
         }
         return;
      }

      std::optional< int64_t > prev_item_idx;
      int pos = 0;
      for(const auto& obj : atom->get_objects()) {
         const std::string object_node = RelationFormatter::format_object(*obj);
         const std::string pos_name = fmt::format("{}:{}", base_name, pos);
         const auto color = static_cast< float >(color_for(pos_name));

         if(config_.edge_features) {
            const auto obj_idx = ensure_node(object_node, std::nullopt);
            const auto item_idx = ensure_node(base_name, std::nullopt);
            add_edge(obj_idx, item_idx, color);
            if(predicate_idx.has_value() and pos == 0) {
               add_edge(*predicate_idx, item_idx, static_cast< float >(color_for(base_name)));
            }
         } else {
            const auto obj_idx = ensure_node(object_node, 0.0f);
            const auto item_idx = ensure_node(pos_name, color);
            add_edge(obj_idx, item_idx, std::nullopt);
            if(prev_item_idx.has_value()) {
               add_edge(*prev_item_idx, item_idx, std::nullopt);
            } else if(predicate_idx.has_value()) {
               add_edge(*predicate_idx, item_idx, std::nullopt);
            }
            prev_item_idx = item_idx;
         }
         ++pos;
      }
   };

   auto encode_literal = [&](const auto& literal, std::optional< GoalLevel > goal_level) {
      const auto atom = literal->get_atom();
      const auto predicate = atom->get_predicate();
      const auto arity = predicate->get_arity();
      const bool polarity = literal->get_polarity();

      std::optional< std::string > predicate_node;
      std::optional< int64_t > predicate_idx;
      if(config_.enable_global_predicate_nodes) {
         if(goal_level.has_value()) {
            predicate_node = RelationFormatter::format_predicate(
               predicate, *goal_level, std::nullopt, polarity, ""
            );
         } else {
            predicate_node = RelationFormatter::format_predicate(
               predicate, std::nullopt, std::nullopt, polarity, ""
            );
         }
         predicate_idx = ensure_node(*predicate_node, 0.0f);
         add_predicate_self_edge(*predicate_node, *predicate_idx);
      }

      const std::string base_name = [&]() {
         if(goal_level.has_value()) {
            const GoalLevel level(*goal_level);
            return RelationFormatter::format_literal(literal, level, std::nullopt, polarity, "");
         }
         return RelationFormatter::format_literal(
            literal, std::nullopt, std::nullopt, polarity, ""
         );
      }();

      if(arity == 0) {
         const auto color = static_cast< float >(color_for(base_name));
         if(config_.edge_features) {
            const auto idx = ensure_node(base_name, std::nullopt);
            add_edge(idx, idx, color);
         } else {
            ensure_node(base_name, color);
         }
         return;
      }

      std::optional< int64_t > prev_item_idx;
      int pos = 0;
      for(const auto& obj : atom->get_objects()) {
         const std::string object_node = RelationFormatter::format_object(*obj);
         const std::string pos_name = fmt::format("{}:{}", base_name, pos);
         const auto color = static_cast< float >(color_for(pos_name));

         if(config_.edge_features) {
            const auto obj_idx = ensure_node(object_node, std::nullopt);
            const auto item_idx = ensure_node(base_name, std::nullopt);
            add_edge(obj_idx, item_idx, color);
            if(predicate_idx.has_value() and pos == 0) {
               add_edge(*predicate_idx, item_idx, static_cast< float >(color_for(base_name)));
            }
         } else {
            const auto obj_idx = ensure_node(object_node, 0.0f);
            const auto item_idx = ensure_node(pos_name, color);
            add_edge(obj_idx, item_idx, std::nullopt);
            if(prev_item_idx.has_value()) {
               add_edge(*prev_item_idx, item_idx, std::nullopt);
            } else if(predicate_idx.has_value()) {
               add_edge(*predicate_idx, item_idx, std::nullopt);
            }
            prev_item_idx = item_idx;
         }
         ++pos;
      }
   };

   // Phase 3: add state facts first, then overlay the goal literals.
   const auto& problem = state.get_problem();
   if(problem.get_initial_literals< mimir::formalism::StaticTag >().size() > 0) {
      for(const auto& literal : problem.get_initial_literals< mimir::formalism::StaticTag >()) {
         if(not literal->get_polarity()) {
            continue;
         }
         encode_atom(literal->get_atom(), std::nullopt);
      }
   }

   const auto& repos = problem.get_repositories();
   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      state.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      encode_atom(atom, std::nullopt);
   }

   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      state.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      encode_atom(atom, std::nullopt);
   }

   for(const auto& literal : goals.static_goals) {
      auto level = find_goal_level(goals.static_goal_levels, literal);
      encode_literal(literal, level);
   }
   for(const auto& literal : goals.fluent_goals) {
      auto level = find_goal_level(goals.fluent_goal_levels, literal);
      encode_literal(literal, level);
   }
   for(const auto& literal : goals.derived_goals) {
      auto level = find_goal_level(goals.derived_goal_levels, literal);
      encode_literal(literal, level);
   }

   // Phase 4: flush the buffered node and edge data into the builder.
   static const std::string node_type = "node";
   builder.add_nodes(node_type, static_cast< int64_t >(buffers.node_names.size()));
   builder.set_node_names(node_type, buffers.node_names);

   if(not config_.edge_features) {
      builder.add_node_features(
         node_type,
         "x",
         std::span< const float >(buffers.node_colors.data(), buffers.node_colors.size()),
         1
      );
   }

   if(not buffers.edge_src.empty()) {
      builder.add_edges(
         node_type,
         "edge",
         node_type,
         std::span< const int64_t >(buffers.edge_src.data(), buffers.edge_src.size()),
         std::span< const int64_t >(buffers.edge_dst.data(), buffers.edge_dst.size())
      );
      if(config_.edge_features) {
         builder.add_edge_features(
            node_type,
            "edge",
            node_type,
            "edge_attr",
            std::span< const float >(buffers.edge_colors.data(), buffers.edge_colors.size()),
            1
         );
      }
   }
}

BatchBuilder::BatchEncoding ColorEncoderEngine::encode_batch(
   const batch_input::parsed::ColorBatchInputs& inputs
)
{
   // Summary:
   // 1. Read one batch item at a time and normalize optional goals.
   // 2. Encode each item into the shared homo batch builder.
   // 3. Advance the builder to the next graph slot.
   BatchBuilder builder;
   builder.set_graph_kind("homo");

   const size_t state_count = inputs.states.states.size();
   for(size_t idx = 0; idx < state_count; ++idx) {
      // Phase 1: collect the state and any optional goal payloads for one batch item.
      const auto& state_entry = inputs.states.states[idx];
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);

      // Fast path for plain state encoding with no goal overlays.
      if(not goals_entry.has_value() and not subgoal_layers_entry.has_value()) {
         encode(state_entry.state, builder);
         builder.next_graph();
         continue;
      }

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

      // Phase 3: encode one graph and move the builder to the next graph.
      encode(state_entry.state, goal_inputs, builder);
      builder.next_graph();
   }

   return builder.build();
}

}  // namespace mifrost
