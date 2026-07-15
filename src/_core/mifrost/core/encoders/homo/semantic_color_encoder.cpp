#include "semantic_color_encoder.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <array>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mifrost {
namespace {

constexpr std::array< std::string_view, 4 > kGoalSuffixes = {"[g]", "[sg]", "[ssg]", "[sssg]"};

struct Buffers {
   hash_map< std::string, int64_t > node_indices;
   std::vector< std::string > node_names;
   std::vector< float > node_colors;
   std::vector< int64_t > edge_src;
   std::vector< int64_t > edge_dst;
   std::vector< float > edge_colors;
   hash_map< std::string, int > colormap;
   hash_set< std::string > predicate_self_edges;
   int next_color = 1;
};

std::string atom_name(
   const SemanticAtom& atom,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< std::string >& objects
)
{
   const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
   std::vector< std::string > arguments;
   arguments.reserve(atom.arguments.size());
   for(const auto argument : atom.arguments) {
      arguments.push_back(objects.at(static_cast< size_t >(argument)));
   }
   if(arguments.empty()) {
      return fmt::format("({})", predicate.name);
   }
   return fmt::format("({} {})", predicate.name, fmt::join(arguments, " "));
}

}  // namespace

SemanticColorEncoderEngine::SemanticColorEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   SemanticColorEncoderConfig config
)
    : predicates_(std::move(predicates)), config_(config)
{
}

void SemanticColorEncoderEngine::encode(
   const SemanticFlatRelationInput& input,
   BatchBuilder& builder
) const
{
   if(not input.actions.empty()) {
      throw std::invalid_argument("SemanticColorEncoderEngine does not support actions");
   }
   if(input.subgoal_layers.size() > 3) {
      throw std::invalid_argument("Semantic color supports at most three subgoal layers");
   }
   builder.set_graph_kind("homo");
   builder.set_schema_flag("edge_features", config_.edge_features);
   builder.set_schema_flag("predicate_nodes", config_.enable_global_predicate_nodes);
   Buffers buffers;

   const auto color_for = [&](const std::string& key) {
      const auto [it, inserted] = buffers.colormap.try_emplace(key, buffers.next_color);
      if(inserted) {
         ++buffers.next_color;
      }
      return it->second;
   };
   const auto ensure_node = [&](const std::string& name, std::optional< float > color) {
      const auto [it, inserted] = buffers.node_indices.try_emplace(
         name, static_cast< int64_t >(buffers.node_names.size())
      );
      if(inserted) {
         buffers.node_names.push_back(name);
         if(not config_.edge_features) {
            buffers.node_colors.push_back(color.value_or(0.0F));
         }
      } else if(not config_.edge_features and color) {
         buffers.node_colors.at(static_cast< size_t >(it->second)) = *color;
      }
      return it->second;
   };
   const auto add_edge = [&](int64_t src, int64_t dst, std::optional< float > color) {
      buffers.edge_src.push_back(src);
      buffers.edge_dst.push_back(dst);
      if(config_.edge_features) {
         buffers.edge_colors.push_back(color.value_or(0.0F));
      }
   };

   const auto encode_atom =
      [&](const SemanticAtom& atom, std::optional< size_t > goal_level, bool positive) {
         const auto& predicate = predicates_.at(static_cast< size_t >(atom.predicate));
         const std::string prefix = goal_level ? (positive ? "[+]" : "[-]") : "";
         const std::string suffix = goal_level ? std::string(kGoalSuffixes.at(*goal_level)) : "";
         const std::string predicate_name = prefix + predicate.name + suffix;
         std::optional< int64_t > predicate_idx;
         if(config_.enable_global_predicate_nodes) {
            predicate_idx = ensure_node(predicate_name, 0.0F);
            if(config_.edge_features
               and buffers.predicate_self_edges.insert(predicate_name).second) {
               add_edge(
                  *predicate_idx, *predicate_idx, static_cast< float >(color_for(predicate_name))
               );
            }
         }
         const std::string base = prefix + atom_name(atom, predicates_, input.objects) + suffix;
         if(atom.arguments.empty()) {
            const auto color = static_cast< float >(color_for(base));
            const auto index = ensure_node(
               base, config_.edge_features ? std::nullopt : std::optional{color}
            );
            if(config_.edge_features) {
               add_edge(index, index, color);
            }
            return;
         }
         std::optional< int64_t > previous;
         for(size_t position = 0; position < atom.arguments.size(); ++position) {
            const auto object = input.objects.at(static_cast< size_t >(atom.arguments[position]));
            const auto position_name = fmt::format("{}:{}", base, position);
            const auto color = static_cast< float >(color_for(position_name));
            if(config_.edge_features) {
               const auto object_idx = ensure_node(object, std::nullopt);
               const auto item_idx = ensure_node(base, std::nullopt);
               add_edge(object_idx, item_idx, color);
               if(predicate_idx and position == 0) {
                  add_edge(*predicate_idx, item_idx, static_cast< float >(color_for(base)));
               }
            } else {
               const auto object_idx = ensure_node(object, 0.0F);
               const auto item_idx = ensure_node(position_name, color);
               add_edge(object_idx, item_idx, std::nullopt);
               if(previous) {
                  add_edge(*previous, item_idx, std::nullopt);
               } else if(predicate_idx) {
                  add_edge(*predicate_idx, item_idx, std::nullopt);
               }
               previous = item_idx;
            }
         }
      };

   for(const auto category : {
          SemanticPredicateCategory::static_predicate,
          SemanticPredicateCategory::fluent,
          SemanticPredicateCategory::derived,
       }) {
      for(const auto& atom : input.state_facts) {
         if(predicates_.at(static_cast< size_t >(atom.predicate)).category == category) {
            encode_atom(atom, std::nullopt, true);
         }
      }
   }
   std::map< SemanticLiteral, size_t > levels;
   for(const auto& literal : input.goals) {
      levels[literal] = 0;
   }
   for(size_t level = 0; level < input.subgoal_layers.size(); ++level) {
      for(const auto& literal : input.subgoal_layers[level]) {
         levels[literal] = level + 1;
      }
   }
   for(const auto category : {
          SemanticPredicateCategory::static_predicate,
          SemanticPredicateCategory::fluent,
          SemanticPredicateCategory::derived,
       }) {
      const auto encode_category_literal = [&](const SemanticLiteral& literal) {
         if(predicates_.at(static_cast< size_t >(literal.atom.predicate)).category == category) {
            encode_atom(literal.atom, levels.at(literal), literal.positive);
         }
      };
      for(const auto& literal : input.goals) {
         encode_category_literal(literal);
      }
      for(const auto& layer : input.subgoal_layers) {
         for(const auto& literal : layer) {
            encode_category_literal(literal);
         }
      }
   }

   static const std::string node_type = "node";
   builder.add_nodes(node_type, static_cast< int64_t >(buffers.node_names.size()));
   builder.set_node_names(node_type, buffers.node_names);
   if(not config_.edge_features) {
      builder.add_node_features(node_type, "x", std::span{buffers.node_colors}, 1);
   }
   if(not buffers.edge_src.empty()) {
      builder.add_edges(
         node_type, "edge", node_type, std::span{buffers.edge_src}, std::span{buffers.edge_dst}
      );
      if(config_.edge_features) {
         builder.add_edge_features(
            node_type, "edge", node_type, "edge_attr", std::span{buffers.edge_colors}, 1
         );
      }
   }
}

BatchBuilder::BatchEncoding SemanticColorEncoderEngine::encode(
   const SemanticFlatRelationInput& input
) const
{
   BatchBuilder builder;
   encode(input, builder);
   builder.next_graph();
   return builder.build();
}

BatchBuilder::BatchEncoding SemanticColorEncoderEngine::encode_batch(
   const std::vector< SemanticFlatRelationInput >& inputs
) const
{
   BatchBuilder builder;
   for(const auto& input : inputs) {
      encode(input, builder);
      builder.next_graph();
   }
   return builder.build();
}

const SemanticColorEncoderConfig& SemanticColorEncoderEngine::get_config() const
{
   return config_;
}
const std::vector< SemanticPredicateSpec >& SemanticColorEncoderEngine::get_predicates() const
{
   return predicates_;
}

}  // namespace mifrost
