#include "semantic_color_encoder.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mifrost {
namespace {

const std::shared_ptr< const SemanticTaskContext >& require_task_context(
   const std::shared_ptr< const SemanticTaskContext >& task_context
)
{
   if(not task_context) {
      throw std::invalid_argument("Semantic color task context must not be null");
   }
   return task_context;
}

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

/** Compact node identity used when display-name export is disabled. */
enum class CompactColorNodeKind : uint8_t {
   object,
   predicate,
   atom,
   position,
};

struct CompactColorNodeKey {
   CompactColorNodeKind kind = CompactColorNodeKind::object;
   int64_t object = -1;
   int64_t predicate = -1;
   SemanticAtom atom;
   int64_t goal_level = -1;
   bool positive = true;
   int64_t position = -1;

   auto operator<=>(const CompactColorNodeKey&) const = default;
};

struct CompactColorNodeKeyHash {
   size_t operator()(const CompactColorNodeKey& key) const noexcept
   {
      size_t value = static_cast< size_t >(key.kind);
      const auto mix = [&value](int64_t part) {
         value ^= std::hash< int64_t >{}(part) + 0x9e3779b97f4a7c15ULL + (value << 6U)
                  + (value >> 2U);
      };
      mix(key.object);
      mix(key.predicate);
      mix(key.atom.predicate);
      for(const auto argument : key.atom.arguments) {
         mix(argument);
      }
      mix(key.goal_level);
      mix(key.positive ? 1 : 0);
      mix(key.position);
      return value;
   }
};

struct CompactColorBuffers {
   hash_map< CompactColorNodeKey, int64_t, CompactColorNodeKeyHash > node_indices;
   hash_map< CompactColorNodeKey, int, CompactColorNodeKeyHash > colormap;
   hash_set< CompactColorNodeKey, CompactColorNodeKeyHash > predicate_self_edges;
   std::vector< float > node_colors;
   std::vector< int64_t > edge_src;
   std::vector< int64_t > edge_dst;
   std::vector< float > edge_colors;
   int64_t node_count = 0;
   int next_color = 1;
};

template < typename Input >
void encode_without_names(
   const Input& input,
   const std::vector< SemanticPredicateSpec >& predicates,
   const SemanticColorEncoderConfig& config,
   BatchBuilder& builder
)
{
   builder.set_graph_kind("homo");
   builder.set_schema_flag("edge_features", config.edge_features);
   builder.set_schema_flag("predicate_nodes", config.enable_global_predicate_nodes);
   CompactColorBuffers buffers;
   const auto& static_facts = semantic_static_facts(input);
   const auto& goals = semantic_goals(input);

   const auto color_for = [&](const CompactColorNodeKey& key) {
      const auto [it, inserted] = buffers.colormap.try_emplace(key, buffers.next_color);
      if(inserted) {
         ++buffers.next_color;
      }
      return it->second;
   };
   const auto ensure_node = [&](const CompactColorNodeKey& key, std::optional< float > color) {
      const auto [it, inserted] = buffers.node_indices.try_emplace(key, buffers.node_count);
      if(inserted) {
         ++buffers.node_count;
         if(not config.edge_features) {
            buffers.node_colors.push_back(color.value_or(0.0F));
         }
      } else if(not config.edge_features and color) {
         buffers.node_colors.at(static_cast< size_t >(it->second)) = *color;
      }
      return it->second;
   };
   const auto add_edge = [&](int64_t src, int64_t dst, std::optional< float > color) {
      buffers.edge_src.push_back(src);
      buffers.edge_dst.push_back(dst);
      if(config.edge_features) {
         buffers.edge_colors.push_back(color.value_or(0.0F));
      }
   };
   const auto encode_atom =
      [&](const SemanticAtom& atom, std::optional< size_t > goal_level, bool positive) {
         const auto predicate = atom.predicate;
         const auto level = goal_level ? static_cast< int64_t >(*goal_level) : int64_t{-1};
         const CompactColorNodeKey predicate_key{
            .kind = CompactColorNodeKind::predicate,
            .predicate = predicate,
            .goal_level = level,
            .positive = positive,
         };
         const CompactColorNodeKey atom_key{
            .kind = CompactColorNodeKind::atom,
            .predicate = predicate,
            .atom = atom,
            .goal_level = level,
            .positive = positive,
         };
         std::optional< int64_t > predicate_idx;
         if(config.enable_global_predicate_nodes) {
            predicate_idx = ensure_node(predicate_key, 0.0F);
            if(config.edge_features and buffers.predicate_self_edges.insert(predicate_key).second) {
               add_edge(
                  *predicate_idx, *predicate_idx, static_cast< float >(color_for(predicate_key))
               );
            }
         }
         if(atom.arguments.empty()) {
            const auto color = static_cast< float >(color_for(atom_key));
            const auto index = ensure_node(
               atom_key, config.edge_features ? std::nullopt : std::optional{color}
            );
            if(config.edge_features) {
               add_edge(index, index, color);
            }
            return;
         }
         std::optional< int64_t > previous;
         for(size_t position = 0; position < atom.arguments.size(); ++position) {
            const CompactColorNodeKey object_key{
               .kind = CompactColorNodeKind::object,
               .object = atom.arguments[position],
            };
            const CompactColorNodeKey position_key{
               .kind = CompactColorNodeKind::position,
               .predicate = predicate,
               .atom = atom,
               .goal_level = level,
               .positive = positive,
               .position = static_cast< int64_t >(position),
            };
            const auto color = static_cast< float >(color_for(position_key));
            if(config.edge_features) {
               const auto object_idx = ensure_node(object_key, std::nullopt);
               const auto item_idx = ensure_node(atom_key, std::nullopt);
               add_edge(object_idx, item_idx, color);
               if(predicate_idx and position == 0) {
                  add_edge(*predicate_idx, item_idx, static_cast< float >(color_for(atom_key)));
               }
            } else {
               const auto object_idx = ensure_node(object_key, 0.0F);
               const auto item_idx = ensure_node(position_key, color);
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
      const auto encode_facts = [&](const std::vector< SemanticAtom >& facts) {
         for(const auto& atom : facts) {
            if(predicates.at(static_cast< size_t >(atom.predicate)).category == category) {
               encode_atom(atom, std::nullopt, true);
            }
         }
      };
      encode_facts(static_facts);
      encode_facts(input.state_facts);
   }

   const auto levels = semantic_goal_levels(input);
   for(const auto category : {
          SemanticPredicateCategory::static_predicate,
          SemanticPredicateCategory::fluent,
          SemanticPredicateCategory::derived,
       }) {
      const auto encode_category_literal = [&](const SemanticLiteral& literal) {
         if(predicates.at(static_cast< size_t >(literal.atom.predicate)).category == category) {
            encode_atom(literal.atom, semantic_goal_level(levels, literal), literal.positive);
         }
      };
      for(const auto& literal : goals) {
         encode_category_literal(literal);
      }
      for(const auto& layer : input.subgoal_layers) {
         for(const auto& literal : layer) {
            encode_category_literal(literal);
         }
      }
   }

   static const std::string node_type = "node";
   builder.add_nodes(node_type, buffers.node_count);
   if(not config.edge_features) {
      builder.add_node_features(node_type, "x", std::span{buffers.node_colors}, 1);
   }
   if(not buffers.edge_src.empty()) {
      builder.add_edges(
         node_type, "edge", node_type, std::span{buffers.edge_src}, std::span{buffers.edge_dst}
      );
      if(config.edge_features) {
         builder.add_edge_features(
            node_type, "edge", node_type, "edge_attr", std::span{buffers.edge_colors}, 1
         );
      }
   }
}

}  // namespace

SemanticColorEncoderEngine::SemanticColorEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   SemanticColorEncoderConfig config
)
    : SemanticColorEncoderEngine(
         std::make_shared< SemanticTaskContext >(
            SemanticTaskContext{.predicates = std::move(predicates)}
         ),
         config
      )
{
}

SemanticColorEncoderEngine::SemanticColorEncoderEngine(
   std::shared_ptr< const SemanticTaskContext > task_context,
   SemanticColorEncoderConfig config
)
    : task_context_(require_task_context(task_context)),
      predicates_(task_context_->predicates),
      config_(config)
{
}

template < typename Input >
void encode_impl(
   const Input& input,
   const std::vector< SemanticPredicateSpec >& predicates,
   const SemanticColorEncoderConfig& config,
   BatchBuilder& builder
)
{
   if(not input.actions.empty()) {
      throw std::invalid_argument("SemanticColorEncoderEngine does not support actions");
   }
   if(input.subgoal_layers.size() > 3) {
      throw std::invalid_argument("Semantic color supports at most three subgoal layers");
   }
   if(not config.export_node_names) {
      encode_without_names(input, predicates, config, builder);
      return;
   }
   builder.set_graph_kind("homo");
   builder.set_schema_flag("edge_features", config.edge_features);
   builder.set_schema_flag("predicate_nodes", config.enable_global_predicate_nodes);
   Buffers buffers;
   const auto& objects = semantic_objects(input);
   const auto& goals = semantic_goals(input);
   const auto& static_facts = semantic_static_facts(input);

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
         if(not config.edge_features) {
            buffers.node_colors.push_back(color.value_or(0.0F));
         }
      } else if(not config.edge_features and color) {
         buffers.node_colors.at(static_cast< size_t >(it->second)) = *color;
      }
      return it->second;
   };
   const auto add_edge = [&](int64_t src, int64_t dst, std::optional< float > color) {
      buffers.edge_src.push_back(src);
      buffers.edge_dst.push_back(dst);
      if(config.edge_features) {
         buffers.edge_colors.push_back(color.value_or(0.0F));
      }
   };

   const auto encode_atom =
      [&](const SemanticAtom& atom, std::optional< size_t > goal_level, bool positive) {
         const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
         const std::string prefix = goal_level ? (positive ? "[+]" : "[-]") : "";
         const std::string suffix = goal_level ? std::string(kGoalSuffixes.at(*goal_level)) : "";
         const std::string predicate_name = prefix + predicate.name + suffix;
         std::optional< int64_t > predicate_idx;
         if(config.enable_global_predicate_nodes) {
            predicate_idx = ensure_node(predicate_name, 0.0F);
            if(config.edge_features
               and buffers.predicate_self_edges.insert(predicate_name).second) {
               add_edge(
                  *predicate_idx, *predicate_idx, static_cast< float >(color_for(predicate_name))
               );
            }
         }
         const std::string base = prefix + atom_name(atom, predicates, objects) + suffix;
         if(atom.arguments.empty()) {
            const auto color = static_cast< float >(color_for(base));
            const auto index = ensure_node(
               base, config.edge_features ? std::nullopt : std::optional{color}
            );
            if(config.edge_features) {
               add_edge(index, index, color);
            }
            return;
         }
         std::optional< int64_t > previous;
         for(size_t position = 0; position < atom.arguments.size(); ++position) {
            const auto object = objects.at(static_cast< size_t >(atom.arguments[position]));
            const auto position_name = fmt::format("{}:{}", base, position);
            const auto color = static_cast< float >(color_for(position_name));
            if(config.edge_features) {
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
      const auto encode_facts = [&](const std::vector< SemanticAtom >& facts) {
         for(const auto& atom : facts) {
            if(predicates.at(static_cast< size_t >(atom.predicate)).category == category) {
               encode_atom(atom, std::nullopt, true);
            }
         }
      };
      encode_facts(static_facts);
      encode_facts(input.state_facts);
   }
   const auto levels = semantic_goal_levels(input);
   for(const auto category : {
          SemanticPredicateCategory::static_predicate,
          SemanticPredicateCategory::fluent,
          SemanticPredicateCategory::derived,
       }) {
      const auto encode_category_literal = [&](const SemanticLiteral& literal) {
         if(predicates.at(static_cast< size_t >(literal.atom.predicate)).category == category) {
            encode_atom(literal.atom, semantic_goal_level(levels, literal), literal.positive);
         }
      };
      for(const auto& literal : goals) {
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
   if(not config.edge_features) {
      builder.add_node_features(node_type, "x", std::span{buffers.node_colors}, 1);
   }
   if(not buffers.edge_src.empty()) {
      builder.add_edges(
         node_type, "edge", node_type, std::span{buffers.edge_src}, std::span{buffers.edge_dst}
      );
      if(config.edge_features) {
         builder.add_edge_features(
            node_type, "edge", node_type, "edge_attr", std::span{buffers.edge_colors}, 1
         );
      }
   }
}

void SemanticColorEncoderEngine::encode(
   const SemanticFlatRelationInput& input,
   BatchBuilder& builder
) const
{
   encode_impl(input, predicates_, config_, builder);
}

void SemanticColorEncoderEngine::encode(
   const SemanticFlatRelationSink& sink,
   BatchBuilder& builder
) const
{
   encode_impl(sink, predicates_, config_, builder);
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

BatchBuilder::BatchEncoding SemanticColorEncoderEngine::encode(
   const SemanticFlatRelationSink& sink
) const
{
   BatchBuilder builder;
   encode(sink, builder);
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
