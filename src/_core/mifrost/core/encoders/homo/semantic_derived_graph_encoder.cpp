#include "semantic_derived_graph_encoder.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mifrost {
namespace {

constexpr std::array< std::string_view, 4 > kGoalSuffixes = {"[g]", "[sg]", "[ssg]", "[sssg]"};

constexpr int64_t kRoleObject = 0;
constexpr int64_t kRoleFact = 1;
constexpr int64_t kRoleGoal = 2;
constexpr int64_t kRoleSubgoal = 3;
constexpr int64_t kRoleHistory = 4;
constexpr int64_t kRoleAction = 5;

constexpr int64_t kKindArgFwd = 0;
constexpr int64_t kKindArgBwd = 1;
constexpr int64_t kKindCliqueFwd = 2;
constexpr int64_t kKindCliqueBwd = 3;
constexpr int64_t kKindChainFwd = 4;
constexpr int64_t kKindChainBwd = 5;
constexpr int64_t kKindStarFirstFwd = 6;
constexpr int64_t kKindStarFirstBwd = 7;
constexpr int64_t kKindNullarySelf = 8;
constexpr int64_t kKindActionFwd = 9;
constexpr int64_t kKindActionBwd = 10;
constexpr int64_t kKindLineShare = 11;

constexpr int kXDim = 6;
constexpr int kEdgeDim = 3;

constexpr std::string_view kHyperedgeSizesField = "hyperedge_sizes";
constexpr std::string_view kHyperedgeNodeIndicesField = "hyperedge_node_indices";
constexpr std::string_view kHyperedgeIdsField = "hyperedge_ids";
constexpr std::string_view kHyperedgeRoleIdsField = "hyperedge_role_ids";
constexpr std::string_view kHyperedgeCountsField = "hyperedge_counts";
constexpr std::string_view kTupleArgsField = "tuple_args";
constexpr std::string_view kTupleSlotSizesField = "tuple_slot_sizes";
constexpr std::string_view kTupleRelIdsField = "tuple_rel_ids";
constexpr std::string_view kTupleRoleIdsField = "tuple_role_ids";
constexpr std::string_view kTupleCountsField = "tuple_counts";
constexpr std::string_view kSpdSrcField = "spd_src";
constexpr std::string_view kSpdDstField = "spd_dst";
constexpr std::string_view kSpdDistField = "spd_dist";

constexpr int64_t kSpdMaxObjects = 4096;

const std::shared_ptr< const SemanticSchemaContext >& require_schema_context(
   const std::shared_ptr< const SemanticSchemaContext >& schema
)
{
   if(not schema) {
      throw std::invalid_argument("Semantic derived-graph schema context must not be null");
   }
   return schema;
}

void validate_config(const SemanticDerivedGraphEncoderConfig& config)
{
   const auto reified = config.node_universe == DerivedNodeUniverse::objects_and_atoms;
   if(not reified and config.atom_expansion == DerivedAtomExpansion::star) {
      throw std::invalid_argument(
         "Derived-graph star expansion requires the objects-and-atoms node universe"
      );
   }
   if(config.include_line_graph and not reified) {
      throw std::invalid_argument(
         "Derived-graph line-graph edges require the objects-and-atoms node universe"
      );
   }
   if(config.spd_max_hops < 2) {
      throw std::invalid_argument(
         "Derived-graph spd_max_hops must be >= 2: bipartite object distances advance "
         "in steps of two, so odd hop counts yield no pairs"
      );
   }
}

struct NodeKey {
   int64_t role = kRoleObject;
   int64_t predicate = -1;
   SemanticArguments arguments;
   bool positive = true;
   int64_t level = 0;
   int64_t dt = 0;

   auto operator<=>(const NodeKey&) const = default;
};

struct NodeKeyHash {
   size_t operator()(const NodeKey& key) const noexcept
   {
      size_t value = static_cast< size_t >(key.role);
      mix_semantic_hash(value, key.predicate);
      for(const auto argument : key.arguments) {
         mix_semantic_hash(value, argument);
      }
      mix_semantic_hash(value, key.positive ? 1 : 0);
      mix_semantic_hash(value, key.level);
      mix_semantic_hash(value, key.dt);
      return value;
   }
};

struct DerivedInstance {
   int64_t role = kRoleFact;
   int64_t predicate = -1;
   SemanticArguments arguments;
};

struct Buffers {
   hash_map< NodeKey, int64_t, NodeKeyHash > node_indices;
   /// Row-major [N, 6]: role, predicate_id+1, sign, goal_level, history_dt, category.
   std::vector< float > x_flat;
   std::vector< int64_t > edge_src;
   std::vector< int64_t > edge_dst;
   /// Row-major [E, 3]: kind, pos_a, pos_b.
   std::vector< float > edge_attr_flat;
   std::vector< std::string > node_names;
   /// Per-object incident (fact-node, position) pairs for line-graph derivation.
   std::vector< std::vector< std::pair< int64_t, int64_t > > > object_incidents;
   /// Encoded literal/action instances in deterministic emission order.
   std::vector< DerivedInstance > instances;

   [[nodiscard]] int64_t node_count() const
   {
      return static_cast< int64_t >(x_flat.size() / kXDim);
   }
};

struct Emitter {
   const std::vector< SemanticPredicateSpec >& predicates;
   const std::vector< SemanticActionSpec >& action_specs;
   const std::vector< std::string >& objects;
   const SemanticDerivedGraphEncoderConfig& config;
   Buffers buffers;

   void write_features(
      int64_t index,
      int64_t role,
      int64_t predicate,
      bool positive,
      int64_t level,
      int64_t dt,
      int64_t category
   )
   {
      const auto slot = static_cast< size_t >(index) * kXDim;
      buffers.x_flat.resize(std::max(buffers.x_flat.size(), slot + kXDim), 0.0F);
      buffers.x_flat[slot] = static_cast< float >(role);
      buffers.x_flat[slot + 1] = static_cast< float >(predicate + 1);
      buffers.x_flat[slot + 2] = positive ? 0.0F : 1.0F;
      buffers.x_flat[slot + 3] = static_cast< float >(level);
      buffers.x_flat[slot + 4] = static_cast< float >(dt);
      buffers.x_flat[slot + 5] = static_cast< float >(category);
   }

   [[nodiscard]] int64_t
   ensure_node(const NodeKey& key, int64_t category, std::optional< std::string > name)
   {
      const auto [it, inserted] = buffers.node_indices.try_emplace(key, buffers.node_count());
      if(inserted) {
         write_features(
            it->second, key.role, key.predicate, key.positive, key.level, key.dt, category
         );
         if(config.export_node_names) {
            buffers.node_names.emplace_back(name.value_or(std::string()));
         }
      }
      return it->second;
   }

   void add_edge(int64_t src, int64_t dst, int64_t kind, int64_t pos_a, int64_t pos_b)
   {
      buffers.edge_src.push_back(src);
      buffers.edge_dst.push_back(dst);
      buffers.edge_attr_flat.push_back(static_cast< float >(kind));
      buffers.edge_attr_flat.push_back(static_cast< float >(pos_a));
      buffers.edge_attr_flat.push_back(static_cast< float >(pos_b));
   }

   [[nodiscard]] std::string
   format_atom_name(const SemanticAtom& atom, bool positive, std::string_view suffix) const
   {
      std::vector< std::string > arguments;
      arguments.reserve(atom.arguments.size());
      for(const auto argument : atom.arguments) {
         arguments.push_back(objects.at(static_cast< size_t >(argument)));
      }
      const std::string body = arguments.empty()
                                  ? fmt::format(
                                       "({})",
                                       predicates.at(static_cast< size_t >(atom.predicate)).name
                                    )
                                  : fmt::format(
                                       "({} {})",
                                       predicates.at(static_cast< size_t >(atom.predicate)).name,
                                       fmt::join(arguments, " ")
                                    );
      return positive ? fmt::format("{}{}", body, suffix) : fmt::format("(not {}){}", body, suffix);
   }

   [[nodiscard]] std::string format_action_name(const SemanticGroundAction& action) const
   {
      std::vector< std::string > arguments;
      arguments.reserve(action.arguments.size());
      for(const auto argument : action.arguments) {
         arguments.push_back(objects.at(static_cast< size_t >(argument)));
      }
      const auto action_name = action.action >= 0
                                     && static_cast< size_t >(action.action) < action_specs.size()
                                  ? action_specs.at(static_cast< size_t >(action.action)).name
                                  : fmt::format("action{}", action.action);
      return arguments.empty() ? fmt::format("@({})", action_name)
                               : fmt::format("@({} {})", action_name, fmt::join(arguments, " "));
   }
};

GraphFieldInc node_offset_inc()
{
   static const std::string node_type = "node";
   return GraphFieldInc{
      .kind = GraphFieldInc::Kind::NODE_OFFSET,
      .node_type = node_type,
   };
}

GraphFieldInc hyperedge_offset_inc()
{
   return GraphFieldInc{
      .kind = GraphFieldInc::Kind::FIELD_OFFSET,
      .field_key = std::string(kHyperedgeCountsField),
   };
}

GraphFieldInc tuple_offset_inc()
{
   return GraphFieldInc{
      .kind = GraphFieldInc::Kind::FIELD_OFFSET,
      .field_key = std::string(kTupleCountsField),
   };
}

void register_instance_fields(
   BatchBuilder& builder,
   const SemanticDerivedGraphEncoderConfig& config
)
{
   if(config.include_hyperedge_incidence) {
      builder.register_field(
         std::string(kHyperedgeCountsField),
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
      );
      builder.register_field(
         std::string(kHyperedgeSizesField),
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::RAGGED_CAT, .dim = 1}
      );
      builder.register_field(
         std::string(kHyperedgeNodeIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = node_offset_inc(),
         }
      );
      builder.register_field(
         std::string(kHyperedgeIdsField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = hyperedge_offset_inc(),
         }
      );
      builder.register_field(
         std::string(kHyperedgeRoleIdsField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = hyperedge_offset_inc(),
         }
      );
   }
   if(config.include_tuple_tensors) {
      builder.register_field(
         std::string(kTupleCountsField),
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
      );
      builder.register_field(
         std::string(kTupleSlotSizesField),
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::RAGGED_CAT, .dim = 1}
      );
      builder.register_field(
         std::string(kTupleArgsField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = node_offset_inc(),
         }
      );
      builder.register_field(
         std::string(kTupleRelIdsField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = tuple_offset_inc(),
         }
      );
      builder.register_field(
         std::string(kTupleRoleIdsField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = tuple_offset_inc(),
         }
      );
   }
}

void set_instance_fields(
   BatchBuilder& builder,
   const SemanticDerivedGraphEncoderConfig& config,
   const std::vector< DerivedInstance >& instances
)
{
   const auto instance_count = static_cast< int64_t >(instances.size());
   if(config.include_hyperedge_incidence) {
      std::vector< int64_t > sizes;
      std::vector< int64_t > members;
      std::vector< int64_t > ids;
      std::vector< int64_t > roles;
      sizes.reserve(instances.size());
      ids.reserve(instances.size());
      roles.reserve(instances.size());
      int64_t next_id = 0;
      for(const auto& instance : instances) {
         sizes.push_back(static_cast< int64_t >(instance.arguments.size()));
         members.insert(members.end(), instance.arguments.begin(), instance.arguments.end());
         ids.push_back(next_id++);
         roles.push_back(instance.role);
      }
      builder.set_field(std::string{kHyperedgeSizesField}, sizes);
      builder.set_field(std::string{kHyperedgeNodeIndicesField}, members);
      builder.set_field(std::string{kHyperedgeIdsField}, ids);
      builder.set_field(std::string{kHyperedgeRoleIdsField}, roles);
      builder.set_field(
         std::string{kHyperedgeCountsField}, std::span< const int64_t >{&instance_count, 1}
      );
   }
   if(config.include_tuple_tensors) {
      std::vector< int64_t > args;
      std::vector< int64_t > slot_sizes;
      std::vector< int64_t > rel_ids;
      std::vector< int64_t > role_ids;
      slot_sizes.reserve(instances.size());
      rel_ids.reserve(instances.size());
      role_ids.reserve(instances.size());
      for(const auto& instance : instances) {
         args.insert(args.end(), instance.arguments.begin(), instance.arguments.end());
         slot_sizes.push_back(static_cast< int64_t >(instance.arguments.size()));
         rel_ids.push_back(instance.predicate);
         role_ids.push_back(instance.role);
      }
      builder.set_field(std::string{kTupleSlotSizesField}, slot_sizes);
      builder.set_field(std::string{kTupleArgsField}, args);
      builder.set_field(std::string{kTupleRelIdsField}, rel_ids);
      builder.set_field(std::string{kTupleRoleIdsField}, role_ids);
      builder.set_field(
         std::string{kTupleCountsField}, std::span< const int64_t >{&instance_count, 1}
      );
   }
}

void emit_spd_fields(
   BatchBuilder& builder,
   const SemanticDerivedGraphEncoderConfig& config,
   int64_t object_count,
   const std::vector< DerivedInstance >& instances
)
{
   if(object_count > kSpdMaxObjects) {
      throw std::invalid_argument("spd requires at most 4096 objects");
   }

   std::vector< std::vector< int64_t > > incident_instances;
   incident_instances.resize(static_cast< size_t >(object_count));
   for(size_t index = 0; index < instances.size(); ++index) {
      for(const auto argument : instances[index].arguments) {
         incident_instances.at(static_cast< size_t >(argument))
            .push_back(static_cast< int64_t >(index));
      }
   }

   const auto instance_count = static_cast< int64_t >(instances.size());
   std::vector< int64_t > visited_epoch(static_cast< size_t >(instance_count), -1);
   std::vector< int64_t > distance(static_cast< size_t >(object_count), -1);
   std::vector< int64_t > queue;
   queue.reserve(static_cast< size_t >(object_count));

   std::vector< int64_t > src_ids;
   std::vector< int64_t > dst_ids;
   std::vector< int64_t > distances;
   for(int64_t src = 0; src < object_count; ++src) {
      if(incident_instances.at(static_cast< size_t >(src)).empty()) {
         continue;
      }
      std::fill(distance.begin(), distance.end(), -1);
      queue.clear();
      distance.at(static_cast< size_t >(src)) = 0;
      queue.push_back(src);
      for(size_t head = 0; head < queue.size(); ++head) {
         const auto current = queue[head];
         const auto next_distance = distance.at(static_cast< size_t >(current)) + 2;
         if(next_distance > config.spd_max_hops) {
            continue;
         }
         for(const auto instance : incident_instances.at(static_cast< size_t >(current))) {
            if(visited_epoch.at(static_cast< size_t >(instance)) == src) {
               continue;
            }
            visited_epoch.at(static_cast< size_t >(instance)) = src;
            for(const auto neighbor : instances.at(static_cast< size_t >(instance)).arguments) {
               if(distance.at(static_cast< size_t >(neighbor)) >= 0) {
                  continue;
               }
               distance.at(static_cast< size_t >(neighbor)) = next_distance;
               queue.push_back(neighbor);
            }
         }
      }
      for(int64_t dst = src + 1; dst < object_count; ++dst) {
         const auto pair_distance = distance.at(static_cast< size_t >(dst));
         if(pair_distance < 0) {
            continue;
         }
         src_ids.push_back(src);
         dst_ids.push_back(dst);
         distances.push_back(pair_distance);
      }
   }

   builder.register_field(
      std::string{kSpdSrcField},
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = node_offset_inc(),
      }
   );
   builder.register_field(
      std::string{kSpdDstField},
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = node_offset_inc(),
      }
   );
   builder.register_field(
      std::string{kSpdDistField},
      GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::CAT, .dim = 1}
   );
   builder.set_field(std::string{kSpdSrcField}, src_ids);
   builder.set_field(std::string{kSpdDstField}, dst_ids);
   builder.set_field(std::string{kSpdDistField}, distances);
}

template < typename Input >
void encode_impl(
   const Input& input,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< SemanticActionSpec >& action_specs,
   const SemanticDerivedGraphEncoderConfig& config,
   BatchBuilder& builder
)
{
   builder.set_graph_kind("homo");
   builder.set_schema_flag("include_reverse_edges", config.include_reverse_edges);
   builder.set_schema_flag("include_line_graph", config.include_line_graph);
   builder.set_graph_attr("node_universe", static_cast< int64_t >(config.node_universe));
   builder.set_graph_attr("atom_expansion", static_cast< int64_t >(config.atom_expansion));

   const auto& objects = semantic_objects(input);
   Emitter emitter{predicates, action_specs, objects, config};
   auto& buffers = emitter.buffers;
   buffers.object_incidents.resize(objects.size());
   const bool collect_instances = config.include_hyperedge_incidence || config.include_tuple_tensors
                                  || config.include_spd;

   for(size_t index = 0; index < objects.size(); ++index) {
      const NodeKey key{
         .role = kRoleObject,
         .predicate = -1,
      };
      buffers.node_indices.emplace(key, static_cast< int64_t >(index));
      emitter.write_features(static_cast< int64_t >(index), kRoleObject, -1, true, 0, 0, 0);
      if(config.export_node_names) {
         buffers.node_names.push_back(objects[index]);
      }
   }

   const auto emit_literal_arguments = [&](int64_t fact_index, const SemanticAtom& atom) {
      for(int64_t position = 0; position < static_cast< int64_t >(atom.arguments.size());
          ++position) {
         const auto object_index = atom.arguments[static_cast< size_t >(position)];
         emitter.add_edge(fact_index, object_index, kKindArgFwd, position, position);
         if(config.include_reverse_edges) {
            emitter.add_edge(object_index, fact_index, kKindArgBwd, position, position);
         }
      }
   };

   const auto emit_object_projection = [&](const SemanticAtom& atom) {
      const auto arity = static_cast< int64_t >(atom.arguments.size());
      switch(config.atom_expansion) {
         case DerivedAtomExpansion::star: {
            throw std::logic_error("star expansion never reaches object projection");
         }
         case DerivedAtomExpansion::clique: {
            for(int64_t i = 0; i < arity; ++i) {
               for(auto j = i + 1; j < arity; ++j) {
                  emitter.add_edge(
                     atom.arguments[static_cast< size_t >(i)],
                     atom.arguments[static_cast< size_t >(j)],
                     kKindCliqueFwd,
                     i,
                     j
                  );
                  if(config.include_reverse_edges) {
                     emitter.add_edge(
                        atom.arguments[static_cast< size_t >(j)],
                        atom.arguments[static_cast< size_t >(i)],
                        kKindCliqueBwd,
                        j,
                        i
                     );
                  }
               }
            }
            break;
         }
         case DerivedAtomExpansion::chain: {
            for(int64_t i = 0; i + 1 < arity; ++i) {
               emitter.add_edge(
                  atom.arguments[static_cast< size_t >(i)],
                  atom.arguments[static_cast< size_t >(i + 1)],
                  kKindChainFwd,
                  i,
                  i + 1
               );
               if(config.include_reverse_edges) {
                  emitter.add_edge(
                     atom.arguments[static_cast< size_t >(i + 1)],
                     atom.arguments[static_cast< size_t >(i)],
                     kKindChainBwd,
                     i + 1,
                     i
                  );
               }
            }
            break;
         }
         case DerivedAtomExpansion::star_first: {
            for(int64_t j = 1; j < arity; ++j) {
               emitter.add_edge(
                  atom.arguments.front(),
                  atom.arguments[static_cast< size_t >(j)],
                  kKindStarFirstFwd,
                  0,
                  j
               );
               if(config.include_reverse_edges) {
                  emitter.add_edge(
                     atom.arguments[static_cast< size_t >(j)],
                     atom.arguments.front(),
                     kKindStarFirstBwd,
                     j,
                     0
                  );
               }
            }
            break;
         }
      }
   };

   const auto ensure_fact_node = [&](
                                    int64_t role,
                                    const SemanticAtom& atom,
                                    bool positive,
                                    int64_t level,
                                    int64_t dt,
                                    int64_t category
                                 ) -> int64_t {
      if(collect_instances) {
         buffers.instances.push_back(DerivedInstance{role, atom.predicate, atom.arguments});
      }
      std::optional< std::string > name;
      if(config.export_node_names) {
         auto suffix = std::string_view();
         if(role == kRoleGoal or role == kRoleSubgoal) {
            const auto bounded = std::min< size_t >(
               static_cast< size_t >(level), kGoalSuffixes.size() - 1
            );
            suffix = kGoalSuffixes.at(bounded);
         }
         name = emitter.format_atom_name(atom, positive, suffix);
         if(role == kRoleHistory) {
            name = fmt::format("{}[dt{}]", *name, dt);
         }
      }
      const NodeKey key{
         .role = role,
         .predicate = atom.predicate,
         .arguments = atom.arguments,
         .positive = positive,
         .level = level,
         .dt = dt,
      };
      const auto reified = config.node_universe == DerivedNodeUniverse::objects_and_atoms;
      if(not reified) {
         emit_object_projection(atom);
         return -1;
      }
      const auto index = emitter.ensure_node(key, category, std::move(name));
      if(atom.arguments.empty()) {
         emitter.add_edge(index, index, kKindNullarySelf, 0, 0);
         return index;
      }
      emit_literal_arguments(index, atom);
      if(config.include_line_graph) {
         for(int64_t position = 0; position < static_cast< int64_t >(atom.arguments.size());
             ++position) {
            buffers.object_incidents
               .at(static_cast< size_t >(atom.arguments[static_cast< size_t >(position)]))
               .emplace_back(index, position);
         }
      }
      return index;
   };

   const auto& static_facts = semantic_static_facts(input);
   for(const auto& atom : static_facts) {
      const auto category = static_cast< int64_t >(
         predicates.at(static_cast< size_t >(atom.predicate)).category
      );
      ensure_fact_node(kRoleFact, atom, true, 0, 0, category);
   }
   for(const auto& atom : semantic_state_facts(input)) {
      const auto category = static_cast< int64_t >(
         predicates.at(static_cast< size_t >(atom.predicate)).category
      );
      ensure_fact_node(kRoleFact, atom, true, 0, 0, category);
   }

   auto levels = semantic_goal_levels(input);
   for(const auto& goal : levels) {
      const auto role = goal.level == 0 ? kRoleGoal : kRoleSubgoal;
      const auto category = static_cast< int64_t >(
         predicates.at(static_cast< size_t >(goal.literal.atom.predicate)).category
      );
      ensure_fact_node(
         role,
         goal.literal.atom,
         goal.literal.positive,
         static_cast< int64_t >(goal.level),
         0,
         category
      );
   }

   auto history_entries = [&]() {
      std::vector< std::pair< int64_t, std::vector< SemanticLiteral > > > entries;
      std::optional< int64_t > history_max_steps;
      if constexpr(requires { input.history_max_steps; }) {
         history_max_steps = input.history_max_steps;
      }
      for(const auto& entry : semantic_history(input)) {
         const auto dt = static_cast< int64_t >(entry.dt);
         if(history_max_steps and std::abs(dt) > *history_max_steps) {
            continue;
         }
         auto& target = entries.emplace_back();
         target.first = dt;
         for(const auto& literal : entry.literals) {
            target.second.push_back(literal);
         }
      }
      std::ranges::stable_sort(
         entries, {}, &std::pair< int64_t, std::vector< SemanticLiteral > >::first
      );
      return entries;
   }();
   for(const auto& [dt, literals] : history_entries) {
      for(const auto& literal : literals) {
         const auto category = static_cast< int64_t >(
            predicates.at(static_cast< size_t >(literal.atom.predicate)).category
         );
         ensure_fact_node(kRoleHistory, literal.atom, literal.positive, 0, dt, category);
      }
   }

   if constexpr(requires { semantic_actions(input); }) {
      if(not std::ranges::empty(semantic_actions(input))) {
         for(const auto& action : semantic_actions(input)) {
            const NodeKey key{
               .role = kRoleAction,
               .predicate = action.action,
               .arguments = action.arguments,
            };
            std::optional< std::string > name;
            if(config.export_node_names) {
               name = emitter.format_action_name(action);
            }
            if(collect_instances) {
               buffers.instances.push_back(
                  DerivedInstance{kRoleAction, action.action, action.arguments}
               );
            }
            const auto index = emitter.ensure_node(key, 0, std::move(name));
            for(int64_t position = 0; position < static_cast< int64_t >(action.arguments.size());
                ++position) {
               const auto object_index = action.arguments[static_cast< size_t >(position)];
               emitter.add_edge(index, object_index, kKindActionFwd, position, position);
               if(config.include_reverse_edges) {
                  emitter.add_edge(object_index, index, kKindActionBwd, position, position);
               }
            }
         }
      }
   }

   if(config.include_line_graph) {
      for(const auto& incidents : buffers.object_incidents) {
         if(incidents.size() < 2
            || static_cast< int64_t >(incidents.size()) > config.line_graph_max_degree) {
            continue;
         }
         for(size_t i = 0; i < incidents.size(); ++i) {
            for(auto j = i + 1; j < incidents.size(); ++j) {
               const auto [src, src_position] = incidents[i];
               const auto [dst, dst_position] = incidents[j];
               if(src == dst) {
                  continue;
               }
               emitter.add_edge(src, dst, kKindLineShare, src_position, dst_position);
               emitter.add_edge(dst, src, kKindLineShare, dst_position, src_position);
            }
         }
      }
   }

   static const std::string node_type = "node";
   builder.add_nodes(node_type, buffers.node_count());
   if(config.export_node_names) {
      builder.set_node_names(node_type, buffers.node_names);
   }
   builder.set_object_names(objects);
   builder.add_node_features(node_type, "x_ids", std::span{buffers.x_flat}, kXDim);
   if(not buffers.edge_src.empty()) {
      builder.add_edges(node_type, "edge", node_type, buffers.edge_src, buffers.edge_dst);
      builder.add_edge_features(
         node_type, "edge", node_type, "edge_attr", std::span{buffers.edge_attr_flat}, kEdgeDim
      );
   }

   if(collect_instances) {
      if(config.include_hyperedge_incidence) {
         builder.set_graph_attr(
            "hyperedge_note",
            std::string(
               "members are object nodes; pair rows via zip(hyperedge_node_indices, "
               "hyperedge_ids)"
            )
         );
      }
      register_instance_fields(builder, config);
      set_instance_fields(builder, config, buffers.instances);
   }
   if(config.include_spd) {
      emit_spd_fields(builder, config, static_cast< int64_t >(objects.size()), buffers.instances);
   }

   builder.set_graph_attr(
      "vocab_roles",
      std::vector< std::string >{"object", "fact", "goal", "subgoal", "history", "action"}
   );
   std::vector< std::string > predicate_vocab;
   predicate_vocab.reserve(predicates.size());
   for(const auto& predicate : predicates) {
      predicate_vocab.push_back(predicate.name);
   }
   builder.set_graph_attr("vocab_predicates", predicate_vocab);
   builder.set_graph_attr(
      "vocab_edge_kinds",
      std::vector< std::string >{
         "arg_fwd",
         "arg_bwd",
         "clique_fwd",
         "clique_bwd",
         "chain_fwd",
         "chain_bwd",
         "star_first_fwd",
         "star_first_bwd",
         "nullary_self",
         "action_fwd",
         "action_bwd",
         "line_share",
      }
   );
   builder.set_graph_attr(
      "vocab_categories", std::vector< std::string >{"static", "fluent", "derived"}
   );
   builder.set_graph_attr(
      "channel_names",
      std::vector< std::string >{
         "role", "predicate_id_plus_one", "sign", "goal_level", "history_dt", "category"
      }
   );
   builder.set_graph_attr(
      "edge_channel_names", std::vector< std::string >{"kind", "pos_a", "pos_b"}
   );
}

}  // namespace

SemanticDerivedGraphEncoderEngine::SemanticDerivedGraphEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   SemanticDerivedGraphEncoderConfig config
)
    : schema_context_(require_schema_context(
         std::make_shared< const SemanticSchemaContext >(
            SemanticSchemaContext{.predicates = std::move(predicates)}
         )
      )),
      config_(config)
{
   validate_config(config_);
}

SemanticDerivedGraphEncoderEngine::SemanticDerivedGraphEncoderEngine(
   std::shared_ptr< const SemanticSchemaContext > schema,
   SemanticDerivedGraphEncoderConfig config
)
    : schema_context_(require_schema_context(schema)), config_(config)
{
   validate_config(config_);
}

const std::vector< SemanticPredicateSpec >&
SemanticDerivedGraphEncoderEngine::get_predicates() const
{
   return schema_context_->predicates;
}

BatchBuilder::BatchEncoding SemanticDerivedGraphEncoderEngine::encode(
   const SemanticFlatRelationInput& input
) const
{
   BatchBuilder builder;
   encode(input, builder);
   builder.next_graph();
   return builder.build();
}

void SemanticDerivedGraphEncoderEngine::encode(
   const SemanticFlatRelationInput& input,
   BatchBuilder& builder
) const
{
   encode_impl(input, schema_context_->predicates, schema_context_->actions, config_, builder);
}

BatchBuilder::BatchEncoding SemanticDerivedGraphEncoderEngine::encode_batch(
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

BatchBuilder::BatchEncoding SemanticDerivedGraphEncoderEngine::encode_batch(
   std::span< const canonical::detail::ViewPreparation* const > preparations
) const
{
   BatchBuilder builder;
   for(const auto* preparation : preparations) {
      encode_view_preparation(*preparation, builder);
      builder.next_graph();
   }
   return builder.build();
}

void SemanticDerivedGraphEncoderEngine::encode_view_preparation(
   const canonical::detail::ViewPreparation& preparation,
   BatchBuilder& builder
) const
{
   encode_impl(
      preparation, schema_context_->predicates, schema_context_->actions, config_, builder
   );
}

}  // namespace mifrost
