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
/// Auxiliary node that carries arity-0 instances (see `emit_anchor` below).
constexpr int64_t kRoleAnchor = 6;

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
constexpr int64_t kKindUnarySelf = 12;

constexpr int64_t kCategoryStatic = 0;
/// Action-role nodes/instances/edges are not predicate categories; id 3 is
/// appended to `vocab_categories` rather than reusing "static".
constexpr int64_t kCategoryAction = 3;

constexpr int kXDim = 6;
constexpr int kEdgeDim = 9;

constexpr std::string_view kHyperedgeSizesField = "hyperedge_sizes";
constexpr std::string_view kHyperedgeNodeIndicesField = "hyperedge_node_indices";
constexpr std::string_view kHyperedgeIdsField = "hyperedge_ids";
constexpr std::string_view kHyperedgeAttrRowsField = "hyperedge_attr_rows";
constexpr std::string_view kHyperedgeCountsField = "hyperedge_counts";
// Instance channels follow multi-set semantics: the same atom occurring as a
// state fact and as a goal literal emits two instances, so per-instance counts
// exceed distinct-atom counts by design. The *graph view* is a set: argument
// edges, self-loops, the object projection and line-graph incidents fire only
// on the first interning of an instance key.
constexpr std::string_view kTupleArgsField = "tuple_args";
constexpr std::string_view kTupleSlotSizesField = "tuple_slot_sizes";
constexpr std::string_view kTupleRelIdsField = "tuple_rel_ids";
constexpr std::string_view kTupleRoleIdsField = "tuple_role_ids";
constexpr std::string_view kTupleSignIdsField = "tuple_sign_ids";
constexpr std::string_view kTupleLevelIdsField = "tuple_level_ids";
constexpr std::string_view kTupleDtIdsField = "tuple_dt_ids";
constexpr std::string_view kTupleCategoryIdsField = "tuple_category_ids";
constexpr std::string_view kTupleCountsField = "tuple_counts";
constexpr std::string_view kInstanceNodeIndicesField = "instance_node_indices";
constexpr std::string_view kAnchorIndexField = "anchor_index";
constexpr std::string_view kHistoryDtOffsetField = "history_dt_offset";
constexpr std::string_view kSpdSrcField = "spd_src";
constexpr std::string_view kSpdDstField = "spd_dst";
constexpr std::string_view kSpdDistField = "spd_dist";

/**
 * Unified relation id: predicates keep `0 .. P-1`, action schemas shift to
 * `P + schema_id`, so a node/edge/instance channel can never confuse an action
 * schema with a predicate. Degenerate (negative) schema ids stay `-1` ("none").
 */
[[nodiscard]] constexpr int64_t action_relation_id(int64_t action, size_t predicate_count)
{
   if(action < 0) {
      return -1;
   }
   return static_cast< int64_t >(predicate_count) + action;
}

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

/// Allocation-free lookup view of a `NodeKey` (arguments are not copied).
struct NodeKeyView {
   int64_t role;
   int64_t predicate;
   std::span< const int64_t > arguments;
   bool positive;
   int64_t level;
   int64_t dt;

   [[nodiscard]] bool operator==(const NodeKey& key) const noexcept
   {
      return role == key.role and predicate == key.predicate
             and std::ranges::equal(arguments, key.arguments) and positive == key.positive
             and level == key.level and dt == key.dt;
   }
};

struct NodeKeyHash {
   using is_transparent = void;

   template < typename Key >
   size_t operator()(const Key& key) const noexcept
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

struct NodeKeyEqual {
   using is_transparent = void;

   [[nodiscard]] bool operator()(const NodeKey& left, const NodeKey& right) const noexcept
   {
      return left == right;
   }

   [[nodiscard]] bool operator()(const NodeKeyView& view, const NodeKey& key) const noexcept
   {
      return view == key;
   }

   [[nodiscard]] bool operator()(const NodeKey& key, const NodeKeyView& view) const noexcept
   {
      return view == key;
   }
};

/**
 * The six per-instance channels mirrored by `x_ids`, `edge_attr` cols 3..8 and
 * both instance tables. `relation` is the *unified* relation id (see
 * `action_relation_id`); `-1` means "none" and is exported as `+1 == 0`.
 */
struct InstanceLabels {
   int64_t role = kRoleFact;
   int64_t relation = -1;
   int64_t sign = 0;
   int64_t level = 0;
   int64_t dt = 0;
   int64_t category = kCategoryStatic;
};

struct DerivedInstance {
   InstanceLabels labels;
   SemanticArguments arguments;
   /// Reified node carrying this instance's labels, or -1 when not reified.
   int64_t node_index = -1;
};

struct Buffers {
   hash_map< NodeKey, int64_t, NodeKeyHash, NodeKeyEqual > node_indices;
   /// First-occurrence guard for the objects-only projection, which emits
   /// edges without interning a node (graph view = set, table = multiset).
   hash_map< NodeKey, int64_t, NodeKeyHash, NodeKeyEqual > projection_seen;
   /// Row-major [N, 6]: role, relation_id+1, sign, goal_level, history_dt, category.
   std::vector< float > x_flat;
   std::vector< int64_t > edge_src;
   std::vector< int64_t > edge_dst;
   /// Row-major [E, 9]: kind, pos_a, pos_b, relation_id+1, role, sign,
   /// goal_level, history_dt, category.
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

/// Result of one `intern_node` call: the node index plus first-insertion flag.
struct InternResult {
   int64_t index = -1;
   bool inserted = false;
};

struct Emitter {
   const std::vector< SemanticPredicateSpec >& predicates;
   const std::vector< SemanticActionSpec >& action_specs;
   const std::vector< std::string >& objects;
   const SemanticDerivedGraphEncoderConfig& config;
   Buffers buffers;

   void write_features(int64_t index, const InstanceLabels& labels)
   {
      const auto slot = static_cast< size_t >(index) * kXDim;
      buffers.x_flat.resize(std::max(buffers.x_flat.size(), slot + kXDim), 0.0F);
      buffers.x_flat[slot] = static_cast< float >(labels.role);
      buffers.x_flat[slot + 1] = static_cast< float >(labels.relation + 1);
      buffers.x_flat[slot + 2] = static_cast< float >(labels.sign);
      buffers.x_flat[slot + 3] = static_cast< float >(labels.level);
      buffers.x_flat[slot + 4] = static_cast< float >(labels.dt);
      buffers.x_flat[slot + 5] = static_cast< float >(labels.category);
   }

   /// Lookup-first interning: ``make_key``/``format_name`` run only on a miss,
   /// so repeated facts neither copy their arguments nor format their names.
   /// The returned `inserted` flag is what keeps the *graph view* a set: every
   /// edge derived from an instance fires only on its first interning.
   template < typename MakeKey, typename FormatName >
   [[nodiscard]] InternResult intern_node(
      const NodeKeyView& view,
      MakeKey&& make_key,
      FormatName&& format_name,
      const InstanceLabels& labels
   )
   {
      auto& nodes = buffers.node_indices;
      const auto found = nodes.find(view);
      if(found != nodes.end()) {
         return InternResult{.index = found->second, .inserted = false};
      }
      // Object keys collapse to one map entry (role/predicate only), so the
      // map size lags the feature-row count; index by row count, never by
      // map size.
      const auto next_index = buffers.node_count();
      const auto [it, inserted] = nodes.try_emplace(
         std::forward< MakeKey >(make_key)(), next_index
      );
      if(inserted) {
         write_features(it->second, labels);
         if(config.export_node_names) {
            buffers.node_names.emplace_back(format_name());
         }
      }
      return InternResult{.index = it->second, .inserted = inserted};
   }

   void add_edge(
      int64_t src,
      int64_t dst,
      int64_t kind,
      int64_t pos_a,
      int64_t pos_b,
      const InstanceLabels& labels
   )
   {
      buffers.edge_src.push_back(src);
      buffers.edge_dst.push_back(dst);
      buffers.edge_attr_flat.push_back(static_cast< float >(kind));
      buffers.edge_attr_flat.push_back(static_cast< float >(pos_a));
      buffers.edge_attr_flat.push_back(static_cast< float >(pos_b));
      buffers.edge_attr_flat.push_back(static_cast< float >(labels.relation + 1));
      buffers.edge_attr_flat.push_back(static_cast< float >(labels.role));
      buffers.edge_attr_flat.push_back(static_cast< float >(labels.sign));
      buffers.edge_attr_flat.push_back(static_cast< float >(labels.level));
      buffers.edge_attr_flat.push_back(static_cast< float >(labels.dt));
      buffers.edge_attr_flat.push_back(static_cast< float >(labels.category));
   }

   /// `line_share` shortcuts are not derived from a single instance: both
   /// endpoints are reified fact nodes that already carry their own labels, so
   /// label columns 3..8 are zero. This is the one permitted "none" case.
   void add_unlabeled_edge(int64_t src, int64_t dst, int64_t kind, int64_t pos_a, int64_t pos_b)
   {
      buffers.edge_src.push_back(src);
      buffers.edge_dst.push_back(dst);
      buffers.edge_attr_flat.push_back(static_cast< float >(kind));
      buffers.edge_attr_flat.push_back(static_cast< float >(pos_a));
      buffers.edge_attr_flat.push_back(static_cast< float >(pos_b));
      buffers.edge_attr_flat.insert(buffers.edge_attr_flat.end(), kEdgeDim - 3, 0.0F);
   }

   /// Byte-identical single-pass form of the historical fmt::join rendering.
   [[nodiscard]] std::string
   format_atom_name(const SemanticAtom& atom, bool positive, std::string_view suffix) const
   {
      const auto& predicate_name = predicates.at(static_cast< size_t >(atom.predicate)).name;
      std::string out;
      out.reserve(predicate_name.size() + atom.arguments.size() * 8 + suffix.size() + 8);
      if(not positive) {
         out.append("(not ");
      }
      out.push_back('(');
      out.append(predicate_name);
      for(const auto argument : atom.arguments) {
         out.push_back(' ');
         out.append(objects.at(static_cast< size_t >(argument)));
      }
      out.push_back(')');
      if(not positive) {
         out.push_back(')');
      }
      out.append(suffix);
      return out;
   }

   [[nodiscard]] std::string format_action_name(const SemanticGroundAction& action) const
   {
      std::string out = "@(";
      if(action.action >= 0 && static_cast< size_t >(action.action) < action_specs.size()) {
         out.append(action_specs.at(static_cast< size_t >(action.action)).name);
      } else {
         out.append("action");
         out.append(std::to_string(action.action));
      }
      for(const auto argument : action.arguments) {
         out.push_back(' ');
         out.append(objects.at(static_cast< size_t >(argument)));
      }
      out.push_back(')');
      return out;
   }
};

/// Constant graph-attribute vocabularies, built once instead of once per graph.
const std::vector< std::string >& role_vocabulary()
{
   static const std::vector< std::string > vocabulary = {
      "object", "fact", "goal", "subgoal", "history", "action", "anchor"
   };
   return vocabulary;
}

const std::vector< std::string >& edge_kind_vocabulary()
{
   static const std::vector< std::string > vocabulary = {
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
      "unary_self"
   };
   return vocabulary;
}

const std::vector< std::string >& category_vocabulary()
{
   static const std::vector< std::string > vocabulary = {"static", "fluent", "derived", "action"};
   return vocabulary;
}

const std::vector< std::string >& channel_names()
{
   static const std::vector< std::string > names = {
      "role", "relation_id_plus_one", "sign", "goal_level", "history_dt", "category"
   };
   return names;
}

const std::vector< std::string >& edge_channel_names()
{
   static const std::vector< std::string > names = {
      "kind",
      "pos_a",
      "pos_b",
      "rel_id_plus_one",
      "role",
      "sign",
      "goal_level",
      "history_dt",
      "category"
   };
   return names;
}

std::vector< std::string > predicate_vocabulary(
   const std::vector< SemanticPredicateSpec >& predicates
)
{
   std::vector< std::string > vocabulary;
   vocabulary.reserve(predicates.size());
   for(const auto& predicate : predicates) {
      vocabulary.push_back(predicate.name);
   }
   return vocabulary;
}

std::vector< std::string > action_vocabulary(const std::vector< SemanticActionSpec >& actions)
{
   std::vector< std::string > vocabulary;
   vocabulary.reserve(actions.size());
   for(const auto& action : actions) {
      vocabulary.push_back(action.name);
   }
   return vocabulary;
}

/// `vocab_predicates ++ vocab_actions`: the unified relation id space.
std::vector< std::string > relation_vocabulary(
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< SemanticActionSpec >& actions
)
{
   std::vector< std::string > vocabulary;
   vocabulary.reserve(predicates.size() + actions.size());
   for(const auto& predicate : predicates) {
      vocabulary.push_back(predicate.name);
   }
   for(const auto& action : actions) {
      vocabulary.push_back(action.name);
   }
   return vocabulary;
}

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
   // One reified-node back-pointer per instance row, in instance order. This
   // is what links a hyperedge/tuple row to the node carrying its labels: the
   // old "tuple row i == node num_objects + i" coincidence no longer holds
   // once duplicates are deduplicated in the graph view or an anchor shifts
   // the node indices.
   builder.register_field(
      std::string(kInstanceNodeIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = node_offset_inc(),
      }
   );
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
      // [M, 6] row-major, mirroring `x_ids`' channels exactly:
      // role, rel_id_plus_one, sign, goal_level, history_dt, category.
      builder.register_field(
         std::string(kHyperedgeAttrRowsField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 6,
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
         }
      );
      for(const auto& key : {
             kTupleRoleIdsField,
             kTupleSignIdsField,
             kTupleLevelIdsField,
             kTupleDtIdsField,
             kTupleCategoryIdsField,
          }) {
         builder.register_field(
            std::string(key),
            GraphFieldSpec{
               .dtype = GraphFieldDType::I64,
               .mode = GraphFieldMode::CAT,
               .dim = 1,
            }
         );
      }
   }
}

void set_instance_fields(
   BatchBuilder& builder,
   const SemanticDerivedGraphEncoderConfig& config,
   const std::vector< DerivedInstance >& instances,
   int64_t anchor_index
)
{
   const auto instance_count = static_cast< int64_t >(instances.size());
   // Reified node per instance. Zero-arity instances that were not reified
   // (objects-only projection) point at the anchor; the rest of the
   // objects-only table has no reified node and reports -1.
   std::vector< int64_t > instance_nodes;
   instance_nodes.reserve(instances.size());
   for(const auto& instance : instances) {
      if(instance.node_index >= 0) {
         instance_nodes.push_back(instance.node_index);
      } else {
         instance_nodes.push_back(instance.arguments.empty() ? anchor_index : int64_t{-1});
      }
   }
   builder.set_field(std::string{kInstanceNodeIndicesField}, instance_nodes);

   if(config.include_hyperedge_incidence) {
      std::vector< int64_t > sizes;
      std::vector< int64_t > members;
      std::vector< int64_t > ids;
      std::vector< int64_t > attr_rows;
      sizes.reserve(instances.size());
      ids.reserve(instances.size());
      attr_rows.reserve(instances.size() * 6);
      int64_t next_id = 0;
      for(const auto& instance : instances) {
         if(instance.arguments.empty()) {
            // A zero-member hyperedge is invisible to HypergraphConv and
            // desyncs its inferred `num_edges`; anchor it instead.
            if(anchor_index < 0) {
               throw std::logic_error(
                  "derived-graph hyperedge incidence requires the anchor node for arity-0 "
                  "instances"
               );
            }
            sizes.push_back(1);
            members.push_back(anchor_index);
         } else {
            sizes.push_back(static_cast< int64_t >(instance.arguments.size()));
            members.insert(members.end(), instance.arguments.begin(), instance.arguments.end());
         }
         ids.push_back(next_id++);
         attr_rows.push_back(instance.labels.role);
         attr_rows.push_back(instance.labels.relation + 1);
         attr_rows.push_back(instance.labels.sign);
         attr_rows.push_back(instance.labels.level);
         attr_rows.push_back(instance.labels.dt);
         attr_rows.push_back(instance.labels.category);
      }
      builder.set_field(std::string{kHyperedgeSizesField}, sizes);
      builder.set_field(std::string{kHyperedgeNodeIndicesField}, members);
      builder.set_field(std::string{kHyperedgeIdsField}, ids);
      builder.set_field(std::string{kHyperedgeAttrRowsField}, attr_rows);
      builder.set_field(
         std::string{kHyperedgeCountsField}, std::span< const int64_t >{&instance_count, 1}
      );
   }
   if(config.include_tuple_tensors) {
      std::vector< int64_t > args;
      std::vector< int64_t > slot_sizes;
      std::vector< int64_t > rel_ids;
      std::vector< int64_t > role_ids;
      std::vector< int64_t > sign_ids;
      std::vector< int64_t > level_ids;
      std::vector< int64_t > dt_ids;
      std::vector< int64_t > category_ids;
      slot_sizes.reserve(instances.size());
      rel_ids.reserve(instances.size());
      role_ids.reserve(instances.size());
      sign_ids.reserve(instances.size());
      level_ids.reserve(instances.size());
      dt_ids.reserve(instances.size());
      category_ids.reserve(instances.size());
      for(const auto& instance : instances) {
         args.insert(args.end(), instance.arguments.begin(), instance.arguments.end());
         slot_sizes.push_back(static_cast< int64_t >(instance.arguments.size()));
         rel_ids.push_back(instance.labels.relation);
         role_ids.push_back(instance.labels.role);
         sign_ids.push_back(instance.labels.sign);
         level_ids.push_back(instance.labels.level);
         dt_ids.push_back(instance.labels.dt);
         category_ids.push_back(instance.labels.category);
      }
      builder.set_field(std::string{kTupleSlotSizesField}, slot_sizes);
      builder.set_field(std::string{kTupleArgsField}, args);
      builder.set_field(std::string{kTupleRelIdsField}, rel_ids);
      builder.set_field(std::string{kTupleRoleIdsField}, role_ids);
      builder.set_field(std::string{kTupleSignIdsField}, sign_ids);
      builder.set_field(std::string{kTupleLevelIdsField}, level_ids);
      builder.set_field(std::string{kTupleDtIdsField}, dt_ids);
      builder.set_field(std::string{kTupleCategoryIdsField}, category_ids);
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
   const auto validate_atom_arguments = [&](const SemanticAtom& atom) {
      for(const auto argument : atom.arguments) {
         if(argument < 0 || argument >= static_cast< int64_t >(objects.size())) {
            throw std::invalid_argument(
               "semantic derived-graph input references object index " + std::to_string(argument)
               + " outside the problem's " + std::to_string(objects.size()) + " objects"
            );
         }
      }
   };
   for(const auto& atom : semantic_static_facts(input)) {
      validate_atom_arguments(atom);
   }
   for(const auto& atom : semantic_state_facts(input)) {
      validate_atom_arguments(atom);
   }
   for(const auto& goal : semantic_goal_levels(input)) {
      validate_atom_arguments(goal.literal.atom);
   }
   Emitter emitter{predicates, action_specs, objects, config};
   auto& buffers = emitter.buffers;
   if(config.include_line_graph) {
      buffers.object_incidents.resize(objects.size());
   }
   const bool collect_instances = config.include_hyperedge_incidence || config.include_tuple_tensors
                                  || config.include_spd;

   // Size the accumulation buffers from the lanes that report their length up
   // front; goals/history/actions still grow past these lower bounds.
   const auto& reserved_static_facts = semantic_static_facts(input);
   const auto& state_fact_span = semantic_state_facts(input);
   const size_t fact_estimate = objects.size() + reserved_static_facts.size()
                                + std::ranges::size(state_fact_span);
   int64_t argument_slots = 0;
   for(const auto& atom : reserved_static_facts) {
      argument_slots += static_cast< int64_t >(atom.arguments.size());
   }
   for(const auto& atom : state_fact_span) {
      argument_slots += static_cast< int64_t >(atom.arguments.size());
   }
   const auto edge_estimate = static_cast< size_t >(std::max< int64_t >(
      argument_slots * (config.include_reverse_edges ? 2 : 1) + fact_estimate / 4 + 16, 16
   ));
   buffers.x_flat.reserve(fact_estimate * kXDim * 2);
   if(config.export_node_names) {
      buffers.node_names.reserve(fact_estimate * 2);
   }
   buffers.edge_src.reserve(edge_estimate);
   buffers.edge_dst.reserve(edge_estimate);
   buffers.edge_attr_flat.reserve(edge_estimate * kEdgeDim);
   if(collect_instances) {
      buffers.instances.reserve(fact_estimate);
   }

   for(size_t index = 0; index < objects.size(); ++index) {
      const NodeKey key{
         .role = kRoleObject,
         .predicate = -1,
      };
      buffers.node_indices.emplace(key, static_cast< int64_t >(index));
      emitter.write_features(
         static_cast< int64_t >(index), InstanceLabels{.role = kRoleObject, .relation = -1}
      );
      if(config.export_node_names) {
         buffers.node_names.push_back(objects[index]);
      }
   }

   // A single auxiliary node carrying arity-0 instances. Emitted for the
   // objects-only projection (which has no reified atom nodes at all, so a
   // nullary literal would leave no trace) and whenever hyperedge incidence is
   // requested (so no hyperedge is ever empty). It is appended as the *last*
   // node, leaving every object and fact-node index unchanged, and it is not an
   // object: it never enters `object_names` or the `spd_*` object loops.
   const bool emit_anchor = config.node_universe == DerivedNodeUniverse::objects_only
                            or config.include_hyperedge_incidence;
   int64_t anchor_index = -1;
   const auto ensure_anchor = [&]() -> int64_t {
      if(anchor_index >= 0) {
         return anchor_index;
      }
      anchor_index = buffers.node_count();
      buffers.node_indices.emplace(NodeKey{.role = kRoleAnchor, .predicate = -1}, anchor_index);
      emitter.write_features(anchor_index, InstanceLabels{.role = kRoleAnchor, .relation = -1});
      if(config.export_node_names) {
         buffers.node_names.emplace_back("<nullary>");
      }
      return anchor_index;
   };
   /// Arity-0 objects-only instances: the anchor is allocated last, so their
   /// self-loops are replayed once its index is known.
   std::vector< InstanceLabels > pending_anchor_self_loops;

   const auto emit_literal_arguments =
      [&](int64_t fact_index, const SemanticAtom& atom, const InstanceLabels& labels) {
         for(int64_t position = 0; position < static_cast< int64_t >(atom.arguments.size());
             ++position) {
            const auto object_index = atom.arguments[static_cast< size_t >(position)];
            emitter.add_edge(fact_index, object_index, kKindArgFwd, position, position, labels);
            if(config.include_reverse_edges) {
               emitter.add_edge(object_index, fact_index, kKindArgBwd, position, position, labels);
            }
         }
      };

   const auto emit_object_projection = [&](const SemanticAtom& atom, const InstanceLabels& labels) {
      const auto arity = static_cast< int64_t >(atom.arguments.size());
      if(arity == 0) {
         pending_anchor_self_loops.push_back(labels);
         return;
      }
      if(arity == 1) {
         const auto argument = atom.arguments.front();
         emitter.add_edge(argument, argument, kKindUnarySelf, 0, 0, labels);
         return;
      }
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
                     j,
                     labels
                  );
                  if(config.include_reverse_edges) {
                     emitter.add_edge(
                        atom.arguments[static_cast< size_t >(j)],
                        atom.arguments[static_cast< size_t >(i)],
                        kKindCliqueBwd,
                        j,
                        i,
                        labels
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
                  i + 1,
                  labels
               );
               if(config.include_reverse_edges) {
                  emitter.add_edge(
                     atom.arguments[static_cast< size_t >(i + 1)],
                     atom.arguments[static_cast< size_t >(i)],
                     kKindChainBwd,
                     i + 1,
                     i,
                     labels
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
                  j,
                  labels
               );
               if(config.include_reverse_edges) {
                  emitter.add_edge(
                     atom.arguments[static_cast< size_t >(j)],
                     atom.arguments.front(),
                     kKindStarFirstBwd,
                     j,
                     0,
                     labels
                  );
               }
            }
            break;
         }
      }
   };

   /// First-occurrence guard for the objects-only projection: it emits edges
   /// without interning a node, so it needs its own set of seen instance keys.
   const auto first_projection = [&](const NodeKeyView& view) {
      auto& seen = buffers.projection_seen;
      if(seen.find(view) != seen.end()) {
         return false;
      }
      seen.emplace(
         NodeKey{
            .role = view.role,
            .predicate = view.predicate,
            .arguments = SemanticArguments(view.arguments.begin(), view.arguments.end()),
            .positive = view.positive,
            .level = view.level,
            .dt = view.dt,
         },
         int64_t{0}
      );
      return true;
   };

   const auto ensure_fact_node = [&](
                                    int64_t role,
                                    const SemanticAtom& atom,
                                    bool positive,
                                    int64_t level,
                                    int64_t dt,
                                    int64_t category
                                 ) -> int64_t {
      const InstanceLabels labels{
         .role = role,
         .relation = atom.predicate,
         .sign = positive ? 0 : 1,
         .level = level,
         .dt = dt,
         .category = category,
      };
      const NodeKeyView view{
         .role = role,
         .predicate = atom.predicate,
         .arguments = atom.arguments,
         .positive = positive,
         .level = level,
         .dt = dt,
      };
      const auto reified = config.node_universe == DerivedNodeUniverse::objects_and_atoms;
      int64_t index = -1;
      if(reified) {
         const auto interned = emitter.intern_node(
            view,
            [&]() {
               return NodeKey{
                  .role = role,
                  .predicate = atom.predicate,
                  .arguments = SemanticArguments(atom.arguments),
                  .positive = positive,
                  .level = level,
                  .dt = dt,
               };
            },
            [&]() {
               std::string suffix_storage;
               auto suffix = std::string_view();
               if(role == kRoleGoal or role == kRoleSubgoal) {
                  if(level >= 0 and static_cast< size_t >(level) < kGoalSuffixes.size()) {
                     suffix = kGoalSuffixes.at(static_cast< size_t >(level));
                  } else {
                     // Beyond the spelled-out ladder, render the level
                     // explicitly instead of clamping: clamping made every
                     // level >= 3 share the name "[sssg]".
                     suffix_storage = "[sg" + std::to_string(level) + "]";
                     suffix = suffix_storage;
                  }
               }
               std::string name = emitter.format_atom_name(atom, positive, suffix);
               if(role == kRoleHistory) {
                  name += "[dt";
                  name += std::to_string(dt);
                  name += "]";
               }
               return name;
            },
            labels
         );
         index = interned.index;
         if(interned.inserted) {
            if(atom.arguments.empty()) {
               emitter.add_edge(index, index, kKindNullarySelf, 0, 0, labels);
            } else {
               emit_literal_arguments(index, atom, labels);
               if(config.include_line_graph) {
                  for(int64_t position = 0;
                      position < static_cast< int64_t >(atom.arguments.size());
                      ++position) {
                     buffers.object_incidents
                        .at(static_cast< size_t >(atom.arguments[static_cast< size_t >(position)]))
                        .emplace_back(index, position);
                  }
               }
            }
         }
      } else if(first_projection(view)) {
         emit_object_projection(atom, labels);
      }
      if(collect_instances) {
         buffers.instances.push_back(
            DerivedInstance{.labels = labels, .arguments = atom.arguments, .node_index = index}
         );
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
   for(const auto& atom : state_fact_span) {
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
      // ViewPreparation inputs (they expose `history_data`) are already
      // stable-sorted by dt during construction; re-sorting is a no-op there.
      if constexpr(not requires { input.history_data; }) {
         std::ranges::stable_sort(
            entries, {}, &std::pair< int64_t, std::vector< SemanticLiteral > >::first
         );
      }
      return entries;
   }();
   // `history_dt` (x_ids col 4 / edge_attr col 7) is the one *signed* channel:
   // history entries carry negative dt. Export the shift that turns it into a
   // non-negative embedding id, `x_ids[:, 4] + history_dt_offset`, without
   // destroying the signed truth in the channel itself.
   int64_t history_dt_offset = 0;
   for(const auto& [dt, literals] : history_entries) {
      if(literals.empty()) {
         continue;
      }
      history_dt_offset = std::max(history_dt_offset, -dt);
   }
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
         size_t action_count = std::ranges::size(semantic_actions(input));
         if constexpr(requires { input.action_occurrence_indices; }) {
            action_count = input.action_occurrence_indices.size();
         }
         for(size_t occurrence = 0; occurrence < action_count; ++occurrence) {
            const auto& action = [&]() -> const SemanticGroundAction& {
               if constexpr(requires { input.action_occurrence_indices; }) {
                  return semantic_action_at(input, input.action_occurrence_indices.at(occurrence));
               } else {
                  return semantic_action_at(input, occurrence);
               }
            }();
            const NodeKeyView view{
               .role = kRoleAction,
               .predicate = action.action,
               .arguments = action.arguments,
               .positive = true,
               .level = 0,
               .dt = 0,
            };
            // Action schemas live in the *shifted* half of the unified
            // relation id space, so an action can never be decoded as a
            // predicate of the same raw id.
            const InstanceLabels labels{
               .role = kRoleAction,
               .relation = action_relation_id(action.action, predicates.size()),
               .category = kCategoryAction,
            };
            const auto interned = emitter.intern_node(
               view,
               [&]() {
                  return NodeKey{
                     .role = kRoleAction,
                     .predicate = action.action,
                     .arguments = SemanticArguments(action.arguments),
                  };
               },
               [&]() { return emitter.format_action_name(action); },
               labels
            );
            const auto index = interned.index;
            if(interned.inserted) {
               for(int64_t position = 0; position < static_cast< int64_t >(action.arguments.size());
                   ++position) {
                  const auto object_index = action.arguments[static_cast< size_t >(position)];
                  emitter.add_edge(index, object_index, kKindActionFwd, position, position, labels);
                  if(config.include_reverse_edges) {
                     emitter.add_edge(
                        object_index, index, kKindActionBwd, position, position, labels
                     );
                  }
               }
            }
            if(collect_instances) {
               buffers.instances.push_back(
                  DerivedInstance{
                     .labels = labels, .arguments = action.arguments, .node_index = index
                  }
               );
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
               // A fact pair sharing `m` objects yields `m` shortcuts per
               // direction with distinct pos_a/pos_b: deliberate, not a
               // duplicate. The reverse direction obeys include_reverse_edges.
               emitter.add_unlabeled_edge(src, dst, kKindLineShare, src_position, dst_position);
               if(config.include_reverse_edges) {
                  emitter.add_unlabeled_edge(dst, src, kKindLineShare, dst_position, src_position);
               }
            }
         }
      }
   }

   if(emit_anchor) {
      ensure_anchor();
      for(const auto& labels : pending_anchor_self_loops) {
         emitter.add_edge(anchor_index, anchor_index, kKindNullarySelf, 0, 0, labels);
      }
   }

   static const std::string node_type = "node";
   builder.add_nodes(node_type, buffers.node_count());
   if(config.export_node_names) {
      builder.set_node_names(node_type, buffers.node_names);
   }
   builder.set_object_names(objects);
   builder.add_node_features(node_type, "x_ids", std::span{buffers.x_flat}, kXDim);
   builder.add_edges(node_type, "edge", node_type, buffers.edge_src, buffers.edge_dst);
   builder.add_edge_features(
      node_type, "edge", node_type, "edge_attr", std::span{buffers.edge_attr_flat}, kEdgeDim
   );

   if(collect_instances) {
      if(config.include_hyperedge_incidence) {
         builder.set_graph_attr(
            "hyperedge_note",
            std::string(
               "members are object nodes, or the anchor node for an arity-0 "
               "instance, so every hyperedge has at least one member and "
               "hyperedge_index[1].max() + 1 equals the hyperedge count. The "
               "native staging fields are hyperedge_sizes / hyperedge_ids / "
               "hyperedge_node_indices, paired via "
               "repeat_interleave(hyperedge_ids, hyperedge_sizes); encode_pyg "
               "already applies that expansion and exposes the result as "
               "hyperedge_index [2, M] plus hyperedge_attr_ids [M, 6] and "
               "num_hyperedges, so the staging fields are absent there"
            )
         );
      }
      register_instance_fields(builder, config);
      set_instance_fields(builder, config, buffers.instances, anchor_index);
   }
   if(config.include_spd) {
      emit_spd_fields(builder, config, static_cast< int64_t >(objects.size()), buffers.instances);
   }

   // Per-graph scalars must be graph *fields*, not graph attrs: a graph attr
   // is batch-invariant metadata (BatchBuilder rejects a second, different
   // value), while these change with every graph.
   builder.register_field(
      std::string(kHistoryDtOffsetField),
      GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
   );
   builder.set_field(
      std::string(kHistoryDtOffsetField), std::span< const int64_t >{&history_dt_offset, 1}
   );
   if(emit_anchor) {
      builder.register_field(
         std::string(kAnchorIndexField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
            .inc = node_offset_inc(),
         }
      );
      builder.set_field(
         std::string(kAnchorIndexField), std::span< const int64_t >{&anchor_index, 1}
      );
   }

   builder.set_graph_attr("vocab_roles", role_vocabulary());
   builder.set_graph_attr("vocab_predicates", predicate_vocabulary(predicates));
   builder.set_graph_attr("vocab_actions", action_vocabulary(action_specs));
   builder.set_graph_attr("vocab_relations", relation_vocabulary(predicates, action_specs));
   builder.set_graph_attr("num_predicates", static_cast< int64_t >(predicates.size()));
   builder.set_graph_attr("has_anchor", static_cast< int64_t >(emit_anchor ? 1 : 0));
   builder.set_graph_attr("vocab_edge_kinds", edge_kind_vocabulary());
   builder.set_graph_attr("vocab_categories", category_vocabulary());
   builder.set_graph_attr("channel_names", channel_names());
   builder.set_graph_attr("edge_channel_names", edge_channel_names());
}

}  // namespace

SemanticDerivedGraphEncoderConfig normalize_semantic_derived_graph_encoder_config(
   SemanticDerivedGraphEncoderConfig config
)
{
   if(config.node_universe == DerivedNodeUniverse::objects_only) {
      config.include_tuple_tensors = true;
   }
   return config;
}

SemanticDerivedGraphEncoderEngine::SemanticDerivedGraphEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   SemanticDerivedGraphEncoderConfig config
)
    : schema_context_(require_schema_context(
         std::make_shared< const SemanticSchemaContext >(
            SemanticSchemaContext{.predicates = std::move(predicates)}
         )
      )),
      config_(normalize_semantic_derived_graph_encoder_config(config))
{
   validate_config(config_);
}

SemanticDerivedGraphEncoderEngine::SemanticDerivedGraphEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   SemanticDerivedGraphEncoderConfig config
)
    : schema_context_(require_schema_context(
         std::make_shared< const SemanticSchemaContext >(SemanticSchemaContext{
            .predicates = std::move(predicates),
            .actions = std::move(actions),
         })
      )),
      config_(normalize_semantic_derived_graph_encoder_config(config))
{
   validate_config(config_);
}

SemanticDerivedGraphEncoderEngine::SemanticDerivedGraphEncoderEngine(
   std::shared_ptr< const SemanticSchemaContext > schema,
   SemanticDerivedGraphEncoderConfig config
)
    : schema_context_(require_schema_context(schema)),
      config_(normalize_semantic_derived_graph_encoder_config(config))
{
   validate_config(config_);
}

const std::vector< SemanticPredicateSpec >&
SemanticDerivedGraphEncoderEngine::get_predicates() const
{
   return schema_context_->predicates;
}

const std::vector< SemanticActionSpec >& SemanticDerivedGraphEncoderEngine::get_actions() const
{
   return schema_context_->actions;
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
