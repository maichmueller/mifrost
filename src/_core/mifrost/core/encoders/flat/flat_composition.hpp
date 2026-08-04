/**
 * @file flat_composition.hpp
 * @brief Native, backend-neutral composition primitives for flat encoders.
 *
 * The composition layer deliberately stops at the semantic flat carrier
 * boundary.  Planner adapters provide typed input views and components keep
 * their hot loops native; the runtime owns schema compilation, node identity,
 * field ownership, batching, and relation-argument finalization.
 */
#pragma once

#include <any>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "flat_encoder_common.hpp"
#include "flat_lgan.hpp"
#include "flat_relation_schema.hpp"
#include "mifrost/core/api.hpp"

namespace mifrost {

enum class FlatNodeKind : int8_t {
   object,
   transition,
   action,
   auxiliary,
   root_action,
   depth,
};

/**
 * Generic capabilities that a composed flat encoder may provide or require.
 *
 * Named downstream modes are intentionally not part of this backend-neutral
 * API. Adapters may define their own plans using these bits.
 */
enum class FlatCompositionCapability : uint32_t {
   state_facts = 1U << 0U,
   goal_facts = 1U << 1U,
   ground_actions = 1U << 2U,
   transition_effects = 1U << 3U,
   parent_relations = 1U << 4U,
   root_action_nodes = 1U << 5U,
   shared_state = 1U << 6U,
};

[[nodiscard]] constexpr uint32_t
operator|(FlatCompositionCapability lhs, FlatCompositionCapability rhs)
{
   return static_cast< uint32_t >(lhs) | static_cast< uint32_t >(rhs);
}

[[nodiscard]] constexpr uint32_t operator|(uint32_t lhs, FlatCompositionCapability rhs)
{
   return lhs | static_cast< uint32_t >(rhs);
}

[[nodiscard]] constexpr uint32_t operator|(FlatCompositionCapability lhs, uint32_t rhs)
{
   return static_cast< uint32_t >(lhs) | rhs;
}

/** Return capability bits required by a plan but not supplied by an adapter. */
[[nodiscard]] constexpr uint32_t
flat_composition_missing_capabilities(uint32_t required, uint32_t available)
{
   return required & ~available;
}

/** Return true when an adapter supplies every capability required by a plan. */
[[nodiscard]] constexpr bool
flat_composition_capabilities_satisfied(uint32_t required, uint32_t available)
{
   return flat_composition_missing_capabilities(required, available) == 0U;
}

using FlatNodeTypeId = int32_t;

/** A stable symbolic graph-local node reference. */
struct FlatNodeRef {
   FlatNodeTypeId type = -1;
   std::string key;

   auto operator==(const FlatNodeRef&) const -> bool = default;
};

struct FlatNodeTypeSpec {
   std::string name;
   FlatNodeKind kind = FlatNodeKind::auxiliary;
   int feature_dim = 1;
   bool export_names = false;

   auto operator==(const FlatNodeTypeSpec&) const -> bool = default;
};

/** Immutable node-type declarations shared by all graph plans. */
class MIFROST_API FlatNodeSchema {
  public:
   FlatNodeSchema() = default;

   [[nodiscard]] size_t size() const { return specs_.size(); }
   [[nodiscard]] const FlatNodeTypeSpec& spec(FlatNodeTypeId id) const;
   [[nodiscard]] const std::vector< FlatNodeTypeSpec >& specs() const { return specs_; }
   [[nodiscard]] FlatNodeTypeId id_for(std::string_view name) const;
   [[nodiscard]] std::optional< FlatNodeTypeId > try_id_for(std::string_view name) const;

  private:
   friend class FlatNodeSchemaBuilder;
   std::vector< FlatNodeTypeSpec > specs_;
   std::unordered_map< std::string, FlatNodeTypeId > ids_;
};

/** Mutable compile-time node-type declaration builder. */
class MIFROST_API FlatNodeSchemaBuilder {
  public:
   FlatNodeTypeId declare_node_type(
      std::string name,
      FlatNodeKind kind = FlatNodeKind::auxiliary,
      int feature_dim = 1,
      bool export_names = false
   );

   [[nodiscard]] FlatNodeSchema finalize() &&;

  private:
   FlatNodeSchema schema_;
};

/** Per-graph node layout after components have planned their symbolic nodes. */
class MIFROST_API FlatNodePlan {
  public:
   FlatNodePlan() = default;

   [[nodiscard]] int64_t index(FlatNodeRef ref) const;
   [[nodiscard]] int64_t index(FlatNodeTypeId type, std::string_view key) const;
   [[nodiscard]] int64_t index_from_source(FlatNodeTypeId type, int64_t source_index) const;
   [[nodiscard]] int64_t count(FlatNodeTypeId type) const;
   [[nodiscard]] const std::vector< std::string >& names(FlatNodeTypeId type) const;
   [[nodiscard]] const FlatNodeSchema& schema() const
   {
      if(schema_ == nullptr) {
         throw std::logic_error("Flat node plan is not initialized");
      }
      return *schema_;
   }

  private:
   friend class FlatNodePlanBuilder;
   const FlatNodeSchema* schema_ = nullptr;
   std::vector< std::vector< std::string > > names_;
   std::vector< std::unordered_map< std::string, int64_t > > indices_;
   std::vector< std::unordered_map< int64_t, int64_t > > indices_by_source_;
};

/** Mutable graph-local node layout builder. */
class MIFROST_API FlatNodePlanBuilder {
  public:
   explicit FlatNodePlanBuilder(const FlatNodeSchema& schema) : schema_(schema) {}
   FlatNodePlanBuilder(const FlatNodeSchema&&) = delete;

   [[nodiscard]] FlatNodeRef add_node(FlatNodeTypeId type, std::string key);
   [[nodiscard]] FlatNodeRef add_node(std::string_view type_name, std::string key);
   [[nodiscard]] FlatNodeRef
   add_node_from_source(FlatNodeTypeId type, int64_t source_index, std::string key);
   [[nodiscard]] FlatNodeRef
   add_node_from_source(std::string_view type_name, int64_t source_index, std::string key);
   [[nodiscard]] FlatNodePlan finish() &&;

  private:
   const FlatNodeSchema& schema_;
   std::vector< std::vector< std::string > > names_;
   std::vector< std::unordered_map< std::string, int64_t > > indices_;
   std::vector< std::unordered_map< int64_t, int64_t > > indices_by_source_;
   std::vector< std::vector< std::optional< int64_t > > > source_indices_by_node_;
};

enum class FlatSlotResolverKind : int8_t {
   source_slot,
   source_node,
   constant,
   node,
};

enum class FlatUnownedRelationPolicy : int8_t {
   unique_owner,
   broadcast,
};

struct FlatProjectionHandle {
   size_t component_id = std::numeric_limits< size_t >::max();
   std::string component;
   /** Stable declaration identity; unlike component_id, it survives plan recompilation. */
   std::string identity;

   [[nodiscard]] bool valid() const noexcept
   {
      return component_id != std::numeric_limits< size_t >::max() and not component.empty()
             and not identity.empty();
   }
};

/** One output-tuple slot in a relation projection. */
struct FlatSlotResolver {
   FlatSlotResolverKind kind = FlatSlotResolverKind::source_slot;
   int source_slot = -1;
   int64_t constant = 0;
   FlatNodeRef node;
   FlatNodeTypeId source_node_type = -1;

   static FlatSlotResolver source(int slot);
   static FlatSlotResolver source_node(int slot, FlatNodeTypeId node_type);
   static FlatSlotResolver constant_value(int64_t value);
   static FlatSlotResolver node_ref(FlatNodeRef ref);
};

/** Declarative relation remapping compiled into integer relation ids. */
struct FlatRelationProjection {
   RelationKey source_relation;
   RelationKey output_relation;
   std::vector< FlatSlotResolver > slots;
   std::string component;
   size_t component_id = std::numeric_limits< size_t >::max();
};

/** Alternate symbolic relation key resolved to one canonical relation id. */
struct FlatRelationAlias {
   RelationKey alias;
   RelationKey target;
   int target_relation_id = -1;
};

struct CompiledFlatRelationProjection {
   int source_relation_id = -1;
   int output_relation_id = -1;
   std::vector< FlatSlotResolver > slots;
   FlatProjectionHandle handle;

   [[nodiscard]] std::vector< int64_t >
   project(std::span< const int64_t > source_args, const FlatNodePlan& nodes) const;
   void project_into(
      std::span< int64_t > output,
      std::span< const int64_t > source_args,
      const FlatNodePlan& nodes
   ) const;
};

struct FlatFieldPlanEntry {
   std::string key;
   GraphFieldSpec spec;
   std::string owner;
};

/** One owner-declared node feature column. */
struct FlatNodeFeaturePlanEntry {
   std::string node_type;
   std::string attr;
   int feature_dim = 1;
   std::string owner;
};

/** Immutable ownership declarations for non-field graph metadata. */
struct FlatMetadataPlan {
   std::optional< std::string > object_names_owner;
   std::unordered_map< std::string, std::string > graph_attr_owners;
   std::unordered_map< std::string, std::string > optional_graph_attr_owners;
};

struct FlatRelationBinding {
   std::string component;
   std::vector< int > relation_ids;
};

struct FlatRelationDeclaration {
   std::string component;
   RelationKey key;
};

/** Immutable compiled schema portion of a flat composition plan. */
struct FlatSchemaPlan {
   FlatRelationSchema relation_schema;
   FlatNodeSchema node_schema;
   std::vector< FlatFieldPlanEntry > fields;
   std::vector< FlatNodeFeaturePlanEntry > node_features;
   std::vector< CompiledFlatRelationProjection > projections;
   FlatMetadataPlan metadata;
   std::vector< FlatRelationAlias > relation_aliases;
   std::vector< FlatRelationBinding > relation_bindings;
};

/** Compile-time field declarations with explicit single-owner semantics. */
class MIFROST_API FlatFieldPlanBuilder {
  public:
   explicit FlatFieldPlanBuilder(std::string owner) : owner_(std::move(owner)) {}

   void register_field(std::string key, GraphFieldSpec spec);
   [[nodiscard]] const std::vector< FlatFieldPlanEntry >& entries() const { return entries_; }

  private:
   std::string owner_;
   std::vector< FlatFieldPlanEntry > entries_;
   std::unordered_map< std::string, size_t > index_by_key_;
};

/** Compile-time ownership declarations for node feature columns. */
class MIFROST_API FlatNodeFeaturePlanBuilder {
  public:
   explicit FlatNodeFeaturePlanBuilder(std::string owner) : owner_(std::move(owner)) {}

   void register_feature(std::string node_type, std::string attr, int feature_dim);
   [[nodiscard]] const std::vector< FlatNodeFeaturePlanEntry >& entries() const { return entries_; }

  private:
   std::string owner_;
   std::vector< FlatNodeFeaturePlanEntry > entries_;
   std::unordered_map< std::string, size_t > index_by_key_;
};

/** Compile-time ownership declarations for non-field graph metadata. */
class MIFROST_API FlatMetadataPlanBuilder {
  public:
   explicit FlatMetadataPlanBuilder(std::string owner) : owner_(std::move(owner)) {}

   void claim_object_names();
   void claim_graph_attr(std::string key);
   void claim_optional_graph_attr(std::string key);
   [[nodiscard]] bool claims_object_names() const { return object_names_claimed_; }
   [[nodiscard]] const std::unordered_map< std::string, std::string >& graph_attrs() const
   {
      return graph_attrs_;
   }
   [[nodiscard]] const std::unordered_map< std::string, std::string >& optional_graph_attrs() const
   {
      return optional_graph_attrs_;
   }

  private:
   std::string owner_;
   bool object_names_claimed_ = false;
   std::unordered_map< std::string, std::string > graph_attrs_;
   std::unordered_map< std::string, std::string > optional_graph_attrs_;
};

struct FlatCompositionConfig {
   int max_goal_level = 0;
   bool support_literals = false;
   std::set< GoalDerivation > goal_derivations;
   std::string empty_schema_error = "Flat composition plan declares no relations";
   FlatBuilderGraphConfig graph_config;
   /// Node type whose graph-local offsets are applied to relation argument fields.
   std::string relation_args_node_type = std::string(kFlatEntityNodeType);
   /// Exported node type for the homogeneous flat entity index space.
   std::optional< std::string > entity_node_type;
   FlatUnownedRelationPolicy unowned_relation_policy = FlatUnownedRelationPolicy::unique_owner;
   bool track_relation_instances = false;
   bool pack_relation_args_relation_major = true;
};

class MIFROST_API FlatSchemaPlanBuilder {
  public:
   void begin_component(std::string component);
   void register_relation(RelationKey key, FlatTupleLayout layout, RelationUsage usage);
   void register_relation_alias(RelationKey alias, RelationKey target);
   [[nodiscard]] FlatProjectionHandle add_projection(FlatRelationProjection projection);
   [[nodiscard]] FlatNodeTypeId declare_node_type(
      std::string name,
      FlatNodeKind kind = FlatNodeKind::auxiliary,
      int feature_dim = 1,
      bool export_names = false
   );

   [[nodiscard]] FlatNodeSchemaBuilder& node_schema() { return node_schema_; }
   [[nodiscard]] const std::vector< FlatRelationProjection >& projections() const
   {
      return projections_;
   }
   [[nodiscard]] const std::vector< FlatRelationAlias >& relation_aliases() const
   {
      return relation_aliases_;
   }
   [[nodiscard]] const std::vector< FlatRelationProjection >& projection_declarations() const
   {
      return projections_;
   }
   [[nodiscard]] const std::vector< FlatRelationDeclaration >& relation_declarations() const
   {
      return relation_declarations_;
   }

   [[nodiscard]] FlatRelationSchema finalize_schema(const FlatCompositionConfig& config) &&;
   [[nodiscard]] FlatNodeSchema finalize_nodes() && { return std::move(node_schema_).finalize(); }

  private:
   FlatRelationSchemaBuilder relation_schema_;
   FlatNodeSchemaBuilder node_schema_;
   std::vector< FlatRelationProjection > projections_;
   std::vector< FlatRelationAlias > relation_aliases_;
   std::vector< FlatRelationDeclaration > relation_declarations_;
   std::string current_component_;
};

/** A type-erased, checked view used at component boundaries. */
class MIFROST_API FlatInputView {
  public:
   FlatInputView() = default;

   template < typename T >
   static FlatInputView from(const T& value)
   {
      return FlatInputView(&value, typeid(T));
   }

   template < typename T >
   static FlatInputView from(const T&&) = delete;

   template < typename T >
   [[nodiscard]] const T& get() const
   {
      if(type_ != std::type_index(typeid(T)) or data_ == nullptr) {
         throw std::invalid_argument("FlatInputView type mismatch");
      }
      return *static_cast< const T* >(data_);
   }

   [[nodiscard]] bool empty() const { return data_ == nullptr; }
   [[nodiscard]] std::type_index type() const { return type_; }

  private:
   FlatInputView(const void* data, const std::type_info& type) : data_(data), type_(type) {}
   const void* data_ = nullptr;
   std::type_index type_ = typeid(void);
};

class FlatSchemaPlanBuilder;
class FlatFieldPlanBuilder;
class FlatNodePlanBuilder;
class FlatFieldWriter;
class FlatMetadataWriter;
class FlatMetadataPlanBuilder;
class FlatNodeFeatureWriter;
class FlatNodeFeaturePlanBuilder;

/** One graph-local symbolic node supplied by a backend-neutral adapter. */
struct FlatCompositionNodeRecord {
   std::string node_type;
   std::string key;
   std::optional< int64_t > source_index;
};

/** One already-resolved relation tuple supplied to a native emitter. */
struct FlatCompositionRelationRecord {
   int relation_id = -1;
   std::vector< int64_t > args;
   /** Optional emitter owner; an empty owner is resolved by the configured ownership policy. */
   std::string component;
};

/** One typed graph-field value supplied to a native field component. */
struct FlatCompositionFieldRecord {
   std::string key;
   NumericColumnData values;
};

/**
 * Minimal backend-neutral graph input for the built-in native components.
 *
 * Adapters resolve relation keys to ids once per compiled plan and populate
 * `relations`; components never format names or search a relation dictionary
 * inside a fact/action loop.
 */
struct FlatCompositionInput {
   std::vector< std::string > objects;
   std::vector< FlatCompositionNodeRecord > nodes;
   std::vector< FlatCompositionRelationRecord > relations;
   std::vector< FlatCompositionFieldRecord > fields;
};

struct FlatCompositionRelationSpec {
   RelationKey key;
   FlatTupleLayout layout;
   RelationUsage usage = RelationUsage::state;
};

/** Graph-local preparation storage shared by all components for one graph. */
class MIFROST_API FlatGraphScratch {
  public:
   template < typename T, typename... Args >
   T& emplace(Args&&... args)
   {
      auto& value = values_[std::type_index(typeid(T))];
      if(value.has_value()) {
         throw std::invalid_argument("Flat graph scratch type was emplaced more than once");
      }
      value.emplace< T >(std::forward< Args >(args)...);
      return std::any_cast< T& >(value);
   }

   template < typename T >
   T& get()
   {
      const auto it = values_.find(std::type_index(typeid(T)));
      if(it == values_.end() or not it->second.has_value()) {
         throw std::invalid_argument("Flat graph scratch value is not prepared");
      }
      return std::any_cast< T& >(it->second);
   }

   template < typename T >
   const T& get() const
   {
      const auto it = values_.find(std::type_index(typeid(T)));
      if(it == values_.end() or not it->second.has_value()) {
         throw std::invalid_argument("Flat graph scratch value is not prepared");
      }
      return std::any_cast< const T& >(it->second);
   }

   void clear() { values_.clear(); }

  private:
   std::unordered_map< std::type_index, std::any > values_;
};

/**
 * Adapter-side builder for the backend-neutral composition carrier.
 *
 * Relation-key overloads are intended for setup code.  Hot loops should call
 * `relation_id()` once and append records with the integer-id overload.
 */
class MIFROST_API FlatCompositionInputBuilder {
  public:
   explicit FlatCompositionInputBuilder(const FlatRelationSchema& schema) : schema_(schema) {}
   FlatCompositionInputBuilder(const FlatRelationSchema&&) = delete;
   explicit FlatCompositionInputBuilder(const FlatSchemaPlan& plan)
       : schema_(plan.relation_schema), aliases_(plan.relation_aliases)
   {
   }
   FlatCompositionInputBuilder(const FlatSchemaPlan&&) = delete;

   [[nodiscard]] int relation_id(const RelationKey& key) const;
   void add_object(std::string name);
   void add_node(std::string node_type, std::string key);
   void add_node(std::string node_type, int64_t source_index, std::string key);
   void add_relation(int relation_id, std::span< const int64_t > args, std::string component = {});
   void add_relation(
      const RelationKey& key,
      std::span< const int64_t > args,
      std::string component = {}
   );
   void set_field(std::string key, NumericColumnData values);

   [[nodiscard]] FlatCompositionInput finish() && { return std::move(input_); }

  private:
   const FlatRelationSchema& schema_;
   std::vector< FlatRelationAlias > aliases_;
   FlatCompositionInput input_;
};

struct FlatGraphContext {
   const FlatInputView& input;
   const FlatRelationSchema& schema;
   const FlatNodePlan& nodes;
   FlatRelationSink& relations;
   std::span< const CompiledFlatRelationProjection > projections;
   std::span< const FlatRelationAlias > relation_aliases;
   std::span< const FlatRelationBinding > relation_bindings;
   FlatUnownedRelationPolicy unowned_relation_policy;
   FlatGraphScratch scratch;
   mutable std::vector< int64_t > projection_buffer;

   [[nodiscard]] int relation_id(const RelationKey& key) const;
   [[nodiscard]] bool component_owns_relation(std::string_view component, int relation_id) const;
   [[nodiscard]] bool should_emit_relation(std::string_view component, int relation_id) const;
   void emit(int relation_id, std::span< const int64_t > args) const;
   void emit(const RelationKey& key, std::span< const int64_t > args) const;
   void emit_projection(
      const FlatProjectionHandle& projection,
      std::span< const int64_t > source_args
   ) const;
};

/** Owner-scoped writer for native node feature columns. */
class MIFROST_API FlatNodeFeatureWriter {
  public:
   FlatNodeFeatureWriter(
      BatchBuilder& builder,
      const FlatNodeSchema& schema,
      const FlatNodePlan& nodes,
      const std::vector< FlatNodeFeaturePlanEntry >& plan,
      std::string_view owner
   )
       : builder_(builder), schema_(schema), nodes_(nodes), plan_(plan), owner_(owner)
   {
   }

   void
   set(std::string_view node_type, std::string_view attr, std::span< const float > values) const;
   /** Reject a component that declared a feature column but did not write it. */
   void finish() const;

  private:
   BatchBuilder& builder_;
   const FlatNodeSchema& schema_;
   const FlatNodePlan& nodes_;
   const std::vector< FlatNodeFeaturePlanEntry >& plan_;
   std::string owner_;
   mutable std::unordered_set< std::string > written_;
};

/** Native component contract. Virtual dispatch occurs only once per phase/graph. */
class MIFROST_API FlatEmitterComponent {
  public:
   virtual ~FlatEmitterComponent();
   [[nodiscard]] virtual std::string_view name() const noexcept = 0;
   virtual void declare_schema(FlatSchemaPlanBuilder&) const {}
   virtual void declare_fields(FlatFieldPlanBuilder&) const {}
   virtual void declare_node_features(FlatNodeFeaturePlanBuilder&) const {}
   virtual void plan_graph(const FlatInputView&, FlatNodePlanBuilder&) const {}
   virtual void prepare_graph(const FlatInputView&, FlatGraphContext&) const {}
   virtual void emit(const FlatInputView&, FlatGraphContext&) const {}
   virtual void write_node_features(const FlatGraphContext&, FlatNodeFeatureWriter&) const {}
   virtual void write_fields(const FlatGraphContext&, FlatFieldWriter&) const {}
   virtual void declare_metadata(FlatMetadataPlanBuilder&) const {}
   virtual void write_metadata(const FlatGraphContext&, FlatMetadataWriter&) const {}
};

/** Owner-scoped writer for non-field graph metadata emitted by a component. */
class MIFROST_API FlatMetadataWriter {
  public:
   FlatMetadataWriter(
      BatchBuilder& builder,
      const FlatMetadataPlan& plan,
      std::string_view owner,
      std::unordered_map< std::string, BatchBuilder::GraphAttrValue >* batch_constants = nullptr
   )
       : builder_(builder), plan_(plan), owner_(owner), batch_constants_(batch_constants)
   {
   }

   void set_object_names(std::vector< std::string > names) const;
   void add_lazy_target_names(std::span< const std::string > names) const;
   void set_graph_attr(std::string_view key, BatchBuilder::GraphAttrValue value) const;
   void set_graph_attr(std::string_view key, int64_t value) const;
   void set_graph_attr(std::string_view key, std::string value) const;
   void set_graph_attr(std::string_view key, std::vector< int64_t > value) const;
   void set_graph_attr(std::string_view key, std::vector< std::string > value) const;
   void finish() const;

  private:
   BatchBuilder& builder_;
   const FlatMetadataPlan& plan_;
   std::string owner_;
   std::unordered_map< std::string, BatchBuilder::GraphAttrValue >* batch_constants_ = nullptr;
   mutable std::unordered_set< std::string > written_graph_attrs_;
   mutable bool wrote_object_names_ = false;
};

/** Adds graph-local object rows from `FlatCompositionInput::objects`. */
class MIFROST_API FlatObjectNodeComponent final: public FlatEmitterComponent {
  public:
   explicit FlatObjectNodeComponent(
      std::string component_name = "objects",
      std::string node_type = std::string(kFlatEntityNodeType),
      FlatNodeKind kind = FlatNodeKind::object,
      int feature_dim = 1,
      bool export_names = true
   );

   [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
   void declare_schema(FlatSchemaPlanBuilder&) const override;
   void plan_graph(const FlatInputView&, FlatNodePlanBuilder&) const override;
   void declare_node_features(FlatNodeFeaturePlanBuilder&) const override;
   void declare_metadata(FlatMetadataPlanBuilder&) const override;
   void write_node_features(const FlatGraphContext&, FlatNodeFeatureWriter&) const override;
   void write_metadata(const FlatGraphContext&, FlatMetadataWriter&) const override;

  private:
   std::string component_name_;
   std::string node_type_;
   FlatNodeKind kind_;
   int feature_dim_;
   bool export_names_;
};

/** Adds typed transition/action/auxiliary rows from `FlatCompositionInput::nodes`. */
class MIFROST_API FlatNodeRecordComponent final: public FlatEmitterComponent {
  public:
   FlatNodeRecordComponent(
      std::string component_name,
      std::string node_type,
      FlatNodeKind kind,
      int feature_dim = 1,
      bool export_names = false
   );

   [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
   void declare_schema(FlatSchemaPlanBuilder&) const override;
   void plan_graph(const FlatInputView&, FlatNodePlanBuilder&) const override;
   void declare_node_features(FlatNodeFeaturePlanBuilder&) const override;
   void write_node_features(const FlatGraphContext&, FlatNodeFeatureWriter&) const override;

  private:
   std::string component_name_;
   std::string node_type_;
   FlatNodeKind kind_;
   int feature_dim_;
   bool export_names_;
};

/** Emits pre-resolved relation records and declares their immutable schema. */
class MIFROST_API FlatRelationEmitterComponent final: public FlatEmitterComponent {
  public:
   FlatRelationEmitterComponent(
      std::string component_name,
      std::vector< FlatCompositionRelationSpec > relations
   );

   [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
   void declare_schema(FlatSchemaPlanBuilder&) const override;
   void emit(const FlatInputView&, FlatGraphContext&) const override;

  private:
   std::string component_name_;
   std::vector< FlatCompositionRelationSpec > relations_;
};

/** Writes pre-resolved typed fields and enforces one component owner per key. */
class MIFROST_API FlatFieldEmitterComponent final: public FlatEmitterComponent {
  public:
   using FieldDeclaration = std::pair< std::string, GraphFieldSpec >;

   FlatFieldEmitterComponent(std::string component_name, std::vector< FieldDeclaration > fields);

   [[nodiscard]] std::string_view name() const noexcept override { return component_name_; }
   void declare_fields(FlatFieldPlanBuilder&) const override;
   void write_fields(const FlatGraphContext&, FlatFieldWriter&) const override;

  private:
   std::string component_name_;
   std::vector< FieldDeclaration > fields_;
};

/** Owner-checked writer for component-declared graph fields. */
class MIFROST_API FlatFieldWriter {
  public:
   FlatFieldWriter(
      BatchBuilder& builder,
      const std::vector< FlatFieldPlanEntry >& fields,
      std::string_view owner
   )
       : builder_(builder), fields_(fields), owner_(owner)
   {
   }

   void set(std::string_view key, std::span< const int64_t > values) const;
   void set(std::string_view key, std::span< const float > values) const;

  private:
   void validate(std::string_view key, GraphFieldDType dtype) const;
   BatchBuilder& builder_;
   const std::vector< FlatFieldPlanEntry >& fields_;
   std::string owner_;
};

class MIFROST_API CompiledFlatPlan {
  public:
   CompiledFlatPlan() = delete;

   [[nodiscard]] const FlatSchemaPlan& schema_plan() const { return schema_plan_; }
   [[nodiscard]] const FlatRelationSchema& schema() const { return schema_plan_.relation_schema; }
   [[nodiscard]] const FlatNodeSchema& node_schema() const { return schema_plan_.node_schema; }
   [[nodiscard]] const std::vector< FlatFieldPlanEntry >& fields() const
   {
      return schema_plan_.fields;
   }
   [[nodiscard]] const std::vector< FlatNodeFeaturePlanEntry >& node_features() const
   {
      return schema_plan_.node_features;
   }
   [[nodiscard]] const std::vector< CompiledFlatRelationProjection >& projections() const
   {
      return schema_plan_.projections;
   }
   [[nodiscard]] const FlatCompositionConfig& config() const { return config_; }

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const FlatInputView& input) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const FlatInputView > inputs
   ) const;
   /** Write one graph without committing it; the caller owns `next_graph()`. */
   void append_graph(const FlatInputView& input, BatchBuilder& builder) const;
   void encode(const FlatInputView& input, BatchBuilder& builder) const;
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

  private:
   friend class FlatEncoderPlan;
   CompiledFlatPlan(
      FlatSchemaPlan schema_plan,
      std::vector< std::shared_ptr< FlatEmitterComponent > > components,
      FlatCompositionConfig config
   )
       : schema_plan_(std::move(schema_plan)),
         components_(std::move(components)),
         config_(std::move(config))
   {
   }
   FlatSchemaPlan schema_plan_;
   std::vector< std::shared_ptr< FlatEmitterComponent > > components_;
   FlatCompositionConfig config_;

   void configure_builder(BatchBuilder& builder) const;
   void encode_graph(
      const FlatInputView& input,
      BatchBuilder& builder,
      std::unordered_map< std::string, BatchBuilder::GraphAttrValue >& batch_constants
   ) const;
};

/**
 * Reusable executor façade for a compiled plan.
 *
 * Keeping execution separate from plan construction lets a downstream
 * encoder share one compiled schema across worker runtimes without recompiling
 * relation projections or field ownership metadata. The shared pointer makes
 * the compiled plan's lifetime explicit.
 */
class MIFROST_API FlatBatchRuntime {
  public:
   explicit FlatBatchRuntime(std::shared_ptr< const CompiledFlatPlan > plan);

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const FlatInputView& input) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const FlatInputView > inputs
   ) const;
   void append_graph(const FlatInputView& input, BatchBuilder& builder) const;
   void encode(const FlatInputView& input, BatchBuilder& builder) const;
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

  private:
   std::shared_ptr< const CompiledFlatPlan > plan_;
};

/// Compatibility name used by the architectural design document.
using FlatCompositionKernel = FlatBatchRuntime;

/**
 * Result of an exact native carrier comparison.
 *
 * A non-empty `mismatch` identifies the first field whose value, dtype, shape,
 * or metadata differs.  This is intended for migration fixtures: an adapter
 * can compare its composed output with the legacy encoder before changing the
 * public path, without relying on Python tensor conversion.
 */
struct MIFROST_API FlatBatchParityResult {
   bool equal = true;
   std::string mismatch;
};

/** Compare two fully materialized native encodings field-for-field. */
[[nodiscard]] MIFROST_API FlatBatchParityResult compare_flat_batch_encodings(
   const BatchBuilder::BatchEncoding& expected,
   const BatchBuilder::BatchEncoding& actual
);

/**
 * Shared relation-major finalizer used by composed and legacy flat encoders.
 *
 * It owns a copy of the compiled encoded arities, so a writer can be retained
 * alongside an encoder plan without depending on a temporary schema view.
 */
class MIFROST_API FlatRelationMajorWriter {
  public:
   FlatRelationMajorWriter(std::span< const int64_t > relation_arities, bool enabled = true)
       : relation_arities_(relation_arities.begin(), relation_arities.end()), enabled_(enabled)
   {
   }

   void finalize(BatchBuilder::BatchEncoding& encoding) const;

  private:
   std::vector< int64_t > relation_arities_;
   bool enabled_ = true;
};

/** Mutable component assembly; compilation freezes all symbolic lookups. */
class MIFROST_API FlatEncoderPlan {
  public:
   void add_component(std::shared_ptr< FlatEmitterComponent > component);

   template < typename Component, typename... Args >
   void emplace_component(Args&&... args)
   {
      add_component(std::make_shared< Component >(std::forward< Args >(args)...));
   }

   [[nodiscard]] CompiledFlatPlan compile(FlatCompositionConfig config = {}) const;
   [[nodiscard]] size_t component_count() const { return components_.size(); }

  private:
   std::vector< std::shared_ptr< FlatEmitterComponent > > components_;
};

}  // namespace mifrost
