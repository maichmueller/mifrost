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

#include <cstdint>
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
#include <utility>
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
   [[nodiscard]] int64_t count(FlatNodeTypeId type) const;
   [[nodiscard]] const std::vector< std::string >& names(FlatNodeTypeId type) const;
   [[nodiscard]] const FlatNodeSchema& schema() const { return *schema_; }

  private:
   friend class FlatNodePlanBuilder;
   const FlatNodeSchema* schema_ = nullptr;
   std::vector< std::vector< std::string > > names_;
   std::vector< std::unordered_map< std::string, int64_t > > indices_;
};

/** Mutable graph-local node layout builder. */
class MIFROST_API FlatNodePlanBuilder {
  public:
   explicit FlatNodePlanBuilder(const FlatNodeSchema& schema) : schema_(schema) {}

   [[nodiscard]] FlatNodeRef add_node(FlatNodeTypeId type, std::string key);
   [[nodiscard]] FlatNodeRef add_node(std::string_view type_name, std::string key);
   [[nodiscard]] FlatNodePlan finish() &&;

  private:
   const FlatNodeSchema& schema_;
   std::vector< std::vector< std::string > > names_;
   std::vector< std::unordered_map< std::string, int64_t > > indices_;
};

enum class FlatSlotResolverKind : int8_t {
   source_slot,
   constant,
   node,
};

/** One output-tuple slot in a relation projection. */
struct FlatSlotResolver {
   FlatSlotResolverKind kind = FlatSlotResolverKind::source_slot;
   int source_slot = -1;
   int64_t constant = 0;
   FlatNodeRef node;

   static FlatSlotResolver source(int slot);
   static FlatSlotResolver constant_value(int64_t value);
   static FlatSlotResolver node_ref(FlatNodeRef ref);
};

/** Declarative relation remapping compiled into integer relation ids. */
struct FlatRelationProjection {
   RelationKey source_relation;
   RelationKey output_relation;
   std::vector< FlatSlotResolver > slots;
};

struct CompiledFlatRelationProjection {
   int source_relation_id = -1;
   int output_relation_id = -1;
   std::vector< FlatSlotResolver > slots;

   [[nodiscard]] std::vector< int64_t >
   project(std::span< const int64_t > source_args, const FlatNodePlan& nodes) const;
};

struct FlatFieldPlanEntry {
   std::string key;
   GraphFieldSpec spec;
   std::string owner;
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

struct FlatCompositionConfig {
   int max_goal_level = 0;
   bool support_literals = false;
   std::set< GoalDerivation > goal_derivations;
   std::string empty_schema_error = "Flat composition plan declares no relations";
   FlatBuilderGraphConfig graph_config;
   bool track_relation_instances = false;
   bool pack_relation_args_relation_major = true;
};

class MIFROST_API FlatSchemaPlanBuilder {
  public:
   void register_relation(RelationKey key, FlatTupleLayout layout, RelationUsage usage);
   void add_projection(FlatRelationProjection projection);
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

   [[nodiscard]] FlatRelationSchema finalize_schema(const FlatCompositionConfig& config) &&;
   [[nodiscard]] FlatNodeSchema finalize_nodes() && { return std::move(node_schema_).finalize(); }

  private:
   FlatRelationSchemaBuilder relation_schema_;
   FlatNodeSchemaBuilder node_schema_;
   std::vector< FlatRelationProjection > projections_;
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

struct FlatGraphContext {
   const FlatInputView& input;
   const FlatRelationSchema& schema;
   const FlatNodePlan& nodes;
   FlatRelationSink& relations;
   std::span< const CompiledFlatRelationProjection > projections;

   void emit(RelationKey key, std::span< const int64_t > args) const;
   void emit_projection(size_t projection_id, std::span< const int64_t > source_args) const;
};

/** Native component contract. Virtual dispatch occurs only once per phase/graph. */
class MIFROST_API FlatEmitterComponent {
  public:
   virtual ~FlatEmitterComponent();
   [[nodiscard]] virtual std::string_view name() const noexcept = 0;
   virtual void declare_schema(FlatSchemaPlanBuilder&) const {}
   virtual void declare_fields(FlatFieldPlanBuilder&) const {}
   virtual void plan_graph(const FlatInputView&, FlatNodePlanBuilder&) const {}
   virtual void prepare_graph(const FlatInputView&, FlatGraphContext&) const {}
   virtual void emit(const FlatInputView&, FlatGraphContext&) const {}
   virtual void write_fields(const FlatGraphContext&, FlatFieldWriter&) const {}
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
   CompiledFlatPlan() = default;

   [[nodiscard]] const FlatRelationSchema& schema() const { return schema_; }
   [[nodiscard]] const FlatNodeSchema& node_schema() const { return node_schema_; }
   [[nodiscard]] const std::vector< FlatFieldPlanEntry >& fields() const { return fields_; }
   [[nodiscard]] const std::vector< CompiledFlatRelationProjection >& projections() const
   {
      return projections_;
   }
   [[nodiscard]] const FlatCompositionConfig& config() const { return config_; }

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const FlatInputView& input) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const FlatInputView > inputs
   ) const;
   void encode(const FlatInputView& input, BatchBuilder& builder) const;
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

  private:
   friend class FlatEncoderPlan;
   FlatRelationSchema schema_;
   FlatNodeSchema node_schema_;
   std::vector< FlatFieldPlanEntry > fields_;
   std::vector< std::shared_ptr< FlatEmitterComponent > > components_;
   std::vector< CompiledFlatRelationProjection > projections_;
   FlatCompositionConfig config_;

   void configure_builder(BatchBuilder& builder) const;
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
