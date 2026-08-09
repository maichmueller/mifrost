/** Planner-neutral flat Horizon encoder. */
#pragma once

#include <boost/describe.hpp>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/root_policy.hpp"
#include "mifrost/core/encoders/common/semantic_assembly.hpp"
#include "mifrost/core/encoders/flat/flat_composition.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_assembly.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_horizon_hgraph_encoder.hpp"
#include "mifrost/core/semantic/semantic_transition_dag.hpp"

namespace mifrost {

namespace detail {
struct SemanticFlatHorizonPreparedGraphStorage;
struct SemanticFlatHorizonPreparedGraphAccess;
}  // namespace detail

/**
 * Source- and symbol-compatible Horizon name for the shared annotation carrier.
 *
 * New cross-encoder code may use `SemanticAnnotations` directly. This wrapper
 * keeps the Horizon-specific public type and its exported `contains()` symbol.
 */
class MIFROST_API SemanticFlatHorizonAnnotations: public SemanticAnnotations {
  public:
   SemanticFlatHorizonAnnotations() = default;
   SemanticFlatHorizonAnnotations(SemanticAnnotations annotations)
       : SemanticAnnotations(std::move(annotations))
   {
   }

   template < typename T >
   void set(std::string key, std::shared_ptr< const T > value)
   {
      SemanticAnnotations::set< T >(std::move(key), std::move(value));
   }

   template < typename T, typename... Args >
   std::shared_ptr< const T > emplace(std::string key, Args&&... args)
   {
      return SemanticAnnotations::emplace< T >(std::move(key), std::forward< Args >(args)...);
   }

   template < typename T >
   [[nodiscard]] const T* find(std::string_view key) const
   {
      return SemanticAnnotations::find< T >(key);
   }

   template < typename T >
   [[nodiscard]] const T& get(std::string_view key) const
   {
      return SemanticAnnotations::get< T >(key);
   }

   [[nodiscard]] bool contains(std::string_view key) const;
   [[nodiscard]] bool empty() const noexcept { return SemanticAnnotations::empty(); }
   [[nodiscard]] size_t size() const noexcept { return SemanticAnnotations::size(); }
};

/**
 * Borrowed-or-owned graph input plus immutable graph-local extension data.
 *
 * A reference-backed input is valid for the duration of `encode` or
 * `encode_batch`. The shared-pointer constructor keeps the DAG alive itself.
 */
class MIFROST_API SemanticFlatHorizonInput: public SemanticEncoderInput< SemanticTransitionDAG > {
  public:
   using Base = SemanticEncoderInput< SemanticTransitionDAG >;

   explicit SemanticFlatHorizonInput(
      const SemanticTransitionDAG& graph,
      SemanticFlatHorizonAnnotations annotations = {}
   );
   SemanticFlatHorizonInput(SemanticTransitionDAG&&, SemanticFlatHorizonAnnotations = {}) = delete;
   SemanticFlatHorizonInput(const SemanticTransitionDAG&&, SemanticFlatHorizonAnnotations = {}) =
      delete;
   explicit SemanticFlatHorizonInput(
      std::shared_ptr< const SemanticTransitionDAG > graph,
      SemanticFlatHorizonAnnotations annotations = {}
   );

   [[nodiscard]] const SemanticTransitionDAG& graph() const;
};

/**
 * Stable read-only view shared by canonical and downstream Horizon components.
 *
 * The view owns compact graph-local preparation while borrowing the source DAG
 * for the duration of one synchronous encode operation.
 */
class MIFROST_API SemanticFlatHorizonPreparedGraph {
  public:
   SemanticFlatHorizonPreparedGraph(const SemanticFlatHorizonPreparedGraph&) = delete;
   SemanticFlatHorizonPreparedGraph& operator=(const SemanticFlatHorizonPreparedGraph&) = delete;
   SemanticFlatHorizonPreparedGraph(SemanticFlatHorizonPreparedGraph&&) noexcept = default;
   SemanticFlatHorizonPreparedGraph& operator=(SemanticFlatHorizonPreparedGraph&&) noexcept =
      default;
   ~SemanticFlatHorizonPreparedGraph();

   [[nodiscard]] const SemanticTransitionDAG& source_graph() const;
   [[nodiscard]] const SemanticFlatHorizonEncoderConfig& config() const;
   [[nodiscard]] const SemanticFlatHorizonAnnotations& annotations() const;
   [[nodiscard]] std::span< const SemanticGoalLevel > goal_levels() const;
   [[nodiscard]] std::span< const SemanticLiteral > goals() const;
   [[nodiscard]] int64_t entity_count() const;
   [[nodiscard]] std::span< const std::string > entity_names() const;
   [[nodiscard]] std::span< const int64_t > entity_role_ids() const;
   [[nodiscard]] std::span< const int64_t > object_entity_indices() const;
   [[nodiscard]] std::span< const int64_t > predicate_entity_indices() const;
   [[nodiscard]] std::span< const int64_t > state_entity_indices() const;
   [[nodiscard]] std::span< const int64_t > target_entity_indices() const;
   [[nodiscard]] std::span< const int64_t > target_entity_group_ids() const;
   [[nodiscard]] std::span< const int64_t > target_positions() const;
   [[nodiscard]] std::span< const int64_t > target_indices() const;
   [[nodiscard]] std::span< const int64_t > target_candidate_ids() const;
   [[nodiscard]] std::span< const int64_t > target_depths() const;
   [[nodiscard]] std::span< const int64_t > target_group_ids() const;
   [[nodiscard]] std::span< const std::string > target_names() const;
   [[nodiscard]] int64_t canonical_state_entity_index(int64_t source_node_index) const;
   [[nodiscard]] std::span< const SemanticLiteral > node_deltas(int64_t source_node_index) const;
   [[nodiscard]] bool node_contains_fact(int64_t source_node_index, const SemanticAtom& atom) const;
   [[nodiscard]] bool node_added_fluent(int64_t source_node_index, const SemanticAtom& atom) const;
   [[nodiscard]] bool
   node_removed_fluent(int64_t source_node_index, const SemanticAtom& atom) const;
   [[nodiscard]] bool node_added_derived(int64_t source_node_index, const SemanticAtom& atom) const;
   [[nodiscard]] bool
   node_removed_derived(int64_t source_node_index, const SemanticAtom& atom) const;

  private:
   friend struct detail::SemanticFlatHorizonPreparedGraphAccess;
   explicit SemanticFlatHorizonPreparedGraph(
      std::shared_ptr< detail::SemanticFlatHorizonPreparedGraphStorage > storage
   );
   std::shared_ptr< detail::SemanticFlatHorizonPreparedGraphStorage > storage_;
};

/** Runtime policy for planner-neutral flat Horizon encoding. */
struct SemanticFlatHorizonEncoderConfig: FlatRelationEncoderConfig {
   bool ignore_actions = true;
   SemanticHorizonMode transition_mode = SemanticHorizonMode::full;
   std::string parent_relation = defaults::parent_relation;
   std::string sibling_relation = defaults::sibling_relation;
   std::string cousin_relation = defaults::cousin_relation;
   bool enable_parent_relation = false;
   bool enable_sibling_relation = false;
   bool enable_cousin_relation = false;
   RootPolicy root_policy = RootPolicy::exclude;
};

BOOST_DESCRIBE_STRUCT(
   SemanticFlatHorizonEncoderConfig,
   (FlatRelationEncoderConfig),
   (ignore_actions,
    transition_mode,
    parent_relation,
    sibling_relation,
    cousin_relation,
    enable_parent_relation,
    enable_sibling_relation,
    enable_cousin_relation,
    root_policy)
)

/** Encode an owned semantic transition DAG into packed flat relations. */
class MIFROST_API SemanticFlatHorizonEncoderEngine {
  public:
   using Config = SemanticFlatHorizonEncoderConfig;

   SemanticFlatHorizonEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticFlatHorizonEncoderEngine(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config config = {}
   );
   SemanticFlatHorizonEncoderEngine(const SemanticFlatHorizonEncoderEngine&) = delete;
   SemanticFlatHorizonEncoderEngine& operator=(const SemanticFlatHorizonEncoderEngine&) = delete;
   SemanticFlatHorizonEncoderEngine(SemanticFlatHorizonEncoderEngine&&) noexcept;
   SemanticFlatHorizonEncoderEngine& operator=(SemanticFlatHorizonEncoderEngine&&) noexcept;
   ~SemanticFlatHorizonEncoderEngine();

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticTransitionDAG& dag) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatHorizonInput& input) const;
   void encode(const SemanticTransitionDAG& dag, BatchBuilder& builder) const;
   void encode(const SemanticFlatHorizonInput& input, BatchBuilder& builder) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticTransitionDAG >& dags
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::initializer_list< SemanticTransitionDAG > dags
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      std::span< const SemanticFlatHorizonInput > inputs
   ) const;
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;
   [[nodiscard]] const std::vector< SemanticActionSpec >& get_actions() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_names() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_arities() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_sources() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_logical_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_encoded_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_roles() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_role_offsets() const;
   [[nodiscard]] const std::vector< std::string >& get_slot_role_names() const;

  private:
   friend class SemanticFlatHorizonAssemblyBuilder;
   SemanticFlatHorizonEncoderEngine(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config config,
      std::vector< std::shared_ptr< FlatEmitterComponent > > components
   );
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

/** Mutable canonical Horizon assembly; `compile()` freezes all components. */
class MIFROST_API SemanticFlatHorizonAssemblyBuilder {
  public:
   using Config = SemanticFlatHorizonEncoderConfig;

   SemanticFlatHorizonAssemblyBuilder(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticFlatHorizonAssemblyBuilder(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config config = {}
   );
   SemanticFlatHorizonAssemblyBuilder(const SemanticFlatHorizonAssemblyBuilder&) = delete;
   SemanticFlatHorizonAssemblyBuilder& operator=(const SemanticFlatHorizonAssemblyBuilder&) =
      delete;
   SemanticFlatHorizonAssemblyBuilder(SemanticFlatHorizonAssemblyBuilder&&) noexcept;
   SemanticFlatHorizonAssemblyBuilder& operator=(SemanticFlatHorizonAssemblyBuilder&&) noexcept;
   ~SemanticFlatHorizonAssemblyBuilder();

   /** Transfer exclusive component ownership into this assembly. */
   void add_component(std::unique_ptr< FlatEmitterComponent > component);

   template < typename Component, typename... Args >
   void emplace_component(Args&&... args)
   {
      add_component(std::make_unique< Component >(std::forward< Args >(args)...));
   }

   [[nodiscard]] SemanticFlatHorizonEncoderEngine compile() &&;

  private:
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost
