/** Planner-neutral flat Horizon encoder. */
#pragma once

#include <boost/describe.hpp>
#include <memory>
#include <string>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/root_policy.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_horizon_hgraph_encoder.hpp"
#include "mifrost/core/semantic/semantic_transition_dag.hpp"

namespace mifrost {

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
   void encode(const SemanticTransitionDAG& dag, BatchBuilder& builder) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticTransitionDAG >& dags
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
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost
