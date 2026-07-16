/** Planner-neutral multi-step Horizon heterogeneous graph encoder. */
#pragma once

#include <boost/describe.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/root_policy.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/semantic/semantic_transition_dag.hpp"

namespace mifrost {

enum class SemanticHorizonMode {
   full,
   delta,
   action,
};

/** Runtime policy for `SemanticHorizonHGraphEncoderEngine`. */
struct SemanticHorizonHGraphEncoderConfig: SemanticHGraphEncoderConfig {
   SemanticHorizonHGraphEncoderConfig() { target_sources = {TargetSource::states}; }

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
   SemanticHorizonHGraphEncoderConfig,
   (SemanticHGraphEncoderConfig),
   (transition_mode,
    parent_relation,
    sibling_relation,
    cousin_relation,
    enable_parent_relation,
    enable_sibling_relation,
    enable_cousin_relation,
    root_policy)
)

/** Encode a validated owned semantic transition DAG as a Horizon HGraph. */
class MIFROST_API SemanticHorizonHGraphEncoderEngine {
  public:
   using Config = SemanticHorizonHGraphEncoderConfig;

   SemanticHorizonHGraphEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticHorizonHGraphEncoderEngine(const SemanticHorizonHGraphEncoderEngine&) = delete;
   SemanticHorizonHGraphEncoderEngine& operator=(const SemanticHorizonHGraphEncoderEngine&) =
      delete;
   SemanticHorizonHGraphEncoderEngine(SemanticHorizonHGraphEncoderEngine&&) noexcept;
   SemanticHorizonHGraphEncoderEngine& operator=(SemanticHorizonHGraphEncoderEngine&&) noexcept;
   ~SemanticHorizonHGraphEncoderEngine();

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticTransitionDAG& dag) const;
   void encode(const SemanticTransitionDAG& dag, BatchBuilder& builder) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticTransitionDAG >& dags
   ) const;

   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;
   [[nodiscard]] const std::vector< SemanticActionSpec >& get_actions() const;
   [[nodiscard]] const std::map< std::string, int >& get_relation_arities() const;

  private:
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost
