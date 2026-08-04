/** Planner-neutral homogeneous color encoder. */
#pragma once

#include <memory>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

namespace mifrost {

struct SemanticColorEncoderConfig {
   bool edge_features = false;
   bool enable_global_predicate_nodes = false;
   bool export_node_names = true;
};

class MIFROST_API SemanticColorEncoderEngine {
  public:
   SemanticColorEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      SemanticColorEncoderConfig config = {}
   );
   SemanticColorEncoderEngine(
      std::shared_ptr< const SemanticTaskContext > task_context,
      SemanticColorEncoderConfig config = {}
   );

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationInput& input) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
   ) const;
   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;

   [[nodiscard]] const std::shared_ptr< const SemanticTaskContext >& get_task_context() const
   {
      return task_context_;
   }
   [[nodiscard]] const SemanticColorEncoderConfig& get_config() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;

  private:
   std::shared_ptr< const SemanticTaskContext > task_context_;
   const std::vector< SemanticPredicateSpec >& predicates_;
   SemanticColorEncoderConfig config_;
};

}  // namespace mifrost
