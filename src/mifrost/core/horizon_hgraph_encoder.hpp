#pragma once

#include "hgraph_stream_encoder.hpp"
#include "transition_dag.hpp"

namespace mifrost {

/**
 * @brief Encoder that handles lookahead transition graphs (OR-graphs) using HGraph semantics.
 *
 * Logic mirrors plangolin.encoding.horizon_hetero_encoder.HorizonHGraphEncoder.
 */
class HorizonHGraphEncoderEngine: public HGraphEncoderEngine {
  public:
   enum class Mode {
      Full,  // Each transition encodes full successor state
      Delta,  // Each transition encodes only changed literals vs root
      Action  // Only encodes actions, no state atoms for transitions
   };

   struct Config: HGraphEncoderEngine::Config {
      Mode transition_mode = Mode::Full;
      std::string target_symbol_prefix = "target:";
      std::string parent_relation = "parent";
      bool enable_parent_relation = false;
      bool enable_sibling_relation = false;
      bool exclude_root_candidate = true;
   };

   HorizonHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
   HorizonHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
   HorizonHGraphEncoderEngine(mimir::formalism::Domain domain);
   HorizonHGraphEncoderEngine(mimir::formalism::Domain domain, Config config);

   /**
    * @brief Encode a state and a DAG of lookahead transitions.
    */
   void encode(
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals,
      BatchBuilder& builder
   );

  private:
   Config horizon_config_;

   void encode_impl(
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals,
      BatchBuilder& builder
   );

   std::string target_node_key(int idx) const;
};

}  // namespace mifrost
