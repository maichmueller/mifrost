#pragma once

#include "default_relations.hpp"
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
      Full,  ///< Each transition encodes full successor state.
      Delta,  ///< Each transition encodes only changed literals vs root.
      Action  ///< Only encodes actions, no state atoms for transitions.
   };

   /// Runtime config for horizon lookahead encoding.
   struct Config: HGraphEncoderEngine::Config {
      Mode transition_mode = Mode::Full;
      std::string target_symbol_prefix = "target:";
      std::string parent_relation = defaults::parent_relation;
      std::string sibling_relation = defaults::sibling_relation;
      std::string cousin_relation = defaults::cousin_relation;
      bool enable_parent_relation = false;
      bool enable_sibling_relation = false;
      bool enable_cousin_relation = false;
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
   /// Effective horizon-specific config.
   Config horizon_config_;

   /// Internal lookahead encode implementation.
   void encode_impl(
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals,
      BatchBuilder& builder
   );

   /// Register transition relation types based on config flags.
   void configure_relations();
   /// Register one transition relation in relation metadata.
   void register_relation_type(const std::string& relation);

   /// Build target-symbol node key for one transition node index.
   [[nodiscard]] std::string target_node_key(int idx) const;
};

}  // namespace mifrost
