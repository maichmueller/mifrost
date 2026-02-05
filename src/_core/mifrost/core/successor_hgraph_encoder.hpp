#pragma once

#include "hgraph_stream_encoder.hpp"

namespace mifrost {

/**
 * @brief Encoder that handles state + immediate successor with delta or full representation.
 *
 * Logic mirrors plangolin.encoding.transition_hetero_encoder.TransitionHGraphEncoder.
 */
class SuccessorHGraphEncoderEngine: public HGraphEncoderEngine {
  public:
   enum class Mode {
      Full,  ///< Successor encodes full state.
      Delta  ///< Successor encodes only changed literals (add/delete).
   };

   /// Runtime config for successor transition encoding.
   struct Config: HGraphEncoderEngine::Config {
      Mode successor_mode = Mode::Full;
      std::string successor_suffix = "[suc]";
      bool include_successor_goal_satisfaction = false;
   };

   SuccessorHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
   SuccessorHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
   SuccessorHGraphEncoderEngine(mimir::formalism::Domain domain);
   SuccessorHGraphEncoderEngine(mimir::formalism::Domain domain, Config config);

   /**
    * @brief Encode current state and its immediate successor.
    */
   void encode(
      const mimir::search::State& current,
      const mimir::search::State& successor,
      const GoalInputs& goals,
      BatchBuilder& builder
   );

  private:
   /// Effective successor-specific config.
   Config successor_config_;

   /// Internal transition encode implementation.
   void encode_impl(
      const mimir::search::State& current,
      const mimir::search::State& successor,
      const GoalInputs& goals,
      BatchBuilder& builder
   );
};

}  // namespace mifrost
