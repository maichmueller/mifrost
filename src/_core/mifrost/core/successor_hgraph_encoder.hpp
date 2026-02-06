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

/**
 * @brief Payload for one streaming transition encode step.
 */
struct TransitionStepInput {
   const mimir::search::State* current = nullptr;
   const mimir::search::State* successor = nullptr;
   const GoalInputs* goals = nullptr;
};

/**
 * @brief Streaming transition encoder with static dispatch.
 */
class TransitionStreamEncoder:
    public StreamEncoderBase< TransitionStreamEncoder, TransitionStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "hetero"; }

   explicit TransitionStreamEncoder(SuccessorHGraphEncoderEngine& engine) : engine_(&engine)
   {
      reset();
   }

   int64_t append(
      const mimir::search::State& current,
      const mimir::search::State& successor,
      const GoalInputs& goals
   )
   {
      TransitionStepInput step;
      step.current = &current;
      step.successor = &successor;
      step.goals = &goals;
      return StreamEncoderBase::append(step);
   }

   void update(
      int64_t id,
      const mimir::search::State& current,
      const mimir::search::State& successor,
      const GoalInputs& goals
   )
   {
      TransitionStepInput step;
      step.current = &current;
      step.successor = &successor;
      step.goals = &goals;
      StreamEncoderBase::update(id, step);
   }

   void encode_step(const TransitionStepInput& step, BatchBuilder& builder)
   {
      if(engine_ == nullptr or step.current == nullptr or step.successor == nullptr
         or step.goals == nullptr) {
         throw std::invalid_argument("TransitionStreamEncoder requires current/successor/goals");
      }
      engine_->encode(*step.current, *step.successor, *step.goals, builder);
   }

  private:
   SuccessorHGraphEncoderEngine* engine_ = nullptr;
};

}  // namespace mifrost
