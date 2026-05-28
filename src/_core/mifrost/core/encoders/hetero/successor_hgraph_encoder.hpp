/**
 * @file successor_hgraph_encoder.hpp
 * @brief Heterogeneous immediate-successor encoder built on `HGraphEncoderEngine`.
 *
 * This file specializes the hetero encoder for a current state plus one
 * successor state instead of a full transition DAG.
 */
#pragma once

#include "hgraph_stream_encoder.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct SuccessorBatchInputs;
}
}  // namespace batch_input

/**
 * @brief Encoder that handles state + immediate successor with delta or full representation.
 *
 * Logic mirrors plangolin.encoding.transition_hetero_encoder.TransitionHGraphEncoder.
 */
class MIFROST_API SuccessorHGraphEncoderEngine: public HGraphEncoderEngine {
  public:
   enum class Mode {
      full,  ///< Successor encodes full state.
      delta  ///< Successor encodes only changed literals (add/delete).
   };

   /// Runtime config for successor transition encoding.
   struct Config: HGraphEncoderEngine::Config {
      Mode successor_mode = Mode::full;
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

   /// Encode a parsed batch input plan into one batch encoding.
   BatchBuilder::BatchEncoding encode_batch(
      const batch_input::parsed::SuccessorBatchInputs& inputs
   );

   /// Return effective successor config (includes inherited hgraph fields).
   const Config& get_config() const { return successor_config_; }

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

BOOST_DESCRIBE_STRUCT(
   SuccessorHGraphEncoderEngine::Config,
   (HGraphEncoderEngine::Config),
   (successor_mode, successor_suffix, include_successor_goal_satisfaction)
)

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
      // Successor streaming always needs both states and normalized goals.
      engine_->encode(*step.current, *step.successor, *step.goals, builder);
   }

  private:
   SuccessorHGraphEncoderEngine* engine_ = nullptr;
};

}  // namespace mifrost
