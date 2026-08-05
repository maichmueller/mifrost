/** Pymimir compatibility facade for the semantic successor HGraph engine. */
#pragma once

#include <boost/describe.hpp>
#include <map>
#include <memory>
#include <string>

#include "hgraph_stream_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_successor_hgraph_encoder.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct SuccessorBatchInputs;
}
}  // namespace batch_input

class MIFROST_API SuccessorHGraphEncoderEngine: public HGraphEncoderEngine {
  public:
   enum class Mode { full, delta };
   struct Config: HGraphEncoderEngine::Config {
      Mode successor_mode = Mode::full;
      std::string successor_suffix = "[suc]";
      bool include_successor_goal_satisfaction = false;
   };

   SuccessorHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
   SuccessorHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
   SuccessorHGraphEncoderEngine(mimir::formalism::Domain domain);
   SuccessorHGraphEncoderEngine(mimir::formalism::Domain domain, Config config);

   void encode(
      const mimir::search::State& current,
      const mimir::search::State& successor,
      const GoalInputs& goals,
      BatchBuilder& builder
   );
   BatchBuilder::BatchEncoding encode_batch(
      const batch_input::parsed::SuccessorBatchInputs& inputs
   );
   [[nodiscard]] const Config& get_config() const { return successor_config_; }
   void update_relations(RelationDict relation_dict) override;

  private:
   static SemanticSuccessorHGraphEncoderConfig semantic_config(const Config& config);
   Config successor_config_;
   std::unique_ptr< SemanticSuccessorHGraphEncoderEngine > semantic_successor_;
   std::shared_ptr< const SemanticTaskContext > semantic_task_context_;
   std::map< std::string, int > semantic_relation_arities_;
};

BOOST_DESCRIBE_STRUCT(
   SuccessorHGraphEncoderEngine::Config,
   (HGraphEncoderEngine::Config),
   (successor_mode, successor_suffix, include_successor_goal_satisfaction)
)

struct TransitionStepInput {
   const mimir::search::State* current = nullptr;
   const mimir::search::State* successor = nullptr;
   const GoalInputs* goals = nullptr;
};

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
   );
   void update(
      int64_t id,
      const mimir::search::State& current,
      const mimir::search::State& successor,
      const GoalInputs& goals
   );
   void encode_step(const TransitionStepInput& step, BatchBuilder& builder);

  private:
   SuccessorHGraphEncoderEngine* engine_ = nullptr;
};

}  // namespace mifrost
