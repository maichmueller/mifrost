/** Pymimir compatibility facade for the semantic Horizon HGraph engine. */
#pragma once

#include <boost/describe.hpp>
#include <memory>
#include <string>

#include "hgraph_stream_encoder.hpp"
#include "mifrost/backends/pymimir/encoders/common/transition_dag.hpp"
#include "mifrost/core/encoders/common/root_policy.hpp"
#include "mifrost/core/encoders/hetero/semantic_horizon_hgraph_encoder.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct HorizonBatchInputs;
}
}  // namespace batch_input

class MIFROST_API HorizonHGraphEncoderEngine: public HGraphEncoderEngine {
  public:
   enum class Mode { full, delta, action };
   struct Config: HGraphEncoderEngine::Config {
      Config() { target_sources = {TargetSource::states}; }
      Mode transition_mode = Mode::full;
      std::string parent_relation = defaults::parent_relation;
      std::string sibling_relation = defaults::sibling_relation;
      std::string cousin_relation = defaults::cousin_relation;
      bool enable_parent_relation = false;
      bool enable_sibling_relation = false;
      bool enable_cousin_relation = false;
      RootPolicy root_policy = RootPolicy::exclude;
   };

   HorizonHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
   HorizonHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
   HorizonHGraphEncoderEngine(mimir::formalism::Domain domain);
   HorizonHGraphEncoderEngine(mimir::formalism::Domain domain, Config config);

   void encode(
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals,
      BatchBuilder& builder
   );
   BatchBuilder::BatchEncoding encode_batch(const batch_input::parsed::HorizonBatchInputs& inputs);
   [[nodiscard]] const Config& get_config() const { return horizon_config_; }
   void update_relations(RelationDict relation_dict) override;

  private:
   static SemanticHorizonHGraphEncoderConfig semantic_config(const Config& config);
   static SemanticTransitionDAG materialize_dag(
      const TransitionDAG& dag,
      const std::shared_ptr< const SemanticTaskContext >& context,
      const pymimir::hetero_bridge::Schema& schema,
      const pymimir::views::Context& view_context,
      const GoalInputs& goals
   );
   Config horizon_config_;
   std::unique_ptr< SemanticHorizonHGraphEncoderEngine > semantic_horizon_;
};

BOOST_DESCRIBE_STRUCT(
   HorizonHGraphEncoderEngine::Config,
   (HGraphEncoderEngine::Config),
   (transition_mode,
    parent_relation,
    sibling_relation,
    cousin_relation,
    enable_parent_relation,
    enable_sibling_relation,
    enable_cousin_relation,
    root_policy)
)

struct HorizonStepInput {
   const mimir::search::State* root = nullptr;
   const TransitionDAG* dag = nullptr;
   const GoalInputs* goals = nullptr;
   std::shared_ptr< TransitionDAG > owned_dag;
};

class HorizonStreamEncoder: public StreamEncoderBase< HorizonStreamEncoder, HorizonStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "hetero"; }
   explicit HorizonStreamEncoder(HorizonHGraphEncoderEngine& engine) : engine_(&engine) { reset(); }
   int64_t
   append(const mimir::search::State& root, const TransitionDAG& dag, const GoalInputs& goals);
   int64_t append(const mimir::search::State& root, const GoalInputs& goals);
   void update(
      int64_t id,
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals
   );
   void update(int64_t id, const mimir::search::State& root, const GoalInputs& goals);
   void encode_step(const HorizonStepInput& step, BatchBuilder& builder);

  private:
   HorizonHGraphEncoderEngine* engine_ = nullptr;
};

}  // namespace mifrost
