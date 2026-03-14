#pragma once

#include "default_relations.hpp"
#include "hgraph_stream_encoder.hpp"
#include "transition_dag.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct HorizonBatchInputs;
}
}  // namespace batch_input

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
      Config() { target_sources = {TargetSource::States}; }
      Mode transition_mode = Mode::Full;
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

   /// Encode a parsed batch input plan into one batch encoding.
   BatchBuilder::BatchEncoding encode_batch(const batch_input::parsed::HorizonBatchInputs& inputs);

   /// Return effective horizon config (includes inherited hgraph fields).
   const Config& get_config() const { return horizon_config_; }

   /// Replace relation dictionary and re-apply horizon-specific relation registrations.
   void update_relations(RelationDict relation_dict);

  private:
   /// Effective horizon-specific config.
   Config horizon_config_;

   /// Internal lookahead encode implementation.
   void encode_impl(
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals,
      BatchBuilder& builder,
      std::vector< mimir::search::State >* batch_target_name_states = nullptr
   );

   /// Register transition relation types based on config flags.
   void configure_relations();
   /// Register one transition relation in relation metadata.
   void register_relation_type(const std::string& relation);

   /// Build target-symbol node key for one transition node index.
   [[nodiscard]] std::string target_node_key(int idx) const;
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
    exclude_root_candidate)
)

/**
 * @brief Payload for one streaming horizon encode step.
 */
struct HorizonStepInput {
   const mimir::search::State* root = nullptr;
   const TransitionDAG* dag = nullptr;
   const GoalInputs* goals = nullptr;
};

/**
 * @brief Streaming horizon encoder with static dispatch.
 */
class HorizonStreamEncoder: public StreamEncoderBase< HorizonStreamEncoder, HorizonStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "hetero"; }

   explicit HorizonStreamEncoder(HorizonHGraphEncoderEngine& engine) : engine_(&engine) { reset(); }

   int64_t
   append(const mimir::search::State& root, const TransitionDAG& dag, const GoalInputs& goals)
   {
      HorizonStepInput step;
      step.root = &root;
      step.dag = &dag;
      step.goals = &goals;
      return StreamEncoderBase::append(step);
   }

   int64_t append(const mimir::search::State& root, const GoalInputs& goals)
   {
      TransitionDAG dag(root);
      HorizonStepInput step;
      step.root = &root;
      step.dag = &dag;
      step.goals = &goals;
      return StreamEncoderBase::append(step);
   }

   void update(
      int64_t id,
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals
   )
   {
      HorizonStepInput step;
      step.root = &root;
      step.dag = &dag;
      step.goals = &goals;
      StreamEncoderBase::update(id, step);
   }

   void update(int64_t id, const mimir::search::State& root, const GoalInputs& goals)
   {
      TransitionDAG dag(root);
      HorizonStepInput step;
      step.root = &root;
      step.dag = &dag;
      step.goals = &goals;
      StreamEncoderBase::update(id, step);
   }

   void encode_step(const HorizonStepInput& step, BatchBuilder& builder)
   {
      if(engine_ == nullptr or step.root == nullptr or step.dag == nullptr
         or step.goals == nullptr) {
         throw std::invalid_argument("HorizonStreamEncoder requires root/dag/goals");
      }
      engine_->encode(*step.root, *step.dag, *step.goals, builder);
   }

  private:
   HorizonHGraphEncoderEngine* engine_ = nullptr;
};

}  // namespace mifrost
