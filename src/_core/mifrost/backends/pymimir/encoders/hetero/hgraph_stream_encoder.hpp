/** Pymimir compatibility facade for the backend-neutral HGraph engine. */
#pragma once

#include <boost/describe.hpp>
#include <memory>
#include <mimir/formalism/domain.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "mifrost/backends/pymimir/encoders/common/goal_inputs.hpp"
#include "mifrost/backends/pymimir/encoders/common/relation_dict.hpp"
#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/stream_encoder_base.hpp"
#include "mifrost/core/encoders/common/target_source.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "semantic_hgraph_view_bridge.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct HGraphBatchInputs;
}
}  // namespace batch_input

class MIFROST_API HGraphEncoderEngine {
  public:
   using HistorySubgoal = std::pair< int, std::vector< LiteralVariant > >;

   struct Config {
      std::string symbol_type_id = defaults::symbol_type_id;
      std::string target_symbol_prefix = "target:";
      std::string nullary_object_name = "![nullary_symbol]!";
      std::string lgan_tn_edge_pos = defaults::lgan_tn_edge_pos;
      std::string lgan_nn_edge_pos = defaults::lgan_nn_edge_pos;
      std::string lgan_rr_edge_pos = defaults::lgan_rr_edge_pos;
      std::string history_link_relation = defaults::history_link_relation;
      size_t max_goal_level = 0;
      bool support_literals = false;
      bool add_nullary_predicates = false;
      bool ignore_actions = true;
      bool include_lgan_edges = false;
      bool include_static = true;
      bool include_empty_edge_types = true;
      bool export_node_names = true;
      std::set< TargetSource > lgan_anchor_sources = {};
      std::set< TargetSource > target_sources = {};
      std::set< GoalDerivation > goal_derivations = {
         GoalDerivation::plain,
         GoalDerivation::satisfied,
      };
   };

   explicit HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
   HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
   explicit HGraphEncoderEngine(mimir::formalism::Domain domain);
   HGraphEncoderEngine(mimir::formalism::Domain domain, Config config);
   virtual ~HGraphEncoderEngine();

   void encode(const mimir::search::State& state, BatchBuilder& builder);
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      const std::vector< HistorySubgoal >& history,
      std::optional< int > history_max_steps,
      BatchBuilder& builder
   );
   BatchBuilder::BatchEncoding encode_batch(
      const batch_input::parsed::HGraphBatchInputs& inputs,
      std::optional< int > history_max_steps
   );

   [[nodiscard]] const Config& get_config() const { return config_; }
   [[nodiscard]] const RelationDict& get_relation_dict() const { return relation_dict_; }
   virtual void update_relations(RelationDict relation_dict);

  protected:
   HGraphEncoderEngine(
      const mimir::formalism::DomainImpl& domain,
      Config config,
      SemanticHGraphEncoderConfig semantic_config
   );
   HGraphEncoderEngine(
      mimir::formalism::Domain domain,
      Config config,
      SemanticHGraphEncoderConfig semantic_config
   );
   [[nodiscard]] const mimir::formalism::DomainImpl& domain() const { return domain_; }
   [[nodiscard]] const pymimir::hetero_bridge::Schema& schema() const { return schema_; }
   [[nodiscard]] SemanticHGraphEncoderEngine& semantic_engine() { return *semantic_; }
   [[nodiscard]] const SemanticHGraphEncoderEngine& semantic_engine() const { return *semantic_; }
   [[nodiscard]] std::shared_ptr< const SemanticTaskContext > make_task_context(
      const mimir::search::State& state
   ) const;
   [[nodiscard]] const pymimir::views::Context& view_context(
      const mimir::search::State& state
   ) const;
   [[nodiscard]] SemanticFlatRelationInput make_input(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      const std::vector< HistorySubgoal >& history = {},
      std::optional< int > history_max_steps = std::nullopt
   ) const;
   [[nodiscard]] canonical::FlatRelationViewInput make_view_input(
      const mimir::search::State& state,
      const GoalInputs* goals = nullptr,
      std::span< const mimir::formalism::GroundAction > actions = {},
      std::span< const HistorySubgoal > history = {},
      std::optional< int > history_max_steps = std::nullopt
   ) const;
   void encode_semantic(
      const mimir::search::State& state,
      SemanticFlatRelationInput input,
      BatchBuilder& builder
   ) const;
   void encode_semantic_views(
      const mimir::search::State& state,
      canonical::FlatRelationViewInput input,
      BatchBuilder& builder
   ) const;
   void ensure_problem(const mimir::search::State& state) const;

   Config config_;
   mimir::formalism::Domain domain_holder_;
   const mimir::formalism::DomainImpl& domain_;
   pymimir::hetero_bridge::Schema schema_;
   mutable std::unique_ptr< SemanticHGraphEncoderEngine > semantic_;
   RelationDict relation_dict_;
   mutable std::unique_ptr< pymimir::SemanticProblemAdapter > problem_adapter_;
   mutable const mimir::formalism::ProblemImpl* problem_ = nullptr;
};

BOOST_DESCRIBE_STRUCT(
   HGraphEncoderEngine::Config,
   (),
   (symbol_type_id,
    target_symbol_prefix,
    nullary_object_name,
    lgan_tn_edge_pos,
    lgan_nn_edge_pos,
    lgan_rr_edge_pos,
    history_link_relation,
    max_goal_level,
    support_literals,
    add_nullary_predicates,
    ignore_actions,
    include_lgan_edges,
    include_static,
    include_empty_edge_types,
    export_node_names,
    lgan_anchor_sources,
    target_sources,
    goal_derivations)
)

struct HGraphStepInput {
   const mimir::search::State* state = nullptr;
   const GoalInputs* goals = nullptr;
   const std::vector< mimir::formalism::GroundAction >* actions = nullptr;
   const std::vector< HGraphEncoderEngine::HistorySubgoal >* history = nullptr;
   std::optional< int > history_max_steps;
};

class HGraphMutableStreamEncoder:
    public StreamEncoderBase< HGraphMutableStreamEncoder, HGraphStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "hetero"; }
   explicit HGraphMutableStreamEncoder(HGraphEncoderEngine& engine) : engine_(&engine) { reset(); }
   int64_t append(const mimir::search::State& state);
   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   );
   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
      std::optional< int > history_max_steps
   );
   void update(int64_t id, const mimir::search::State& state);
   void update(
      int64_t id,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   );
   void update(
      int64_t id,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
      std::optional< int > history_max_steps
   );
   void encode_step(const HGraphStepInput& step, BatchBuilder& builder);

  private:
   HGraphEncoderEngine* engine_ = nullptr;
};

class HGraphStreamEncoder {
  public:
   explicit HGraphStreamEncoder(HGraphEncoderEngine& engine) : engine_(&engine) { reset(); }
   int64_t append(const mimir::search::State& state);
   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   );
   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
      std::optional< int > history_max_steps
   );
   BatchEncoding flush();
   void reset();

  private:
   void ensure_valid() const;
   HGraphEncoderEngine* engine_ = nullptr;
   BatchBuilder builder_;
   int64_t next_id_ = 0;
};

}  // namespace mifrost
