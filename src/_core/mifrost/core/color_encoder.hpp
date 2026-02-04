#pragma once

#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "batch_builder.hpp"
#include "common_types.hpp"
#include "goal_inputs.hpp"
#include "relation_formatter.hpp"
#include "stream_encoder_base.hpp"

namespace mifrost {

class ColorEncoderEngine: public StreamEncoderBase< ColorEncoderEngine > {
  public:
   struct Config {
      bool edge_features = false;
      bool enable_global_predicate_nodes = false;
   };

   explicit ColorEncoderEngine(const mimir::formalism::DomainImpl& domain);
   ColorEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);

   explicit ColorEncoderEngine(mimir::formalism::Domain domain);
   ColorEncoderEngine(mimir::formalism::Domain domain, Config config);

   void encode_state(const mimir::search::State& state, BatchBuilder& builder) override
   {
      encode_state_impl(state, builder);
   }

   void encode(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state(state, builder);
   }

   void encode(const mimir::search::State& state, const GoalInputs& goals, BatchBuilder& builder)
   {
      encode_impl(state, goals, {}, builder);
   }

   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      encode_impl(state, goals, actions, builder);
   }

   const Config& get_config() const { return config_; }

  private:
   friend class StreamEncoderBase< ColorEncoderEngine >;

   void encode_state_impl(const mimir::search::State& state, BatchBuilder& builder);

   void encode_impl(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   std::optional< mimir::formalism::Domain > domain_holder_;
   const mimir::formalism::DomainImpl& domain_;
   Config config_;
};

}  // namespace mifrost
