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

/**
 * @brief Color-encoding engine producing homogeneous graph payloads.
 */
class ColorEncoderEngine: public StreamEncoderBase< ColorEncoderEngine > {
  public:
   /// Runtime config for color encoding behavior.
   struct Config {
      /// Emit edge feature tensors.
      bool edge_features = false;
      /// Emit global predicate nodes in addition to atom/object nodes.
      bool enable_global_predicate_nodes = false;
   };

   explicit ColorEncoderEngine(const mimir::formalism::DomainImpl& domain);
   ColorEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);

   explicit ColorEncoderEngine(mimir::formalism::Domain domain);
   ColorEncoderEngine(mimir::formalism::Domain domain, Config config);

   /// StreamEncoderInterface entrypoint.
   void encode_state(const mimir::search::State& state, BatchBuilder& builder) override
   {
      encode_state_impl(state, builder);
   }

   /// Encode state-only into an existing builder.
   void encode(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state(state, builder);
   }

   /// Encode state + goals into an existing builder.
   void encode(const mimir::search::State& state, const GoalInputs& goals, BatchBuilder& builder)
   {
      encode_impl(state, goals, {}, builder);
   }

   /// Encode state + goals + actions into an existing builder.
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      encode_impl(state, goals, actions, builder);
   }

   /// Return effective engine config.
   const Config& get_config() const { return config_; }

  private:
   friend class StreamEncoderBase< ColorEncoderEngine >;

   /// Internal state-only encode implementation.
   void encode_state_impl(const mimir::search::State& state, BatchBuilder& builder);

   /// Internal full encode implementation.
   void encode_impl(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   /// Optional owning domain storage for handle-based construction.
   std::optional< mimir::formalism::Domain > domain_holder_;
   /// Active domain implementation reference.
   const mimir::formalism::DomainImpl& domain_;
   /// Effective runtime config.
   Config config_;
};

}  // namespace mifrost
