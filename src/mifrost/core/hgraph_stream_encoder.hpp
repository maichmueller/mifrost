#pragma once

#include "batch_builder.hpp"
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>

#include <span>
#include <string>
#include <vector>

namespace mifrost {

/**
 * @brief Streams Mimir states into a BatchBuilder using HGraph semantics.
 *
 * Logic mirrors plangolin.encoding.hetero_encoder.HGraphEncoder.
 */
class HGraphStreamEncoder {
public:
  struct Config {
    std::string symbol_type_id = "_symbol_";
    bool ignore_actions = true;
    bool add_nullary_predicates = false;
    bool include_lgan_edges = false;
  };

  HGraphStreamEncoder(const mimir::formalism::Domain &domain);
  HGraphStreamEncoder(const mimir::formalism::Domain &domain, Config config);

  /**
   * @brief Encodes a single step (state, goals, acts) into the builder.
   *
   * @param state Current state
   * @param goals Goal literals (optional)
   * @param actions Action objects (if not ignored)
   * @param builder Target builder
   */
  void encode_step(const mimir::search::State &state,
                   std::span<const mimir::formalism::GroundLiteral<
                       mimir::formalism::FluentTag>>
                       goals,
                   std::span<const mimir::formalism::GroundAction> actions,
                   BatchBuilder &builder);

  void encode_state(const mimir::search::State &state, BatchBuilder &builder);

private:
  const mimir::formalism::Domain &domain_;
  Config config_;

  // --- Encoding Helpers ---
  void encode_objects(const mimir::search::State &state, BatchBuilder &builder);
  void encode_facts(const mimir::search::State &state, BatchBuilder &builder);
  void encode_goals(std::span<const mimir::formalism::GroundLiteral<
                        mimir::formalism::FluentTag>>
                        goals,
                    BatchBuilder &builder);
  // void encode_actions(...)
};

} // namespace mifrost
