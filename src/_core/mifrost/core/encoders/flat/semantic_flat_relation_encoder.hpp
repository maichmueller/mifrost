/**
 * @file semantic_flat_relation_encoder.hpp
 * @brief Planning-backend-neutral native flat relation encoder.
 */
#pragma once

#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "flat_relation_config.hpp"
#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"

namespace mifrost {

/** Category of a predicate in the semantic input schema. */
enum class SemanticPredicateCategory : int64_t {
   static_predicate = 0,
   fluent = 1,
   derived = 2,
};

/** One predicate declaration. Predicate indices refer to this vector. */
struct SemanticPredicateSpec {
   SemanticPredicateCategory category = SemanticPredicateCategory::fluent;
   std::string name;
   int64_t arity = 0;

   auto operator<=>(const SemanticPredicateSpec&) const = default;
};

/** One action declaration. Action indices refer to this vector. */
struct SemanticActionSpec {
   std::string name;
   int64_t arity = 0;

   auto operator<=>(const SemanticActionSpec&) const = default;
};

/** Ground atom using schema-local predicate and graph-local object indices. */
struct SemanticAtom {
   int64_t predicate = -1;
   std::vector< int64_t > arguments;

   auto operator<=>(const SemanticAtom&) const = default;
};

/** Polarized ground literal. Its category is defined by the predicate schema. */
struct SemanticLiteral {
   SemanticAtom atom;
   bool positive = true;

   auto operator<=>(const SemanticLiteral&) const = default;
};

/** Ground action using schema-local action and graph-local object indices. */
struct SemanticGroundAction {
   int64_t action = -1;
   std::vector< int64_t > arguments;

   auto operator<=>(const SemanticGroundAction&) const = default;
};

/** One history row before stable time-delta ordering and distance filtering. */
struct SemanticHistoryEntry {
   int64_t dt = 0;
   std::vector< SemanticLiteral > literals;

   auto operator<=>(const SemanticHistoryEntry&) const = default;
};

/**
 * @brief Owned semantic input for one flat graph.
 *
 * All numeric indices are deliberately local to this value and its engine
 * schema. They are compact transport into the hot native loop, not persistent
 * or cross-repository identities.
 */
struct SemanticFlatRelationInput {
   std::vector< std::string > objects;
   std::vector< SemanticAtom > state_facts;
   std::vector< SemanticLiteral > goals;
   std::vector< SemanticGroundAction > actions;
   std::vector< std::vector< SemanticLiteral > > subgoal_layers;
   std::vector< SemanticHistoryEntry > history;
   std::optional< int64_t > history_max_steps = std::nullopt;
};

/**
 * @brief Encode owned semantic values directly to native BatchEncoding.
 *
 * The engine owns only predicate/action schema and encoding policy. It never
 * accepts or retains a Mimir, Tyr, or other planning-repository view.
 */
class MIFROST_API SemanticFlatRelationEncoderEngine {
  public:
   using Config = FlatRelationEncoderConfig;

   SemanticFlatRelationEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticFlatRelationEncoderEngine(const SemanticFlatRelationEncoderEngine&) = delete;
   SemanticFlatRelationEncoderEngine& operator=(const SemanticFlatRelationEncoderEngine&) = delete;
   SemanticFlatRelationEncoderEngine(SemanticFlatRelationEncoderEngine&&) noexcept;
   SemanticFlatRelationEncoderEngine& operator=(SemanticFlatRelationEncoderEngine&&) noexcept;
   ~SemanticFlatRelationEncoderEngine();

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationInput& input) const;
   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
   ) const;
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;
   [[nodiscard]] const std::vector< SemanticActionSpec >& get_actions() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_names() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_arities() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_sources() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_logical_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_encoded_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_roles() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_role_offsets() const;
   [[nodiscard]] const std::vector< std::string >& get_slot_role_names() const;

  private:
   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost
