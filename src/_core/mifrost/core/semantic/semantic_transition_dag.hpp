/**
 * @file semantic_transition_dag.hpp
 * @brief Owned, planning-backend-neutral transition DAG contract.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

namespace mifrost {

/** Candidate-id coverage over a selected set of DAG nodes. */
enum class SemanticCandidateIdCoverage : int64_t {
   none = 0,
   complete = 1,
   partial = 2,
};

/**
 * @brief Immutable-by-contract owned transition DAG.
 *
 * Nodes use stable contiguous indices. Edges are stored in deterministic
 * lexicographic order. Construction validates the complete semantic payload,
 * graph topology, reachability, and exact shortest-path depths.
 */
class MIFROST_API SemanticTransitionDAG {
  public:
   using Edge = std::pair< int64_t, int64_t >;

   struct Node {
      SemanticFlatRelationInput state;
      int64_t index = -1;
      int64_t depth = -1;
      std::optional< SemanticGroundAction > incoming_action = std::nullopt;
      std::optional< int64_t > candidate_id = std::nullopt;
      std::optional< std::vector< SemanticLiteral > > delta_literals = std::nullopt;
      std::optional< std::string > display_name = std::nullopt;
   };

   SemanticTransitionDAG(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      std::vector< Node > nodes,
      std::vector< Edge > edges
   );
   SemanticTransitionDAG(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      std::vector< Node > nodes,
      std::vector< Edge > edges,
      bool allow_non_dag_topology
   );

   [[nodiscard]] const std::vector< SemanticPredicateSpec >& predicates() const
   {
      return predicates_;
   }
   [[nodiscard]] const std::vector< SemanticActionSpec >& actions() const { return actions_; }
   [[nodiscard]] const std::vector< Node >& nodes() const { return nodes_; }
   [[nodiscard]] const std::vector< Edge >& edges() const { return edges_; }
   [[nodiscard]] const Node& root() const { return nodes_.front(); }
   [[nodiscard]] size_t size() const { return nodes_.size(); }

   [[nodiscard]] std::vector< int64_t > children(int64_t node_index) const;
   [[nodiscard]] std::vector< int64_t > parents(int64_t node_index) const;

   /** Candidate coverage, excluding the root by default for Horizon targets. */
   [[nodiscard]] SemanticCandidateIdCoverage candidate_id_coverage(bool include_root = false) const;

   /**
    * Validate Horizon's explicit-candidate contract.
    *
    * No candidate ids is valid and selects implicit indices. If any id is
    * present, every selected node must have one and all ids must be unique.
    */
   void validate_candidate_ids(bool include_root = false) const;

   /** Return validated explicit ids, or an empty vector when coverage is none. */
   [[nodiscard]] std::vector< int64_t > candidate_ids(bool include_root = false) const;

  private:
   std::vector< SemanticPredicateSpec > predicates_;
   std::vector< SemanticActionSpec > actions_;
   std::vector< Node > nodes_;
   std::vector< Edge > edges_;
};

}  // namespace mifrost
