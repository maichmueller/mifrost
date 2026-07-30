/**
 * @file relation_catalog.hpp
 * @brief Encoder-independent semantic catalog of relation specifications.
 *
 * A `RelationCatalog` collects `RelationSpec` entries built purely from
 * `RelationKey`/logical-arity/usage values. It carries no backend-specific
 * layout (no flat slot roles, no hetero arity encoding) and performs no
 * string formatting or parsing, so encoder-agnostic transformations (such as
 * duplicating every predicate-derived relation with a modifier) operate on
 * structured values only.
 */
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "mifrost/core/encoders/common/relation_key.hpp"

namespace mifrost {

/// One catalog entry: a structured relation identity, its logical arity, and its usage.
struct RelationSpec {
   RelationKey key;
   int64_t logical_arity = 0;
   RelationUsage usage = RelationUsage::state;
};

/// Ordered collection of `RelationSpec` entries, backend-agnostic.
class RelationCatalog {
  public:
   void add(RelationSpec spec);

   [[nodiscard]] const std::vector< RelationSpec >& specs() const { return specs_; }
   [[nodiscard]] size_t size() const { return specs_.size(); }

  private:
   std::vector< RelationSpec > specs_;
};

/**
 * @brief Duplicate every predicate-derived relation with an added modifier.
 *
 * For every spec whose key has `family == RelationFamily::predicate`, appends a copy whose key's
 * `modifiers` has `modifier` added after any existing modifiers, preserving all other key fields
 * (polarity, goal level, derivation, state_anchored) and the original spec's logical arity and
 * usage. Non-predicate specs (actions, auxiliary/opaque relations) pass through unchanged. The
 * returned catalog contains both the original entries and their modifier-duplicated
 * counterparts; it never mutates `catalog`.
 */
[[nodiscard]] RelationCatalog duplicate_predicate_relations_with_modifier(
   const RelationCatalog& catalog,
   std::string_view modifier
);

}  // namespace mifrost
