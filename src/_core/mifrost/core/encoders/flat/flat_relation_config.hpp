/**
 * @file flat_relation_config.hpp
 * @brief Planning-backend-neutral flat relation encoder configuration.
 */
#pragma once

#include <boost/describe.hpp>
#include <cstddef>
#include <set>
#include <string>

#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/goal_derivation.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/encoders/common/target_source.hpp"

namespace mifrost {

/**
 * @brief Runtime configuration shared by native and semantic flat encoders.
 *
 * The type deliberately contains only encoding policy.  Backend-specific
 * domain/state handles belong to the engine adapters, never this contract.
 */
struct FlatRelationEncoderConfig {
   size_t max_goal_level = 0;
   bool support_literals = false;
   bool include_static = true;
   bool export_node_names = true;
   bool ignore_zero_arity_relations = true;
   bool use_predicate_virtual_nodes = false;
   bool include_lgan_edges = false;
   std::set< TargetSource > lgan_anchor_sources = {};
   std::set< TargetSource > target_sources = {};
   std::string target_symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
   std::string lgan_tn_edge_pos = defaults::lgan_tn_edge_pos;
   std::string lgan_nn_edge_pos = defaults::lgan_nn_edge_pos;
   std::string lgan_rr_edge_pos = defaults::lgan_rr_edge_pos;
   bool pack_relation_args_relation_major = false;
   std::set< GoalDerivation > goal_derivations = {
      GoalDerivation::plain,
      GoalDerivation::satisfied,
   };
};

BOOST_DESCRIBE_STRUCT(
   FlatRelationEncoderConfig,
   (),
   (max_goal_level,
    support_literals,
    include_static,
    export_node_names,
    ignore_zero_arity_relations,
    use_predicate_virtual_nodes,
    include_lgan_edges,
    lgan_anchor_sources,
    target_sources,
    target_symbol_prefix,
    lgan_tn_edge_pos,
    lgan_nn_edge_pos,
    lgan_rr_edge_pos,
    pack_relation_args_relation_major,
    goal_derivations)
)

}  // namespace mifrost
