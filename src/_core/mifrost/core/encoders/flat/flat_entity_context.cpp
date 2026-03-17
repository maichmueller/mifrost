/**
 * @file flat_entity_context.cpp
 * @brief Non-template helpers for flat entity-table bookkeeping.
 */
#include "flat_entity_context.hpp"

namespace mifrost {

auto PredicateSymbolKeyHash::operator()(const PredicateSymbolKey& key) const noexcept -> uint64_t
{
   return pack_state_fact_key(key.predicate_index, key.tag_id);
}

}  // namespace mifrost
