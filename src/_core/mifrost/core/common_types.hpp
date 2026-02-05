#pragma once

#include <ankerl/unordered_dense.h>

namespace mifrost {

/**
 * @brief Project-wide hash map alias.
 *
 * Uses ankerl::unordered_dense for cache-friendly performance.
 */

template < typename Key, typename T >
using hash_map = ankerl::unordered_dense::map< Key, T >;

/// Project-wide hash set alias.
template < typename Key >
using hash_set = ankerl::unordered_dense::set< Key >;

}  // namespace mifrost
