#pragma once

#include <ankerl/unordered_dense.h>

namespace mifrost
{

// Type aliases for hash-based containers.
// Using ankerl::unordered_dense for better performance and memory efficiency.
// Can be easily swapped to std::unordered_map/set if needed.

template<typename Key, typename T>
using hash_map = ankerl::unordered_dense::map<Key, T>;

template<typename Key>
using hash_set = ankerl::unordered_dense::set<Key>;

}  // namespace mifrost
