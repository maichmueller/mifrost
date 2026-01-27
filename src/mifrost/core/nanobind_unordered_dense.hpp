#pragma once

#include <ankerl/unordered_dense.h>
#include <nanobind/stl/detail/nb_dict.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// Treat ankerl::unordered_dense::map like a Python dict at the boundary.
template<class Key, class T, class Hash, class KeyEqual, class AllocatorOrContainer, class Bucket, class BucketContainer, bool IsSegmented>
struct type_caster<ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, BucketContainer, IsSegmented>> :
    dict_caster<ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, BucketContainer, IsSegmented>, Key, T>
{
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
