/**
 * @file ids.hpp
 * @brief Compact identifiers shared by planning Views.
 */
#pragma once

#include <cstdint>

namespace mifrost::views {

using PredicateId = std::int64_t;
using ObjectId = std::int64_t;
using ActionSchemaId = std::int64_t;
using AtomId = std::int64_t;

enum class PredicateCategory : std::int8_t {
   static_predicate = 0,
   fluent = 1,
   derived = 2,
};

}  // namespace mifrost::views
