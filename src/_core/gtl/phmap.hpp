#pragma once

#ifdef MIFROST_USE_EXTERNAL_GTL

   // Tyr/Yggdrasil ships the complete GTL implementation under this same include
   // path. Delegate to the next include root so this legacy Mimir compatibility
   // shim does not shadow it in a both-backend build.
   #include_next <gtl/phmap.hpp>

#else

   #include <parallel_hashmap/phmap.h>

namespace gtl {

using phmap::flat_hash_map;
using phmap::flat_hash_set;
using phmap::parallel_flat_hash_map;
using phmap::parallel_flat_hash_set;

}  // namespace gtl

#endif
