/**
 * @file semantic_flat_assembly.hpp
 * @brief Encoder-neutral ownership carrier for semantic flat components.
 */
#pragma once

#include "mifrost/core/encoders/common/semantic_assembly.hpp"
#include "mifrost/core/encoders/flat/flat_composition.hpp"

namespace mifrost {

/**
 * Components awaiting transfer into any concrete semantic flat assembly.
 *
 * Backend adapters use this carrier when they know how to obtain the semantic
 * schema but downstream native code owns the additional components. The
 * adapter consumes the carrier exactly once and transfers every component into
 * the concrete encoder builder before compilation.
 */
using SemanticFlatAssemblyComponents = SemanticAssemblyComponents< FlatEmitterComponent >;

}  // namespace mifrost
