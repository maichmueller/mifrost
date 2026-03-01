#pragma once

#include <string_view>

namespace mifrost::schema_key {

/// Separator between a type segment and attribute segment (e.g. `atom/x`).
inline constexpr char kTypeAttrSeparator = '/';
/// Separator between edge type segments (e.g. `src|rel|dst`).
inline constexpr char kEdgeTypeSeparator = '|';

inline constexpr std::string_view kTypeAttrSeparatorView = "/";
inline constexpr std::string_view kEdgeTypeSeparatorView = "|";

inline constexpr std::string_view kEdgeIndexAttrPrefix = "edge_index_";
inline constexpr std::string_view kEdgeIndexKeyPrefix = "/edge_index_";
inline constexpr char kEdgeIndexSrcComponent = '0';
inline constexpr char kEdgeIndexDstComponent = '1';

inline constexpr std::string_view kPtrAttr = "ptr";
inline constexpr std::string_view kBatchAttr = "batch";

}  // namespace mifrost::schema_key
