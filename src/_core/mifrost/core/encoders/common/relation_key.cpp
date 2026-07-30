#include "relation_key.hpp"

#include <array>
#include <functional>

namespace mifrost {

namespace {

constexpr std::string_view kPositivePrefix = "[+]";
constexpr std::string_view kNegativePrefix = "[-]";
constexpr std::array< std::string_view, 4 > kGoalLevelSuffixes = {"[g]", "[sg]", "[ssg]", "[sssg]"};
constexpr std::string_view kGoalSatisfiedSuffix = "[sat]";
constexpr std::string_view kGoalUnsatisfiedSuffix = "[unsat]";
constexpr std::string_view kGoalSatisfiedAddedSuffix = "[sat+]";
constexpr std::string_view kGoalSatisfiedRemovedSuffix = "[sat-]";
constexpr std::string_view kStateAnchorModifier = "[state]";

std::string_view polarity_prefix(const std::optional< bool >& polarity)
{
   if(not polarity.has_value()) {
      return "";
   }
   return *polarity ? kPositivePrefix : kNegativePrefix;
}

std::string_view goal_level_suffix(const std::optional< GoalLevel >& level)
{
   if(not level.has_value()) {
      return "";
   }
   return kGoalLevelSuffixes.at(level->value_of());
}

std::string_view goal_derivation_suffix(const std::optional< GoalDerivation >& derivation)
{
   if(not derivation.has_value()) {
      return "";
   }
   switch(*derivation) {
      case GoalDerivation::plain: return kGoalLevelSuffixes[0];
      case GoalDerivation::satisfied: return kGoalSatisfiedSuffix;
      case GoalDerivation::unsatisfied: return kGoalUnsatisfiedSuffix;
      case GoalDerivation::added_satisfied: return kGoalSatisfiedAddedSuffix;
      case GoalDerivation::added_unsatisfied: return kGoalSatisfiedRemovedSuffix;
   }
   return "";
}

void mix_hash(uint64_t& value, uint64_t part)
{
   value ^= part + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
}

uint64_t hash_bytes(std::string_view text)
{
   return std::hash< std::string_view >{}(text);
}

}  // namespace

std::string_view relation_usage_source_label(RelationUsage usage)
{
   switch(usage) {
      case RelationUsage::state: return "state";
      case RelationUsage::goal: return "goal";
      case RelationUsage::goal_derivation: return "goal_derivation";
      case RelationUsage::goal_satisfaction: return "goal_satisfaction";
      case RelationUsage::action: return "action";
      case RelationUsage::history: return "history";
      case RelationUsage::parent: return "parent";
      case RelationUsage::sibling: return "sibling";
      case RelationUsage::cousin: return "cousin";
   }
   return "";
}

std::string format_relation_name(const RelationKey& key)
{
   std::string out;
   out += polarity_prefix(key.polarity);
   out += key.base_name;
   for(const auto& modifier : key.modifiers) {
      out += modifier;
   }
   out += goal_level_suffix(key.goal_level);
   out += goal_derivation_suffix(key.derivation);
   if(key.state_anchored) {
      out += kStateAnchorModifier;
   }
   return out;
}

RelationKey opaque_relation_key(std::string formatted_name)
{
   RelationKey key;
   key.family = RelationFamily::auxiliary;
   key.base_name = std::move(formatted_name);
   return key;
}

RelationKey predicate_relation_key(
   std::string_view base_name,
   std::optional< bool > polarity,
   std::optional< GoalLevel > goal_level,
   std::optional< GoalDerivation > derivation,
   std::string_view modifier,
   bool state_anchored
)
{
   RelationKey key;
   key.family = RelationFamily::predicate;
   key.base_name = std::string(base_name);
   key.polarity = polarity;
   key.goal_level = goal_level;
   key.derivation = derivation;
   if(not modifier.empty()) {
      key.modifiers.emplace_back(modifier);
   }
   key.state_anchored = state_anchored;
   return key;
}

RelationKey action_relation_key(std::string_view schema_name)
{
   RelationKey key;
   key.family = RelationFamily::action;
   key.base_name = std::string(schema_name);
   return key;
}

uint64_t RelationKeyHash::operator()(const RelationKey& key) const noexcept
{
   uint64_t value = static_cast< uint64_t >(key.family);
   mix_hash(value, hash_bytes(key.base_name));
   mix_hash(value, key.polarity.has_value() ? (*key.polarity ? 2U : 1U) : 0U);
   mix_hash(
      value,
      key.goal_level.has_value() ? static_cast< uint64_t >(key.goal_level->value_of()) + 1 : 0U
   );
   mix_hash(value, key.derivation.has_value() ? static_cast< uint64_t >(*key.derivation) + 1 : 0U);
   for(const auto& modifier : key.modifiers) {
      mix_hash(value, hash_bytes(modifier));
   }
   mix_hash(value, key.state_anchored ? 1U : 0U);
   return value;
}

}  // namespace mifrost
