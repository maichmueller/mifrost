#pragma once

#include <mimir/common/formatter.hpp>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/formatter.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_atom.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/formalism/object.hpp>
#include <mimir/formalism/predicate.hpp>
#include <optional>
#include <string>

namespace mifrost
{

enum class GoalSatisfaction
{
    None,
    True,
    False,
    Added,
    Removed,
};

struct RelationFormatter
{
    static constexpr auto kPositivePrefix = "[+]";
    static constexpr auto kNegativePrefix = "[-]";
    static constexpr auto goal_suffixes = std::array { "[g]", "[sg]", "[ssg]", "[sssg]" };
    static constexpr auto kGoalSatisfiedSuffix = "[sat]";
    static constexpr auto kGoalUnsatisfiedSuffix = "[unsat]";
    static constexpr auto kGoalSatisfiedAddedSuffix = "[sat+]";
    static constexpr auto kGoalSatisfiedRemovedSuffix = "[sat-]";
    static constexpr auto kDefaultNullarySymbolName = "![nullary_symbol]!";

    static std::string goal_level_suffix(std::optional<int> level)
    {
        if (!level.has_value())
            return "";
        return goal_suffixes.at(*level);
    }

    static std::string goal_satisfaction_suffix(std::optional<GoalSatisfaction> satisfaction)
    {
        if (!satisfaction.has_value())
            return "";
        switch (*satisfaction)
        {
            case GoalSatisfaction::None:
                return "";
            case GoalSatisfaction::True:
                return kGoalSatisfiedSuffix;
            case GoalSatisfaction::False:
                return kGoalUnsatisfiedSuffix;
            case GoalSatisfaction::Added:
                return kGoalSatisfiedAddedSuffix;
            case GoalSatisfaction::Removed:
                return kGoalSatisfiedRemovedSuffix;
        }
        return "";
    }

    static std::string polarity_prefix(std::optional<bool> polarity)
    {
        if (!polarity.has_value())
            return "";
        return *polarity ? kPositivePrefix : kNegativePrefix;
    }

    static std::string format_predicate(const std::string& name,
                                        std::optional<int> goal_level = std::nullopt,
                                        std::optional<GoalSatisfaction> satisfaction = std::nullopt,
                                        std::optional<bool> polarity = std::nullopt)
    {
        return polarity_prefix(polarity) + name + goal_level_suffix(goal_level) + goal_satisfaction_suffix(satisfaction);
    }

    template<typename P>
    static std::string format_predicate(const mimir::formalism::PredicateImpl<P>& predicate,
                                        std::optional<int> goal_level = std::nullopt,
                                        std::optional<GoalSatisfaction> satisfaction = std::nullopt,
                                        std::optional<bool> polarity = std::nullopt)
    {
        return format_predicate(predicate.get_name(), goal_level, satisfaction, polarity);
    }

    template<typename P>
    static std::string format_atom(mimir::formalism::GroundAtom<P> atom)
    {
        return mimir::to_string(atom);
    }

    template<typename P>
    static std::string format_literal(mimir::formalism::GroundLiteral<P> literal,
                                      std::optional<int> goal_level = std::nullopt,
                                      std::optional<GoalSatisfaction> satisfaction = std::nullopt)
    {
        std::string out = mimir::to_string(literal);
        out.append(goal_level_suffix(goal_level));
        out.append(goal_satisfaction_suffix(satisfaction));
        return out;
    }

    static std::string format_action_schema(const mimir::formalism::ActionImpl& action) { return action.get_name(); }

    static std::string format_action(mimir::formalism::GroundAction action)
    {
        std::string out = "(" + action->get_action()->get_name();
        for (const auto& obj : action->get_objects())
        {
            out.append(" ");
            out.append(obj->get_name());
        }
        out.append(")");
        return out;
    }

    static std::string format_object(const mimir::formalism::ObjectImpl& object) { return object.get_name(); }
};

}  // namespace mifrost
