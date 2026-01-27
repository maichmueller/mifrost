#pragma once

#include <mimir/formalism/ground_literal.hpp>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mifrost
{

// Goal inputs are encoder-agnostic: all encoders consume the same tagged goals.
struct GoalInputs
{
    using FluentLiteral = mimir::formalism::GroundLiteral<mimir::formalism::FluentTag>;
    using DerivedLiteral = mimir::formalism::GroundLiteral<mimir::formalism::DerivedTag>;
    using StaticLiteral = mimir::formalism::GroundLiteral<mimir::formalism::StaticTag>;
    // Order matters for nanobind casting: nanobind tries variants in order.
    using AnyGoalLiteral = std::variant<FluentLiteral, DerivedLiteral, StaticLiteral>;

    mimir::formalism::GroundLiteralList<mimir::formalism::StaticTag> static_goals;
    mimir::formalism::GroundLiteralList<mimir::formalism::FluentTag> fluent_goals;
    mimir::formalism::GroundLiteralList<mimir::formalism::DerivedTag> derived_goals;
    ankerl::unordered_dense::map<mimir::formalism::GroundLiteral<mimir::formalism::StaticTag>, int> static_goal_levels;
    ankerl::unordered_dense::map<mimir::formalism::GroundLiteral<mimir::formalism::FluentTag>, int> fluent_goal_levels;
    ankerl::unordered_dense::map<mimir::formalism::GroundLiteral<mimir::formalism::DerivedTag>, int> derived_goal_levels;

    GoalInputs() = default;

    // Convenience ctor for Python: iterable -> vector<variant<...>> -> GoalInputs.
    explicit GoalInputs(const std::vector<AnyGoalLiteral>& goals) { append(goals, 0); }

    GoalInputs(const std::vector<AnyGoalLiteral>& goals, int level) { append(goals, level); }

private:
    void append(const std::vector<AnyGoalLiteral>& goals, int level)
    {
        for (const auto& goal : goals)
        {
            std::visit(
                [&](const auto& literal)
                {
                    using LiteralT = std::decay_t<decltype(literal)>;
                    if constexpr (std::is_same_v<LiteralT, FluentLiteral>)
                    {
                        fluent_goals.emplace_back(literal);
                        fluent_goal_levels[literal] = level;
                    }
                    else if constexpr (std::is_same_v<LiteralT, DerivedLiteral>)
                    {
                        derived_goals.emplace_back(literal);
                        derived_goal_levels[literal] = level;
                    }
                    else
                    {
                        static_goals.emplace_back(literal);
                        static_goal_levels[literal] = level;
                    }
                },
                goal);
        }
    }
};

}  // namespace mifrost
