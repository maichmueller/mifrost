#pragma once

#include "batch_builder.hpp"
#include "goal_inputs.hpp"
#include "relation_dict.hpp"
#include "stream_encoder_base.hpp"

#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mifrost
{
/**
 * @brief Engine that streams Mimir states into a BatchBuilder using HGraph semantics.
 *
 * Logic mirrors plangolin.encoding.hetero_encoder.HGraphEncoder.
 */
class HGraphEncoderEngine : public StreamEncoderBase<HGraphEncoderEngine>
{
public:
    struct Config
    {
        std::string symbol_type_id = "_symbol_";
        bool ignore_actions = true;
        bool add_nullary_predicates = false;
        bool include_lgan_edges = false;
        bool include_static = true;
        int max_goal_level = 0;
        bool support_literals = false;
        std::set<GoalSatisfaction> goal_satisfaction_derivations = {
            GoalSatisfaction::True,
            GoalSatisfaction::None,
        };
        std::string nullary_object_name = RelationFormatter::kDefaultNullarySymbolName;
        std::string lgan_nn_edge_pos = "lgan_nn";
    };

    HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
    HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
    HGraphEncoderEngine(mimir::formalism::Domain domain);
    HGraphEncoderEngine(mimir::formalism::Domain domain, Config config);

    template<typename GoalTag>
    void encode_step_impl(const mimir::search::State& state,
                          std::span<const mimir::formalism::GroundLiteral<GoalTag>> goals,
                          std::span<const mimir::formalism::GroundAction> actions,
                          BatchBuilder& builder);

    void encode_state_impl(const mimir::search::State& state, BatchBuilder& builder);

    void encode(const mimir::search::State& state, BatchBuilder& builder) { encode_state(state, builder); }

    void encode(const mimir::search::State& state, const GoalInputs& goals, std::span<const mimir::formalism::GroundAction> actions, BatchBuilder& builder)
    {
        encode_impl(state, goals, actions, builder);
    }

private:
    mimir::formalism::Domain domain_holder_;
    const mimir::formalism::DomainImpl& domain_;
    Config config_;
    RelationDict relation_dict_;
    std::vector<std::tuple<std::string, std::string, std::string>> all_edge_types_;

    // --- Encoding Helpers ---
    void encode_objects(const mimir::search::State& state,
                        BatchBuilder& builder,
                        std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& node_indices,
                        std::unordered_map<std::string, std::vector<std::string>>& node_names);
    std::unordered_set<std::string> encode_facts(const mimir::search::State& state,
                                                 BatchBuilder& builder,
                                                 std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& node_indices,
                                                 std::unordered_map<std::string, std::vector<std::string>>& node_names,
                                                 std::unordered_map<std::string, std::unordered_set<std::string>>& relation_to_symbols,
                                                 std::unordered_map<std::string, std::unordered_set<std::string>>& symbol_to_relations);
    template<typename GoalTag>
    void encode_literals(std::span<const mimir::formalism::GroundLiteral<GoalTag>> goals,
                         const ankerl::unordered_dense::map<mimir::formalism::GroundLiteral<GoalTag>, int>& goal_levels,
                         BatchBuilder& builder,
                         std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& node_indices,
                         std::unordered_map<std::string, std::vector<std::string>>& node_names,
                         std::unordered_map<std::string, std::unordered_set<std::string>>& relation_to_symbols,
                         std::unordered_map<std::string, std::unordered_set<std::string>>& symbol_to_relations);
    void encode_actions(std::span<const mimir::formalism::GroundAction> actions,
                        BatchBuilder& builder,
                        std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& node_indices,
                        std::unordered_map<std::string, std::vector<std::string>>& node_names,
                        std::unordered_map<std::string, std::unordered_set<std::string>>& relation_to_symbols,
                        std::unordered_map<std::string, std::unordered_set<std::string>>& symbol_to_relations);
    template<typename GoalTag>
    void encode_goal_satisfaction(std::span<const mimir::formalism::GroundLiteral<GoalTag>> goals,
                                  const ankerl::unordered_dense::map<mimir::formalism::GroundLiteral<GoalTag>, int>& goal_levels,
                                  const std::unordered_set<std::string>& fact_keys,
                                  BatchBuilder& builder,
                                  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& node_indices,
                                  std::unordered_map<std::string, std::vector<std::string>>& node_names,
                                  std::unordered_map<std::string, std::unordered_set<std::string>>& relation_to_symbols,
                                  std::unordered_map<std::string, std::unordered_set<std::string>>& symbol_to_relations);
    void add_lgan_nn_edges(BatchBuilder& builder,
                           const std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& node_indices,
                           const std::unordered_map<std::string, std::unordered_set<std::string>>& relation_to_symbols,
                           const std::unordered_map<std::string, std::unordered_set<std::string>>& symbol_to_relations);
    void ensure_empty_edge_types(BatchBuilder& builder) const;
    void ensure_node_feature_dims(BatchBuilder& builder) const;
    void
    encode_impl(const mimir::search::State& state, const GoalInputs& goals, std::span<const mimir::formalism::GroundAction> actions, BatchBuilder& builder);
    static std::string relation_key(const std::string& node_type, const std::string& node_key);
    int64_t get_or_add_node(const std::string& node_type,
                            const std::string& node_key,
                            BatchBuilder& builder,
                            std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& node_indices,
                            std::unordered_map<std::string, std::vector<std::string>>& node_names);
};
}  // namespace mifrost
