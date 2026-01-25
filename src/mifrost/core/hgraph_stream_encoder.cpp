#include "hgraph_stream_encoder.hpp"

#include <algorithm>
#include <fmt/format.h>
#include <map>
#include <numeric>

namespace mifrost {

HGraphStreamEncoder::HGraphStreamEncoder(const mimir::formalism::Domain &domain)
    : HGraphStreamEncoder(domain, Config{}) {}

HGraphStreamEncoder::HGraphStreamEncoder(const mimir::formalism::Domain &domain,
                                         Config config)
    : domain_(domain), config_(std::move(config)) {}

void HGraphStreamEncoder::encode_step(
    const mimir::search::State &state,
    std::span<const mimir::formalism::GroundLiteral<
        mimir::formalism::FluentTag>>
        goals,
    std::span<const mimir::formalism::GroundAction> actions,
    BatchBuilder &builder) {
  encode_objects(state, builder);
  encode_facts(state, builder);
  if (!goals.empty())
    encode_goals(goals, builder);
  // Actions skipped for now (HGraph basic doesn't include action history
  // typically)
}

void HGraphStreamEncoder::encode_state(const mimir::search::State &state,
                                       BatchBuilder &builder) {
  encode_objects(state, builder);
  encode_facts(state, builder);
}

void HGraphStreamEncoder::encode_objects(const mimir::search::State &state,
                                         BatchBuilder &builder) {
  const auto &problem = state.get_problem();
  const auto &objects = problem.get_problem_and_domain_objects();

  if (objects.empty()) {
    return;
  }

  size_t max_index = 0;
  for (const auto *obj : objects) {
    max_index = std::max(max_index, static_cast<size_t>(obj->get_index()));
  }

  std::vector<float> obj_features(max_index + 1, 1.0f);
  builder.add_node_features(config_.symbol_type_id, "x",
                            std::span<const float>(obj_features), 1);
}

void HGraphStreamEncoder::encode_facts(const mimir::search::State &state,
                                       BatchBuilder &builder) {
  // Buffers for edge indices per predicate
  // Key: pred_name
  struct PredBuffer {
    std::vector<int64_t> src;
    std::vector<int64_t> dst; // one vector per argument position?
    // HGraph edges: (pred_node, arg_pos_name, object_node)
    // We need vectors for each argument position.
    // Or simpler: Edge type (pred, "arg0", symbol) -> src, dst
    std::vector<std::vector<int64_t>> args; // args[0] = list of arg0 indices
  };
  std::map<std::string, PredBuffer> buffers;

  // Feature buffer for predicate nodes (all 1.0f)
  // Key: pred_name
  std::map<std::string, std::vector<float>> pred_features;

  const auto &problem = state.get_problem();
  const auto &repos = problem.get_repositories();

  auto add_ground_atoms = [&]<typename P>() {
    const auto atoms =
        repos.get_ground_atoms_from_indices<P>(state.get_atoms<P>());
    for (const auto &atom : atoms) {
      const auto predicate = atom->get_predicate();
      const auto &pred_name = predicate->get_name();
      const auto &objects = atom->get_objects();

      // 1. Add Fact Node Feature (x=1.0)
      pred_features[pred_name].push_back(1.0f);

      // 2. Buffer Argument Edges
      auto &buffer = buffers[pred_name];
      if (buffer.args.size() < objects.size()) {
        buffer.args.resize(objects.size());
      }

      for (size_t i = 0; i < objects.size(); ++i) {
        // This assumes object indices map 1:1 to our symbol nodes 0..N.
        buffer.args[i].push_back(
            static_cast<int64_t>(objects[i]->get_index()));
      }
    }
  };

  add_ground_atoms.operator()<mimir::formalism::FluentTag>();
  add_ground_atoms.operator()<mimir::formalism::DerivedTag>();

  // Flush buffers to Builder
  for (const auto &[pred_name, x_vec] : pred_features) {
    // Add nodes
    builder.add_node_features(pred_name, "x", x_vec, 1);

    // Add edges
    const auto &buffer = buffers[pred_name];
    size_t num_atoms =
        x_vec.size(); // == current_node_counts for this pred in this graph

    // Reconstruct source indices [0, 1, 2, ... N-1]
    // Efficient way? We can generate a sequence.
    // Actually, we need to push edges for each argument pos.
    for (size_t i = 0; i < buffer.args.size(); ++i) {
      std::vector<int64_t> src_indices(num_atoms);
      std::iota(src_indices.begin(), src_indices.end(), 0);

      std::string arg_rel = fmt::format("arg{}", i);
      // Edge: (Atom, argI, Object)
      builder.add_edges(pred_name, arg_rel, config_.symbol_type_id, src_indices,
                        buffer.args[i]);
    }
  }
}

void HGraphStreamEncoder::encode_goals(
    std::span<const mimir::formalism::GroundLiteral<
        mimir::formalism::FluentTag>>
        goals,
    BatchBuilder &builder) {
  // Similar buffering logic for goals
}

} // namespace mifrost
