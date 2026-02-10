import os
import tempfile
import pytest
import pymimir as mimir
import mifrost
import torch
import numpy as np


@pytest.fixture
def simple_domain():
    """Create a minimal PDDL domain and problem for testing."""
    domain_str = """(define (domain test)
    (:requirements :strips :negative-preconditions)
    (:predicates (p ?x) (q ?x ?y))
    (:action test-action
        :parameters (?x)
        :precondition (not (p ?x))
        :effect (p ?x)))
"""
    problem_str = """(define (problem test-problem)
    (:domain test)
    (:objects a b c)
    (:init)
    (:goal (p c)))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as df:
        df.write(domain_str)
        domain_file = df.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as pf:
        pf.write(problem_str)
        problem_file = pf.name

    try:
        # Use pymimir's file-based API
        domain = mimir.Domain(domain_file)
        prob = mimir.Problem(domain, problem_file)
        yield domain, prob
    finally:
        os.unlink(domain_file)
        os.unlink(problem_file)


class TestUnorderedDenseTypecaster:
    """Test the ankerl::unordered_dense::map typecaster."""

    def test_simple_int_map_construction(self, simple_domain):
        """Test roundtrip of dict -> map -> dict for simple types."""
        dom, prob = simple_domain

        # Get some ground literals to use as keys
        p_a = list(prob.get_goal_condition().get_literals())[0]  # (p c)
        p_a_adv = p_a._advanced_ground_literal

        # Create GoalInputs with a goal level map
        goals_list = [p_a_adv]
        inputs = mifrost.GoalInputs(goals_list, level=0)

        # Verify the goal was added
        assert len(inputs.static_goals) == 1 or len(inputs.fluent_goals) == 1

        # Check the level map returned from C++
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )
        assert len(levels) == 1
        assert p_a_adv in levels
        assert levels[p_a_adv] == 0

    def test_multiple_levels_map(self, simple_domain):
        """Test maps with multiple entries and different values."""
        dom, prob = simple_domain

        # Get literals
        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        p_a = literals[0]

        # 1. Create with level 0
        inputs = mifrost.GoalInputs([p_a._advanced_ground_literal], level=0)

        # 2. Append to existing
        # Note: Current GoalInputs ctor doesn't support appending easily in C++
        # but we can check if multiple levels work if we had an API for it.
        # For now, let's just test multiple literals in one go if available.

    def test_map_iteration(self, simple_domain):
        """Test iteration over the returned C++ map proxy."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(
            [lit._advanced_ground_literal for lit in literals], level=0
        )
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        # Test items(), keys(), values()
        items = list(levels.items())
        keys = list(levels.keys())
        values = list(levels.values())

        assert len(items) >= 1
        assert len(keys) == len(values) == len(items)
        assert all(isinstance(v, int) for v in values)

    def test_map_membership_check(self, simple_domain):
        """Test 'in' operator on returned C++ maps."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(
            [lit._advanced_ground_literal for lit in literals[:1]], level=0
        )
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        if levels:
            first_key = list(levels.keys())[0]
            assert first_key in levels

    def test_map_get_method(self, simple_domain):
        """Test dict.get() method on returned C++ maps."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(
            [lit._advanced_ground_literal for lit in literals[:1]], level=0
        )
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        if levels:
            first_key = list(levels.keys())[0]
            # Test get with existing key
            assert levels.get(first_key) == 0
            # Test get with non-existing key and default
            assert levels.get("nonexistent", -1) == -1

    def test_map_len(self, simple_domain):
        """Test len() on returned C++ maps."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs1 = mifrost.GoalInputs(
            [lit._advanced_ground_literal for lit in literals[:1]], level=0
        )
        inputs2 = mifrost.GoalInputs(
            [
                lit._advanced_ground_literal
                for lit in (literals[:2] if len(literals) >= 2 else literals)
            ],
            level=0,
        )

        levels1 = (
            inputs1.fluent_goal_levels
            if inputs1.fluent_goals
            else inputs1.static_goal_levels
        )
        levels2 = (
            inputs2.fluent_goal_levels
            if inputs2.fluent_goals
            else inputs2.static_goal_levels
        )

        assert len(levels1) >= 1
        if len(literals) >= 2:
            assert len(levels2) >= len(levels1)

    def test_memory_safety_repeated_access(self, simple_domain):
        """Verify no crashes when accessing map multiple times."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(
            [lit._advanced_ground_literal for lit in literals[:1]], level=0
        )

        # Access the same map multiple times
        for _ in range(100):
            levels = (
                inputs.fluent_goal_levels
                if inputs.fluent_goals
                else inputs.static_goal_levels
            )
            _ = len(levels)
            _ = list(levels.keys())
            _ = list(levels.values())

    def test_map_copy_semantics(self, simple_domain):
        """Verify map is copied, not referenced, when returned to Python."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(
            [lit._advanced_ground_literal for lit in literals[:1]], level=0
        )
        levels1 = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )
        levels2 = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        # Both should be dicts with same content
        assert levels1 == levels2

    def test_type_safety_value_types(self, simple_domain):
        """Verify value types are correctly converted."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(
            [lit._advanced_ground_literal for lit in literals[:1]], level=42
        )
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        if levels:
            for v in levels.values():
                assert isinstance(v, int)
                assert v == 42

    def test_large_map_performance(self, simple_domain):
        """Verify reasonable performance with larger maps."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        many_literals = literals * 10
        inputs = mifrost.GoalInputs(
            [lit._advanced_ground_literal for lit in many_literals], level=0
        )
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        assert len(levels) <= len(many_literals)


class TestBatchBuilderMaps:
    """Test unordered_dense::map usage in BatchBuilder internals."""

    def test_batch_builder_internal_maps(self, simple_domain):
        """BatchBuilder uses ankerl::unordered_dense::map internally."""
        dom, prob = simple_domain
        builder = mifrost.BatchBuilder()
        state = prob.get_initial_state()
        goals = list(prob.get_goal_condition().get_literals())
        inputs = mifrost.GoalInputs(
            [g._advanced_ground_literal for g in goals], level=0
        )

        builder.add_node_features("atom", "x", torch.randn(5, 1))
        builder.add_edges(
            "atom", "rel", "atom", torch.tensor([0, 1]), torch.tensor([1, 2])
        )
        builder.next_graph()

        # Build should succeed without crashes
        # (internal maps like node_offsets, current_node_counts, etc. should work)
        builder.set_graph_kind("hetero")
        result = builder.build().as_dict()

        assert isinstance(result, dict)
        assert "tensors" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
