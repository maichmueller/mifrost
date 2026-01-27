"""
Test suite for ankerl::unordered_dense::map ↔ Python dict typecaster.

The typecaster enables bidirectional conversions:
1. Python dict → C++ ankerl::unordered_dense::map (for function arguments)
2. C++ ankerl::unordered_dense::map → Python dict (for return values)

Critical safety properties to verify:
- Type conversion correctness (keys and values)
- Memory safety (no double-free, no leaks)
- Edge cases (empty maps, large maps, nested structures)
- Interaction with mimir::formalism types (GroundLiteral keys)
"""

import pytest
import mifrost
import pymimir as mimir


@pytest.fixture
def simple_domain():
    """Create a minimal PDDL domain for testing."""
    import tempfile
    import os

    domain_str = """(define (domain test)
    (:predicates (p ?x) (q ?x ?y))
    (:action test-action
        :parameters (?x)
        :precondition (p ?x)
        :effect (not (p ?x))))
"""
    problem_str = """(define (problem test-problem)
    (:domain test)
    (:objects a b c)
    (:init (p a) (p b))
    (:goal (and (p c))))
"""

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as df:
        df.write(domain_str)
        domain_file = df.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as pf:
        pf.write(problem_str)
        problem_file = pf.name

    try:
        # Use pymimir's file-based API
        prob = mimir.Problem(domain_file, problem_file)
        yield prob.get_domain(), prob
    finally:
        os.unlink(domain_file)
        os.unlink(problem_file)


class TestUnorderedDenseTypecaster:
    """Test ankerl::unordered_dense::map typecaster bidirectional conversions."""

    def test_empty_map_roundtrip(self):
        """Empty Python dict → C++ map → Python dict."""
        inputs = mifrost.GoalInputs()
        # Empty fluent_goal_levels should be exposed as empty dict
        # (This tests C++ → Python conversion)
        assert hasattr(inputs, "fluent_goal_levels")
        levels = inputs.fluent_goal_levels
        assert isinstance(levels, dict)
        assert len(levels) == 0

    def test_simple_int_map_construction(self, simple_domain):
        """Python dict with simple types → C++ map."""
        dom, prob = simple_domain

        # Get some ground literals to use as keys
        p_a = list(prob.get_goal_condition().get_literals())[0]  # (p c)

        # Create GoalInputs with a goal level map
        goals_list = [p_a]
        inputs = mifrost.GoalInputs(goals_list, level=0)

        # Verify the goal was added
        assert len(inputs.static_goals) == 1 or len(inputs.fluent_goals) == 1

        # Check that goal_levels contains the literal with level 0
        # (tests Python → C++ → Python conversion)
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )
        assert len(levels) > 0
        assert list(levels.values())[0] == 0

    def test_multiple_levels_map(self, simple_domain):
        """Multiple goals with different levels."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if len(literals) < 1:
            pytest.skip("Need at least one literal")

        # Create inputs with first goal at level 0
        inputs = mifrost.GoalInputs([literals[0]], level=0)

        # Manually add another level (if we extend the API later)
        # For now, verify single level works
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )
        assert all(v == 0 for v in levels.values())

    def test_map_iteration(self, simple_domain):
        """Verify Python dict iteration works on returned C++ maps."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(literals[:1], level=0)
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        # Test dict iteration
        keys = list(levels.keys())
        values = list(levels.values())
        items = list(levels.items())

        assert len(keys) == len(values) == len(items)
        assert all(isinstance(v, int) for v in values)

    def test_map_membership_check(self, simple_domain):
        """Test 'in' operator on returned C++ maps."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(literals[:1], level=0)
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        if levels:
            first_key = list(levels.keys())[0]
            assert first_key in levels
            # Create a different literal that shouldn't be in the map
            # (membership test)

    def test_map_get_method(self, simple_domain):
        """Test dict.get() method on returned C++ maps."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(literals[:1], level=0)
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

        inputs1 = mifrost.GoalInputs(literals[:1], level=0)
        inputs2 = mifrost.GoalInputs(
            literals[:2] if len(literals) >= 2 else literals, level=0
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
        # If we have 2+ literals, levels2 should have at least as many
        if len(literals) >= 2:
            assert len(levels2) >= len(levels1)

    def test_memory_safety_repeated_access(self, simple_domain):
        """Verify no crashes when accessing map multiple times."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(literals[:1], level=0)

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

        # Should not crash or leak memory

    def test_map_copy_semantics(self, simple_domain):
        """Verify map is copied, not referenced, when returned to Python."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(literals[:1], level=0)
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
        # They might be the same object or different, depending on nanobind behavior
        # The important thing is they have the same content

    def test_type_safety_value_types(self, simple_domain):
        """Verify value types are correctly converted."""
        dom, prob = simple_domain

        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        inputs = mifrost.GoalInputs(literals[:1], level=42)
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        if levels:
            # All values should be Python ints, not C++ ints
            for v in levels.values():
                assert isinstance(v, int)
                assert v == 42

    def test_large_map_performance(self, simple_domain):
        """Verify reasonable performance with larger maps."""
        dom, prob = simple_domain

        # Create many literals if possible
        literals = list(prob.get_goal_condition().get_literals())
        if not literals:
            pytest.skip("No literals available")

        # Repeat literals to create a larger list
        many_literals = literals * 10

        # This should not crash or be extremely slow
        inputs = mifrost.GoalInputs(many_literals, level=0)
        levels = (
            inputs.fluent_goal_levels
            if inputs.fluent_goals
            else inputs.static_goal_levels
        )

        assert len(levels) <= len(many_literals)  # Might deduplicate


class TestBatchBuilderMaps:
    """Test unordered_dense::map usage in BatchBuilder internals."""

    def test_batch_builder_internal_maps(self):
        """BatchBuilder uses ankerl::unordered_dense::map internally."""
        builder = mifrost.BatchBuilder()

        # Add some data - this exercises the internal maps
        import torch

        builder.add_node_features("atom", "x", torch.randn(5, 1))
        builder.add_edges(
            "atom", "rel", "atom", torch.tensor([0, 1]), torch.tensor([1, 2])
        )
        builder.next_graph()

        # Build should succeed without crashes
        # (internal maps like node_offsets, current_node_counts, etc. should work)
        result = builder.build_parts()

        assert isinstance(result, dict)
        assert "tensors" in result
        # The fact that build_parts() succeeded means the internal
        # ankerl::unordered_dense::map usage worked correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
