import pytest
from tests.conftest import problem_setup
from mifrost.encoders import HGraphEncoder


@pytest.fixture(scope="session")
def small_blocks():
    return problem_setup("blocks", "probBLOCKS-4-0")


@pytest.fixture(scope="session")
def medium_blocks():
    return problem_setup("blocks", "probBLOCKS-8-1")


@pytest.fixture(scope="session")
def large_blocks():
    pytest.skip("No large blocks instance available in data/pddl/blocks")


def encoded_state(
    domain: str,
    problem: str,
    which_state: str,
    encoder_class,
    **kwargs,
):
    space, domain, problem = problem_setup(domain, problem)

    if which_state == "initial":
        state = problem.get_initial_state()
    elif which_state == "goal":
        state = next(iter(space.goal_states_iter()))
    else:
        raise ValueError(
            "Unknown state wanted. Choose 'initial' or 'goal' state. Given: "
            + which_state
        )
    encoder = encoder_class(
        domain,
        **kwargs,
    )
    return encoder.encode(state), encoder


@pytest.fixture
def hetero_encoded_state(request):
    if len(request.param) == 4:
        domain_param, prob_param, which_state_param, kwargs = request.param
    else:
        domain_param, prob_param, which_state_param = request.param
        kwargs = {}
    return encoded_state(
        domain_param,
        prob_param,
        which_state_param,
        HGraphEncoder,
        **kwargs,
    )
