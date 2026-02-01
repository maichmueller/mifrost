import pytest
from tests.conftest import problem_setup
from mifrost.encoders import HGraphEncoder


SMALL_DOMAIN_CASES = [
    ("blocks", "probBLOCKS-4-0"),
    ("gripper", "gripper_b-5"),
    ("delivery", "instance_2x2_p-2_0"),
]

MEDIUM_DOMAIN_CASES = [
    ("blocks", "probBLOCKS-8-1"),
    ("spanner", "medium"),
    ("reward", "instance_3x3_0"),
]


@pytest.fixture(
    scope="session",
    params=SMALL_DOMAIN_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in SMALL_DOMAIN_CASES],
)
def small_blocks(request):
    domain, problem = request.param
    return problem_setup(domain, problem)


@pytest.fixture(
    scope="session",
    params=MEDIUM_DOMAIN_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in MEDIUM_DOMAIN_CASES],
)
def medium_blocks(request):
    domain, problem = request.param
    return problem_setup(domain, problem)


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
