import pytest
from tests.conftest import problem_setup
from mifrost.encoders import ColorEncoder, HGraphEncoder


SMALL_DOMAIN_CASES = [
    ("blocks", "smedium"),
    ("gripper", "gripper_b-5"),
    ("delivery", "instance_2x2_p-2_0"),
]

MEDIUM_DOMAIN_CASES = [
    ("blocks", "medium"),
    ("spanner", "medium"),
    ("reward", "instance_3x3_0"),
]

HORIZON_DOMAIN_CASES = SMALL_DOMAIN_CASES + MEDIUM_DOMAIN_CASES


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


@pytest.fixture(
    scope="session",
    params=HORIZON_DOMAIN_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in HORIZON_DOMAIN_CASES],
)
def horizon_cases(request):
    domain, problem = request.param
    return problem_setup(domain, problem)


@pytest.fixture(scope="session")
def large_blocks():
    pytest.skip("No large blocks instance available in data/pddl/blocks")


def select_state(space, problem, which_state: str):
    if which_state == "initial":
        return problem.get_initial_state()
    if which_state == "goal":
        if hasattr(space, "goal_states_iter"):
            return next(iter(space.goal_states_iter()))
        if hasattr(space, "sample_state_n_steps_from_goal"):
            return space.sample_state_n_steps_from_goal(0)
        raise AttributeError("StateSpaceSampler does not expose goal state access")
    raise ValueError(
        "Unknown state wanted. Choose 'initial' or 'goal' state. Given: " + which_state
    )


def encoded_state(
    domain: str,
    problem: str,
    which_state: str,
    encoder_class,
    **kwargs,
):
    space, domain, problem = problem_setup(domain, problem)
    state = select_state(space, problem, which_state)
    encoder = encoder_class(
        domain,
        **kwargs,
    )
    return encoder.encode_pyg(state), encoder


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


@pytest.fixture
def color_encoded_state(request):
    if len(request.param) == 4:
        domain_param, prob_param, which_state_param, kwargs = request.param
    else:
        domain_param, prob_param, which_state_param = request.param
        kwargs = {}
    space, domain, problem = problem_setup(domain_param, prob_param)
    state = select_state(space, problem, which_state_param)

    encoder = ColorEncoder(domain, **kwargs)
    return encoder.encode_pyg(state), encoder, state
