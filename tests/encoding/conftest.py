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
        if hasattr(space, "goal_states_iter"):
            state = next(iter(space.goal_states_iter()))
        elif hasattr(space, "sample_state_n_steps_from_goal"):
            state = space.sample_state_n_steps_from_goal(0)
        else:
            raise AttributeError("StateSpaceSampler does not expose goal state access")
    else:
        raise ValueError(
            "Unknown state wanted. Choose 'initial' or 'goal' state. Given: "
            + which_state
        )
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

    if which_state_param == "initial":
        state = problem.get_initial_state()
    elif which_state_param == "goal":
        if hasattr(space, "goal_states_iter"):
            state = next(iter(space.goal_states_iter()))
        elif hasattr(space, "sample_state_n_steps_from_goal"):
            state = space.sample_state_n_steps_from_goal(0)
        else:
            raise AttributeError("StateSpaceSampler does not expose goal state access")
    else:
        raise ValueError(
            "Unknown state wanted. Choose 'initial' or 'goal' state. Given: "
            + which_state_param
        )

    encoder = ColorEncoder(domain, **kwargs)
    return encoder.encode_pyg(state), encoder, state
