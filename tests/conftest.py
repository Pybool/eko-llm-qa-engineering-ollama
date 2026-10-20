import os
import pytest

from tests.helpers.auth_helper import AuthHelper
from llmqa.providers.model_provider import ModelProvider
from tests.helpers.token_counter_helper import TokenCounterHelper
from tests.helpers.utilities import set_budget_policy, set_context_window


@pytest.fixture
def authenticated_client():
    email = os.getenv("CONTEXTLAB_EMAIL", "admin@example.com")
    password = "@10111011qwe"

    auth = AuthHelper()

    try:
        auth.login(email, password)
        yield auth
    finally:
        auth.close()


@pytest.fixture
def token_counter():
    counter = TokenCounterHelper()

    try:
        yield counter
    finally:
        counter.close()


@pytest.fixture(scope="session")
def judge_model():
    print("Model ===> ", ModelProvider.from_env().get_model())
    return ModelProvider.from_env().get_model()


@pytest.fixture(scope="function")
def restore_default_config(authenticated_client):
    default_config = {
        "contextWindowTokens": 4096,
        "policyLimitTokens": 2048,
        "overLimitAction": "compact",
    }

    yield

    set_context_window(
        authenticated_client,
        policy_limit_tokens=default_config["policyLimitTokens"],
        context_window_tokens=default_config["contextWindowTokens"],
    )

    set_budget_policy(
        authenticated_client,
        default_config["overLimitAction"],
    )
    
    print("Context window & budget policy configurations reverted to default.")