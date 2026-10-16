
import os
import pytest

from tests.helpers.auth_helper import AuthHelper
from llmqa.providers.model_provider import ModelProvider
from tests.helpers.token_counter_helper import TokenCounterHelper

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