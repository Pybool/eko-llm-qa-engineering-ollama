
import os
import pytest

from tests.helpers.auth_helper import AuthHelper
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