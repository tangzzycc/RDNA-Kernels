import pytest

from common.python import HipRuntime


@pytest.fixture(scope="session")
def hip():
    return HipRuntime()
