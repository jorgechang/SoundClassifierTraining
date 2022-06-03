"""Fixtures for testing."""
from os import listdir, path

import pytest


def get_source(folder):
    """Return the last element in folder."""
    testdata_path = path.join("testdata", folder)

    audio_names = listdir(testdata_path)

    sources = [path.abspath(f"{testdata_path}/{audio}") for audio in audio_names]

    return sources[-1]


@pytest.fixture(scope="session")
def source(request):
    """Return a source path to any file from testdata folder.

    Args:
        request (object): . pytest request
    """
    return get_source(request.param)
