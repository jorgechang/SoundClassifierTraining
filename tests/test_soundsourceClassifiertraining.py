"""Package related tests."""
from soundsourceClassifiertraining import __version__


def test_version():
    """Checks correct package version."""
    assert __version__ == "0.1.0"
