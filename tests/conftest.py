from pathlib import Path

import pytest

from vreg import content


@pytest.fixture(scope="session")
def tiny_clip(tmp_path_factory) -> Path:
    return content.make_tiny(tmp_path_factory.mktemp("clips") / "tiny.mkv")
