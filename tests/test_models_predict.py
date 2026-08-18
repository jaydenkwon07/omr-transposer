import pytest

pytest.importorskip("torch")

from omrt.models import Model


def test_model_protocol_is_runtime_checkable():
    class Dummy:
        def predict(self, image):
            return ["barline"]

    assert isinstance(Dummy(), Model)

    class NotAModel:
        pass

    assert not isinstance(NotAModel(), Model)
