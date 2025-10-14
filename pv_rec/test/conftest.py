from unittest import mock

import pytest
import requests


@pytest.fixture
def requests_mock():
    mock_response_get = requests.Response()
    mock_response_get._content = (
        b'sample data'
    )

    mock_response_post = requests.Response()
    mock_response_post._content = (
        b'[1, 2, 3]'
    )
    mock_response_post.status_code = 200

    method_mocks = {
        "get": mock.MagicMock(return_value=mock_response_get),
        "post": mock.MagicMock(return_value=mock_response_post),
    }

    return mock.patch.multiple(requests, **method_mocks)

@pytest.fixture
def sentence_transformers_mock():
    method_mocks = {
        "__init__": mock.MagicMock(return_value=None),
        "encode": mock.MagicMock(return_value=[42]),
    }

    return mock.patch.multiple(
        "sentence_transformers.SentenceTransformer.SentenceTransformer",
        **method_mocks
    )