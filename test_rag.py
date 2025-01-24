import pytest
from unittest.mock import Mock
from rag import qa_with_context, get_text_embedding

# Mock client for Mistral API
@pytest.fixture
def mock_client():
    mock = Mock()
    # Mock the embeddings method
    mock.embeddings.return_value = Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
    # Mock the chat.complete method
    mock.chat.complete.return_value = Mock(
        choices=[
            Mock(message=Mock(content="This is a mock response."))
        ]
    )
    return mock

def test_get_text_embedding(mock_client):
    # Mocked embedding test
    text = "Sample text"
    embedding = get_text_embedding(mock_client, text)
    assert embedding == [0.1, 0.2, 0.3]  # Expected mock embedding

def test_qa_with_context(mock_client):
    # Mocked QA with context test
    text = "This is a long document that we will split into chunks."
    question = "What is this document about?"
    response = qa_with_context(mock_client, text, question, chunk_size=10)
    assert "mock response" in response.lower()

def test_split_into_chunks():
    # Test the chunk splitting logic
    from rag import split_into_chunks
    text = "This is a test document for chunking."
    chunks = split_into_chunks(text, chunk_size=10)
    assert chunks == ["This is a ", "test docum", "ent for ch", "unking."]

def test_configure_tools(mock_client):
    # Test tools configuration
    from rag import configure_tools
    text = "Sample text"
    tools, names_to_functions = configure_tools(mock_client, text)
    assert len(tools) > 0
    assert "qa_with_context" in names_to_functions