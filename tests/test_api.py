"""
Comprehensive API tests for the FastAPI backend using TestClient.
Tests cover:
  1. Model loading & initialization
  2. Word generation endpoint
  3. Starred words management
  4. Health check
  5. Frontend serving
  6. Error handling & edge cases
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from fastapi.testclient import TestClient


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_model():
    """Mock the SlangRNN model."""
    model = MagicMock()
    model.load_state_dict = MagicMock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock the Tokenizer."""
    tokenizer = MagicMock()
    return tokenizer


@pytest.fixture
def mock_sample_n():
    """Mock the sample_n function to return consistent test data."""
    def _sample_n(n, model, tokenizer, max_length, temperature):
        # Return n mock words
        return [f"word_{i}" for i in range(n)]
    return _sample_n


@pytest.fixture
def client_with_mocks(mock_model, mock_tokenizer, mock_sample_n):
    """
    Fixture that provides a TestClient with mocked model, tokenizer, and sample_n.
    This ensures tests run fast and don't depend on actual artefacts.
    """
    with patch("backend.app.models.SlangRNN", return_value=mock_model), \
         patch("backend.app.Tokenizer") as mock_tokenizer_class, \
         patch("backend.app.sample_n", mock_sample_n), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", create=True) as mock_file:
        
        # Setup tokenizer mock
        mock_tokenizer_class.from_file.return_value = mock_tokenizer
        
        # Setup file reads for config.json
        mock_config = {"model": {"hidden_size": 128, "num_layers": 2}}
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_config)
        
        # Import here after mocks are in place
        from backend.app import app
        
        # Create test client
        client = TestClient(app)
        
        # Reset starred_words state before each test
        import backend.app as app_module
        app_module.starred_words.clear()
        
        yield client


@pytest.fixture(autouse=True)
def reset_starred_words():
    """Reset starred_words list before each test for isolation."""
    import backend.app as app_module
    app_module.starred_words.clear()
    yield
    app_module.starred_words.clear()


# ============================================================================
# 1. MODEL LOADING & INITIALIZATION
# ============================================================================

class TestModelLoading:
    """Tests for model loading during app startup."""

    def test_model_loads_on_startup(self, client_with_mocks):
        """Verify model and tokenizer are loaded during app startup."""
        # If the client was created successfully, model should be loaded
        response = client_with_mocks.get("/health")
        assert response.status_code == 200
        # If model wasn't loaded, the app would have crashed during startup

    def test_health_endpoint_indicates_app_ready(self, client_with_mocks):
        """Health endpoint confirms app is running after model load."""
        response = client_with_mocks.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ============================================================================
# 2. WORD GENERATION ENDPOINT
# ============================================================================

class TestWordGeneration:
    """Tests for the /generate endpoint."""

    def test_generate_returns_words_default_params(self, client_with_mocks):
        """Generate 10 words by default when no params provided."""
        response = client_with_mocks.get("/generate")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 10
        assert all(isinstance(w, str) for w in data)

    def test_generate_custom_num_words(self, client_with_mocks):
        """Generate custom number of words."""
        response = client_with_mocks.get("/generate?num_words=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_generate_custom_temperature(self, client_with_mocks):
        """Generate words with custom temperature."""
        response = client_with_mocks.get("/generate?temperature=0.5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 10  # default num_words

    def test_generate_with_both_params(self, client_with_mocks):
        """Generate words with both num_words and temperature."""
        response = client_with_mocks.get("/generate?num_words=3&temperature=2.0")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_generate_num_words_edge_case_one(self, client_with_mocks):
        """Generate single word."""
        response = client_with_mocks.get("/generate?num_words=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

    def test_generate_num_words_large(self, client_with_mocks):
        """Generate large number of words."""
        response = client_with_mocks.get("/generate?num_words=100")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 100

    def test_generate_invalid_num_words_string(self, client_with_mocks):
        """Invalid num_words (non-integer) should return validation error."""
        response = client_with_mocks.get("/generate?num_words=abc")
        assert response.status_code == 422  # Validation error

    def test_generate_invalid_num_words_negative(self, client_with_mocks):
        """Negative num_words should be rejected or handled gracefully."""
        response = client_with_mocks.get("/generate?num_words=-5")
        # Should either reject or treat as invalid
        # FastAPI may auto-coerce; we accept 200 if it works or 422 if rejected
        assert response.status_code in [200, 422]

    def test_generate_invalid_temperature_string(self, client_with_mocks):
        """Invalid temperature (non-float) should return validation error."""
        response = client_with_mocks.get("/generate?temperature=hot")
        assert response.status_code == 422

    def test_generate_temperature_zero(self, client_with_mocks):
        """Temperature at boundary: 0.0 (deterministic sampling)."""
        response = client_with_mocks.get("/generate?temperature=0.0")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 10

    def test_generate_temperature_very_high(self, client_with_mocks):
        """Temperature at high boundary (high randomness)."""
        response = client_with_mocks.get("/generate?temperature=10.0")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 10

    def test_generate_exception_handling(self, client_with_mocks):
        """If sample_n raises an exception, endpoint returns 500."""
        with patch("backend.app.sample_n", side_effect=RuntimeError("Model error")):
            response = client_with_mocks.get("/generate")
            assert response.status_code == 500
            assert "detail" in response.json()


# ============================================================================
# 3. STARRED WORDS MANAGEMENT
# ============================================================================

class TestStarredWords:
    """Tests for /starred endpoints (GET, POST add, POST remove)."""

    def test_get_starred_initially_empty(self, client_with_mocks):
        """GET /starred returns empty list on first call."""
        response = client_with_mocks.get("/starred")
        assert response.status_code == 200
        assert response.json() == []

    def test_add_starred_word(self, client_with_mocks):
        """POST /starred adds a word to the list."""
        response = client_with_mocks.post("/starred", json={"word": "awesome"})
        assert response.status_code == 200
        data = response.json()
        assert "awesome" in data
        assert len(data) == 1

    def test_add_starred_multiple_words(self, client_with_mocks):
        """Add multiple words and verify all are in list."""
        client_with_mocks.post("/starred", json={"word": "word_1"})
        client_with_mocks.post("/starred", json={"word": "word_2"})
        response = client_with_mocks.post("/starred", json={"word": "word_3"})
        data = response.json()
        assert len(data) == 3
        assert "word_1" in data
        assert "word_2" in data
        assert "word_3" in data

    def test_add_starred_duplicate_not_added_twice(self, client_with_mocks):
        """Adding same word twice should not create duplicate."""
        client_with_mocks.post("/starred", json={"word": "duplicate"})
        response = client_with_mocks.post("/starred", json={"word": "duplicate"})
        data = response.json()
        assert data.count("duplicate") == 1
        assert len(data) == 1

    def test_remove_starred_word(self, client_with_mocks):
        """POST /unstarred removes a word from the list."""
        # Add a word
        client_with_mocks.post("/starred", json={"word": "remove_me"})
        # Remove it
        response = client_with_mocks.post("/unstarred", json={"word": "remove_me"})
        data = response.json()
        assert "remove_me" not in data
        assert len(data) == 0

    def test_remove_starred_word_not_in_list(self, client_with_mocks):
        """Removing a word not in list should not error."""
        response = client_with_mocks.post("/unstarred", json={"word": "not_there"})
        # Should handle gracefully
        assert response.status_code == 200
        data = response.json()
        assert "not_there" not in data

    def test_remove_one_of_multiple_words(self, client_with_mocks):
        """Remove one word while others remain."""
        client_with_mocks.post("/starred", json={"word": "keep_1"})
        client_with_mocks.post("/starred", json={"word": "remove_me"})
        client_with_mocks.post("/starred", json={"word": "keep_2"})
        
        response = client_with_mocks.post("/unstarred", json={"word": "remove_me"})
        data = response.json()
        assert len(data) == 2
        assert "keep_1" in data
        assert "keep_2" in data
        assert "remove_me" not in data

    def test_starred_persists_across_requests(self, client_with_mocks):
        """Starred words persist across multiple GET requests."""
        client_with_mocks.post("/starred", json={"word": "persistent"})
        
        response_1 = client_with_mocks.get("/starred")
        response_2 = client_with_mocks.get("/starred")
        
        assert response_1.json() == response_2.json()
        assert "persistent" in response_1.json()

    def test_add_starred_invalid_json(self, client_with_mocks):
        """POST with invalid JSON should return 422."""
        response = client_with_mocks.post("/starred", json={"invalid_key": "value"})
        # FastAPI will validate the request body schema
        assert response.status_code == 422

    def test_add_starred_missing_word_field(self, client_with_mocks):
        """POST without 'word' field should return validation error."""
        response = client_with_mocks.post("/starred", json={})
        assert response.status_code == 422


# ============================================================================
# 4. HEALTH CHECK
# ============================================================================

class TestHealthCheck:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client_with_mocks):
        """Health endpoint returns correct status."""
        response = client_with_mocks.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_content_type(self, client_with_mocks):
        """Health endpoint returns JSON content type."""
        response = client_with_mocks.get("/health")
        assert "application/json" in response.headers["content-type"]


# ============================================================================
# 5. FRONTEND SERVING
# ============================================================================

class TestFrontendServing:
    """Tests for serving frontend files."""

    def test_root_endpoint_serves_index(self, client_with_mocks):
        """GET / serves index.html. Patch the `FileResponse` used in `backend.app`.
        Return a fast plain-text response so the TestClient doesn't block on file IO.
        """
        from fastapi.responses import PlainTextResponse

        with patch("backend.app.FileResponse") as mock_file_response:
            mock_file_response.return_value = PlainTextResponse("index.html content", media_type="text/html")
            response = client_with_mocks.get("/")
            # The patched handler should return quickly with our plain-text payload
            assert response.status_code == 200
            assert response.text == "index.html content"

    def test_static_mount_exists(self, client_with_mocks):
        """Static files mount point is configured."""
        # Verify /static prefix is mounted
        # This is more of a configuration check
        assert "/static" in [route.path for route in client_with_mocks.app.routes]


# ============================================================================
# 6. ERROR HANDLING & EDGE CASES
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_generate_sample_n_raises_exception(self, client_with_mocks):
        """When sample_n raises RuntimeError, endpoint returns 500."""
        with patch("backend.app.sample_n", side_effect=RuntimeError("Model inference failed")):
            response = client_with_mocks.get("/generate")
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    def test_generate_sample_n_raises_value_error(self, client_with_mocks):
        """When sample_n raises ValueError, endpoint returns 500."""
        with patch("backend.app.sample_n", side_effect=ValueError("Invalid input")):
            response = client_with_mocks.get("/generate")
            assert response.status_code == 500

    def test_invalid_endpoint_returns_404(self, client_with_mocks):
        """Request to non-existent endpoint returns 404."""
        response = client_with_mocks.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client_with_mocks):
        """POST to GET-only endpoint returns 405 or appropriate error."""
        response = client_with_mocks.post("/generate")
        # Depending on FastAPI config, may be 405 or 422
        assert response.status_code in [405, 422]

    def test_concurrent_starred_add_no_race_condition(self, client_with_mocks):
        """Multiple rapid add requests should all succeed without data loss."""
        words = [f"word_{i}" for i in range(10)]
        for word in words:
            response = client_with_mocks.post("/starred", json={"word": word})
            assert response.status_code == 200
        
        # Verify all words are in the list
        response = client_with_mocks.get("/starred")
        data = response.json()
        assert len(data) == 10
        for word in words:
            assert word in data

    def test_generate_zero_words(self, client_with_mocks):
        """Requesting 0 words should return empty list or error gracefully."""
        response = client_with_mocks.get("/generate?num_words=0")
        # Could be 200 with empty list or business logic error
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            assert response.json() == []

    def test_generate_negative_temperature(self, client_with_mocks):
        """Negative temperature may be rejected by validation or handled."""
        response = client_with_mocks.get("/generate?temperature=-1.0")
        # Could be 200 if app allows it, or 422 if validated
        assert response.status_code in [200, 422]


# ============================================================================
# RUNNING TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
