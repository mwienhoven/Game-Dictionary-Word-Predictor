"""
Integration tests for the complete Game-Dictionary-Word-Predictor system running in Docker.

Tests cover:
  1. Complete workflow (generate words, star, unstar, persistence)
  2. Error recovery (invalid inputs, edge cases)
  3. API endpoint integration
  4. State management across requests
  5. Docker containerization validation

This test suite assumes:
  - Docker containers are running (backend on port 8000, frontend on port 80)
  - Tests are run after containers are healthy
  - Containers can be accessed via localhost
"""

import logging
import time
from typing import List

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & FIXTURES
# ============================================================================

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:80"
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_BACKOFF = 0.5


def create_session_with_retries():
    """Create requests session with retry strategy for flaky Docker connections."""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@pytest.fixture(scope="session")
def backend_session():
    """Create session for backend API testing."""
    return create_session_with_retries()


@pytest.fixture(scope="function")
def client():
    """Provide a fresh client for each test."""
    return create_session_with_retries()


@pytest.fixture(scope="session", autouse=True)
def wait_for_backend():
    """Wait for backend to be healthy before running tests."""
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                logger.info("Backend is healthy")
                return
        except requests.RequestException as e:
            logger.warning(f"Backend health check failed (attempt {attempt + 1}/{max_attempts}): {e}")
        
        attempt += 1
        time.sleep(1)
    
    pytest.skip(f"Backend not healthy after {max_attempts} attempts. Docker containers may not be running.")


# ============================================================================
# 1. COMPLETE WORKFLOW TESTS
# ============================================================================

class TestCompleteWorkflow:
    """Tests for the complete system workflow."""
    
    def test_generate_words_default_params(self, client):
        """Test generating words with default parameters."""
        response = client.get(
            f"{BACKEND_URL}/generate",
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        
        assert isinstance(data, list), "Response should be a list"
        assert len(data) == 10, f"Expected 10 words (default), got {len(data)}"
        assert all(isinstance(w, str) for w in data), "All items should be strings"
        logger.info(f"Generated 10 words: {data[:3]}...")
    
    def test_generate_words_custom_count(self, client):
        """Test generating custom number of words."""
        for num_words in [1, 5, 20]:
            response = client.get(
                f"{BACKEND_URL}/generate",
                params={"num_names": num_words, "creativity": 1.0},
                timeout=REQUEST_TIMEOUT
            )
            
            assert response.status_code == 200, f"Status {response.status_code}"
            data = response.json()
            
            assert len(data) <= num_words, f"Expected ≤{num_words}, got {len(data)}"
            logger.info(f"Generated {len(data)} words (requested {num_words})")
    
    def test_generate_words_custom_creativity(self, client):
        """Test generating words with different creativity (temperature) levels."""
        for creativity in [0.1, 1.0, 2.0]:
            response = client.get(
                f"{BACKEND_URL}/generate",
                params={"num_names": 5, "creativity": creativity},
                timeout=REQUEST_TIMEOUT
            )
            
            assert response.status_code == 200, f"Creativity {creativity}: Status {response.status_code}"
            data = response.json()
            assert isinstance(data, list), f"Creativity {creativity}: Response should be list"
            logger.info(f"Generated {len(data)} words with creativity={creativity}")
    
    def test_star_word(self, client):
        """Test adding a word to starred list."""
        # Generate some words first
        response = client.get(f"{BACKEND_URL}/generate", timeout=REQUEST_TIMEOUT)
        assert response.status_code == 200
        words = response.json()
        
        test_word = words[0] if words else "test_word"
        
        # Star the word
        response = client.post(
            f"{BACKEND_URL}/starred",
            json={"word": test_word},
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        starred = response.json()
        
        assert isinstance(starred, list), "Starred should be a list"
        assert test_word in starred, f"'{test_word}' should be in starred list"
        logger.info(f"Starred word: {test_word}")
    
    def test_get_starred_words(self, client):
        """Test retrieving starred words list."""
        response = client.get(f"{BACKEND_URL}/starred", timeout=REQUEST_TIMEOUT)
        
        assert response.status_code == 200, f"Status {response.status_code}"
        data = response.json()
        
        assert isinstance(data, list), "Response should be a list"
        assert all(isinstance(w, str) for w in data), "All items should be strings"
        logger.info(f"Current starred words: {data}")
    
    def test_unstar_word(self, client):
        """Test removing a word from starred list."""
        # Star a word first
        star_response = client.post(
            f"{BACKEND_URL}/starred",
            json={"word": "test_unstar"},
            timeout=REQUEST_TIMEOUT
        )
        assert star_response.status_code == 200
        
        # Unstar it
        response = client.post(
            f"{BACKEND_URL}/unstarred",
            json={"word": "test_unstar"},
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        starred = response.json()
        
        assert "test_unstar" not in starred, "Word should be removed from starred"
        logger.info("Successfully unstarred word")
    
    def test_star_unstar_workflow(self, client):
        """Test complete star/unstar workflow."""
        words_to_star = ["word1", "word2", "word3"]
        
        # Star multiple words
        for word in words_to_star:
            response = client.post(
                f"{BACKEND_URL}/starred",
                json={"word": word},
                timeout=REQUEST_TIMEOUT
            )
            assert response.status_code == 200
        
        # Verify all are starred
        response = client.get(f"{BACKEND_URL}/starred", timeout=REQUEST_TIMEOUT)
        starred = response.json()
        for word in words_to_star:
            assert word in starred, f"'{word}' should be starred"
        
        logger.info(f"All {len(words_to_star)} words starred successfully")
        
        # Unstar all
        for word in words_to_star:
            response = client.post(
                f"{BACKEND_URL}/unstarred",
                json={"word": word},
                timeout=REQUEST_TIMEOUT
            )
            assert response.status_code == 200
        
        # Verify all are unstarred
        response = client.get(f"{BACKEND_URL}/starred", timeout=REQUEST_TIMEOUT)
        starred = response.json()
        for word in words_to_star:
            assert word not in starred, f"'{word}' should not be starred"
        
        logger.info(f"All {len(words_to_star)} words unstarred successfully")
    
    def test_starred_persistence_across_requests(self, client):
        """Test that starred words persist across multiple requests."""
        persistent_word = "persistent_test_word"
        
        # Star a word
        response = client.post(
            f"{BACKEND_URL}/starred",
            json={"word": persistent_word},
            timeout=REQUEST_TIMEOUT
        )
        assert response.status_code == 200
        
        # Check it persists across 5 requests
        for i in range(5):
            response = client.get(f"{BACKEND_URL}/starred", timeout=REQUEST_TIMEOUT)
            assert response.status_code == 200
            starred = response.json()
            assert persistent_word in starred, f"Word not persisted on request {i + 1}"
        
        logger.info(f"'{persistent_word}' persisted across 5 requests")
        
        # Clean up
        client.post(
            f"{BACKEND_URL}/unstarred",
            json={"word": persistent_word},
            timeout=REQUEST_TIMEOUT
        )


# ============================================================================
# 2. ERROR RECOVERY & INVALID INPUTS
# ============================================================================

class TestErrorRecovery:
    """Tests for error handling and recovery."""
    
    def test_generate_words_invalid_count_string(self, client):
        """Test handling of invalid num_names (string instead of int)."""
        response = client.get(
            f"{BACKEND_URL}/generate",
            params={"num_names": "not_a_number"},
            timeout=REQUEST_TIMEOUT
        )
        
        # Should return validation error
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
        logger.info("Invalid count properly rejected with 422")
    
    def test_generate_words_negative_count(self, client):
        """Test handling of negative num_names."""
        response = client.get(
            f"{BACKEND_URL}/generate",
            params={"num_names": -5},
            timeout=REQUEST_TIMEOUT
        )
        
        # May return 200 (clamped to 0) or 422 (validation error)
        assert response.status_code in [200, 422], f"Unexpected status {response.status_code}"
        logger.info(f"Negative count handled: status {response.status_code}")
    
    def test_generate_words_zero_count(self, client):
        """Test generating 0 words."""
        response = client.get(
            f"{BACKEND_URL}/generate",
            params={"num_names": 0},
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code == 200, f"Status {response.status_code}"
        data = response.json()
        
        assert data == [], "Should return empty list for 0 words"
        logger.info("Zero word count handled correctly")
    
    def test_generate_words_invalid_creativity_string(self, client):
        """Test handling of invalid creativity (non-numeric)."""
        response = client.get(
            f"{BACKEND_URL}/generate",
            params={"creativity": "high"},
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
        logger.info("Invalid creativity properly rejected with 422")
    
    def test_star_word_empty_string(self, client):
        """Test starring an empty string."""
        response = client.post(
            f"{BACKEND_URL}/starred",
            json={"word": ""},
            timeout=REQUEST_TIMEOUT
        )
        
        # Should accept or reject; either is acceptable
        assert response.status_code in [200, 400, 422], f"Unexpected status {response.status_code}"
        logger.info(f"Empty word star: status {response.status_code}")
    
    def test_star_word_missing_field(self, client):
        """Test starring without 'word' field."""
        response = client.post(
            f"{BACKEND_URL}/starred",
            json={},
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
        logger.info("Missing word field properly rejected with 422")
    
    def test_star_word_invalid_json(self, client):
        """Test starring with invalid JSON."""
        response = client.post(
            f"{BACKEND_URL}/starred",
            data="not json",
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code in [400, 422], f"Unexpected status {response.status_code}"
        logger.info(f"Invalid JSON: status {response.status_code}")
    
    def test_unstar_nonexistent_word(self, client):
        """Test unstarring a word that was never starred."""
        response = client.post(
            f"{BACKEND_URL}/unstarred",
            json={"word": "never_starred_xyz_123"},
            timeout=REQUEST_TIMEOUT
        )
        
        # Should handle gracefully (200 is acceptable)
        assert response.status_code == 200, f"Status {response.status_code}"
        logger.info("Unstarring nonexistent word handled gracefully")
    
    def test_invalid_endpoint(self, client):
        """Test requesting invalid endpoint."""
        response = client.get(
            f"{BACKEND_URL}/nonexistent",
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        logger.info("Invalid endpoint properly returns 404")
    
    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method on endpoint."""
        response = client.post(
            f"{BACKEND_URL}/generate",
            json={},
            timeout=REQUEST_TIMEOUT
        )
        
        assert response.status_code in [405, 422], f"Expected 405 or 422, got {response.status_code}"
        logger.info(f"POST to GET endpoint: status {response.status_code}")


# ============================================================================
# 3. HEALTH & STATUS CHECKS
# ============================================================================

class TestHealthAndStatus:
    """Tests for health checks and system status."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get(f"{BACKEND_URL}/health", timeout=REQUEST_TIMEOUT)
        
        assert response.status_code == 200, f"Status {response.status_code}"
        data = response.json()
        
        assert "status" in data, "Health response should have 'status' field"
        assert data["status"] == "ok", "Status should be 'ok'"
        logger.info("Health check passed")
    
    def test_health_endpoint_content_type(self, client):
        """Test health endpoint returns JSON."""
        response = client.get(f"{BACKEND_URL}/health", timeout=REQUEST_TIMEOUT)
        
        assert "application/json" in response.headers.get("content-type", ""), \
            "Response should be JSON"
        logger.info("Health response is valid JSON")
    
    def test_response_times_reasonable(self, client):
        """Test that API response times are reasonable."""
        response = client.get(
            f"{BACKEND_URL}/generate",
            params={"num_names": 5},
            timeout=REQUEST_TIMEOUT
        )
        
        # Should complete in reasonable time (10 seconds is the timeout)
        assert response.elapsed.total_seconds() < REQUEST_TIMEOUT, \
            f"Response took {response.elapsed.total_seconds()}s (timeout: {REQUEST_TIMEOUT}s)"
        logger.info(f"Response time: {response.elapsed.total_seconds():.2f}s")


# ============================================================================
# 4. CONCURRENT REQUEST HANDLING
# ============================================================================

class TestConcurrentRequests:
    """Tests for handling multiple concurrent requests."""
    
    def test_sequential_generate_requests(self, client):
        """Test multiple sequential generate requests."""
        for i in range(5):
            response = client.get(
                f"{BACKEND_URL}/generate",
                params={"num_names": 3},
                timeout=REQUEST_TIMEOUT
            )
            
            assert response.status_code == 200, f"Request {i + 1}: Status {response.status_code}"
            data = response.json()
            assert len(data) <= 3, f"Request {i + 1}: Got {len(data)} words"
        
        logger.info("All 5 sequential generate requests succeeded")
    
    def test_star_unstar_alternating(self, client):
        """Test alternating star/unstar operations."""
        for i in range(3):
            word = f"word_{i}"
            
            # Star
            response = client.post(
                f"{BACKEND_URL}/starred",
                json={"word": word},
                timeout=REQUEST_TIMEOUT
            )
            assert response.status_code == 200
            
            # Unstar
            response = client.post(
                f"{BACKEND_URL}/unstarred",
                json={"word": word},
                timeout=REQUEST_TIMEOUT
            )
            assert response.status_code == 200
        
        logger.info("Alternating star/unstar operations succeeded")


# ============================================================================
# 5. DOCKER-SPECIFIC CHECKS
# ============================================================================

class TestDockerIntegration:
    """Tests specific to Docker containerization."""
    
    def test_backend_responding(self, client):
        """Test backend container is responding."""
        response = client.get(f"{BACKEND_URL}/health", timeout=REQUEST_TIMEOUT)
        assert response.status_code == 200
        logger.info("Backend container is responding")
    
    def test_api_response_format_consistency(self, client):
        """Test API response format is consistent across calls."""
        responses = []
        
        for _ in range(3):
            response = client.get(
                f"{BACKEND_URL}/generate",
                params={"num_names": 2},
                timeout=REQUEST_TIMEOUT
            )
            responses.append(response.json())
        
        # All responses should be lists
        assert all(isinstance(r, list) for r in responses), "All responses should be lists"
        logger.info("API response format is consistent")
    
    def test_no_internal_server_errors(self, client):
        """Test that API doesn't return 500 errors on normal requests."""
        for _ in range(10):
            response = client.get(
                f"{BACKEND_URL}/generate",
                timeout=REQUEST_TIMEOUT
            )
            
            assert response.status_code != 500, \
                f"Got 500 error: {response.text}"
        
        logger.info("No internal server errors in 10 requests")


# ============================================================================
# RUNNING TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
