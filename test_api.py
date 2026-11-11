"""
Test API endpoints with real queries
Tests all intent types and verifies responses
"""
import requests
import json
import time
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

_raw_host = (os.getenv("API_HOST") or "127.0.0.1").strip()
_client_host = "127.0.0.1" if _raw_host in ("", "0.0.0.0", "::") else _raw_host
_api_port = (os.getenv("API_PORT") or "8000").strip() or "8000"
API_BASE_URL = f"http://{_client_host}:{_api_port}"
# Configurable timeout from .env (default: 60 seconds for menu queries)
API_TIMEOUT = int(os.getenv("API_TIMEOUT") or "60")


def test_query(query: str, conversation_id: Optional[str] = None) -> Optional[Dict]:
    """Send a query to the /chat/text endpoint"""
    url = f"{API_BASE_URL}/chat/text"
    payload = {
        "query": query,
        "conversation_id": conversation_id
    }
    
    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print("   Make sure API server is running: python run_api.py")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        try:
            error_detail = response.json()
            print(f"   Details: {error_detail}")
        except:
            print(f"   Response: {response.text[:200]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_result(query: str, result: Optional[Dict]):
    """Pretty print test result"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    if not result:
        print("❌ No response received")
        return
    
    intent_info = result.get('intent') or {}
    intent_type = intent_info.get('intent_type', 'N/A') if isinstance(intent_info, dict) else 'N/A'
    
    print(f"Intent: {intent_type}")
    print(f"Retrieved Count: {result.get('retrieved_count', 0)}")
    print(f"Items: {len(result.get('items', []))}")
    
    response_text = result.get('response', '')
    print(f"\nResponse ({len(response_text)} chars):")
    print("-" * 60)
    print(response_text if response_text else 'No response')
    print("-" * 60)
    
    # Check for issues
    response_lower = response_text.lower() if response_text else ''
    issues = []
    
    if 'chatgpt' in response_lower or 'ai model' in response_lower:
        issues.append("⚠️  Contains AI disclaimer")
    if 'context' in response_lower or 'retrieved items' in response_lower:
        issues.append("⚠️  Contains technical jargon")
    if len(response_text) < 20:
        issues.append("⚠️  Response too short")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ Response looks good!")


def test_identity_queries():
    """Test identity-related queries"""
    print("\n" + "="*60)
    print("TESTING: Identity Queries")
    print("="*60)
    
    queries = [
        "who are you",
        "what do you do",
        "what can you help with",
    ]
    
    conversation_id = None
    for query in queries:
        result = test_query(query, conversation_id)
        if result:
            conversation_id = result.get('conversation_id')  # Use returned conversation_id if available
        print_result(query, result)
        time.sleep(1)  # Small delay between requests


def test_restaurant_info_queries():
    """Test restaurant info queries"""
    print("\n" + "="*60)
    print("TESTING: Restaurant Info Queries")
    print("="*60)
    
    queries = [
        "where are you located",
        "what are your hours",
        "tell me about Saigon Indian Restaurant",
        "what is your phone number",
    ]
    
    conversation_id = None
    for query in queries:
        result = test_query(query, conversation_id)
        if result:
            conversation_id = result.get('conversation_id')
        print_result(query, result)
        time.sleep(1)


def test_service_queries():
    """Test service queries"""
    print("\n" + "="*60)
    print("TESTING: Service Queries")
    print("="*60)
    
    queries = [
        "book a table",
        "make a reservation",
        "do you deliver",
    ]
    
    conversation_id = None
    for query in queries:
        result = test_query(query, conversation_id)
        if result:
            conversation_id = result.get('conversation_id')
        print_result(query, result)
        time.sleep(1)


def test_menu_queries():
    """Test menu queries"""
    print("\n" + "="*60)
    print("TESTING: Menu Queries")
    print("="*60)
    
    queries = [
        "biryani options",
        "vegetarian starters",
        "spicy non-vegetarian dishes",
        "dosa price",
    ]
    
    conversation_id = None
    for query in queries:
        result = test_query(query, conversation_id)
        if result:
            conversation_id = result.get('conversation_id')
        print_result(query, result)
        time.sleep(1)


def test_conversational_queries():
    """Test conversational queries"""
    print("\n" + "="*60)
    print("TESTING: Conversational Queries")
    print("="*60)
    
    queries = [
        "hello",
        "thank you",
    ]
    
    conversation_id = None
    for query in queries:
        result = test_query(query, conversation_id)
        if result:
            conversation_id = result.get('conversation_id')
        print_result(query, result)
        time.sleep(1)


def test_conversation_persistence():
    """Test that conversation history persists"""
    print("\n" + "="*60)
    print("TESTING: Conversation Persistence")
    print("="*60)
    
    conversation_id = str(time.time())  # Unique ID
    
    # First message
    print("\n1. First message:")
    result1 = test_query("hello", conversation_id)
    print_result("hello", result1)
    time.sleep(1)
    
    # Second message (should have context)
    print("\n2. Second message (should remember context):")
    result2 = test_query("what do you do", conversation_id)
    print_result("what do you do", result2)
    time.sleep(1)
    
    # Third message
    print("\n3. Third message:")
    result3 = test_query("tell me about biryani", conversation_id)
    print_result("tell me about biryani", result3)


def test_health_endpoint():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TESTING: Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        print(f"Status: {health.get('status')}")
        print(f"Message: {health.get('message')}")
        print(f"Collection Count: {health.get('collection_count')}")
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def main():
    """Run all API tests"""
    print("\n" + "="*60)
    print("API TEST SUITE")
    print("="*60)
    print(f"\nTesting API at: {API_BASE_URL}")
    print("Make sure the API server is running: python run_api.py")
    
    # Check if API is running
    if not test_health_endpoint():
        print("\n❌ API server is not running or not accessible!")
        print("Please start the server with: python run_api.py")
        return
    
    # Run tests
    test_identity_queries()
    test_restaurant_info_queries()
    test_service_queries()
    test_menu_queries()
    test_conversational_queries()
    test_conversation_persistence()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
