#!/usr/bin/env python3
"""
Simple interactive API test
Run this after starting the API server
"""
import requests
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

_raw_host = (os.getenv("API_HOST") or "127.0.0.1").strip()
_client_host = "127.0.0.1" if _raw_host in ("", "0.0.0.0", "::") else _raw_host
_api_port = (os.getenv("API_PORT") or "8000").strip() or "8000"
API_BASE_URL = f"http://{_client_host}:{_api_port}"
API_URL = f"{API_BASE_URL}/chat/text"
# Configurable timeout from .env (default: 60 seconds for menu queries)
API_TIMEOUT = int(os.getenv("API_TIMEOUT") or "60")

def test_query(query, conversation_id=None):
    """Test a single query"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    payload = {"query": query}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    try:
        response = requests.post(API_URL, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        
        intent_info = result.get('intent') or {}
        intent_type = intent_info.get('intent_type', 'N/A') if isinstance(intent_info, dict) else 'N/A'
        
        print(f"Intent: {intent_type}")
        print(f"Retrieved: {result.get('retrieved_count', 0)} items")
        
        response_text = result.get('response', 'No response')
        print(f"\nResponse:\n{response_text}")
        
        # Check for issues
        resp_text = response_text.lower() if isinstance(response_text, str) else ''
        if 'chatgpt' in resp_text or 'ai model' in resp_text:
            print("\n⚠️  WARNING: Contains AI disclaimer!")
        elif 'context' in resp_text or 'retrieved items' in resp_text:
            print("\n⚠️  WARNING: Contains technical jargon!")
        else:
            print("\n✅ Response looks good!")
        
        return result.get('conversation_id')
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API. Is the server running?")
        print("   Start with: python run_api.py")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("SIMPLE API TEST")
    print("="*60)
    print("\nMake sure the API server is running: python run_api.py")
    print("Press Ctrl+C to exit\n")
    
    # Check if API is running
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=10)
        print("✅ API server is running\n")
    except:
        print("❌ API server is not running!")
        print("   Please start it with: python run_api.py")
        sys.exit(1)
    
    # Test queries
    conversation_id = None
    
    tests = [
        ("who are you", "Identity query - should return welcome message"),
        ("what do you do", "Identity query - should return capabilities"),
        ("where are you located", "Restaurant info - should return location"),
        ("what are your hours", "Restaurant info - should return hours"),
        ("book a table", "Service query - should return reservation info"),
        ("biryani options", "Menu query - should return menu items"),
        ("vegetarian starters", "Menu query - should return menu items"),
    ]
    
    for query, description in tests:
        print(f"\n📋 {description}")
        conv_id = test_query(query, conversation_id)
        if conv_id:
            conversation_id = conv_id
        input("\nPress Enter to continue...")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
