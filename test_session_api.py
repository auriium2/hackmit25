#!/usr/bin/env python3
"""
Test script for the API session management system.
Tests all session endpoints and functionality.
"""

import requests
import json
import time
import threading
from datetime import datetime

# API Base URL
BASE_URL = "http://localhost:8000"

def test_session_api():
    """Test comprehensive session management functionality."""
    print("🧪 Testing API Session Management System")
    print("=" * 70)

    # Test 1: Create a new session
    print("\n📍 Test 1: Create new session")
    create_data = {"user_id": "test_user_123"}

    try:
        response = requests.post(f"{BASE_URL}/session/create", json=create_data)
        if response.status_code == 200:
            session_result = response.json()
            query_id = session_result['queryID']
            print(f"✅ Created session: {query_id}")
            print(f"   User: {session_result['user_id']}")
            print(f"   Created: {session_result['created_at']}")
        else:
            print(f"❌ Failed to create session: {response.status_code}")
            print(f"   Response: {response.text}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start the test server first:")
        print("   python test_server.py")
        return

    # Test 2: Get session data
    print(f"\n📍 Test 2: Get session data")
    try:
        response = requests.get(f"{BASE_URL}/session?queryID={query_id}")
        if response.status_code == 200:
            session_data = response.json()
            print(f"✅ Retrieved session data")
            print(f"   Status: {session_data['status']}")
            print(f"   Processing Status: {session_data['session']['processing_status']}")
            print(f"   User ID: {session_data['session']['user_id']}")
        else:
            print(f"❌ Failed to get session: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting session: {e}")

    # Test 3: Submit query with existing session
    print(f"\n📍 Test 3: Submit query with existing session")
    query_data = {
        "query": "machine learning transformers",
        "queryID": query_id,
        "user_id": "test_user_123"
    }

    try:
        response = requests.post(f"{BASE_URL}/query", json=query_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query submitted successfully")
            print(f"   QueryID: {result['queryID']}")
            print(f"   SystemID: {result['systemid']}")
            print(f"   Seed Papers: {result.get('seed_papers_count', 0)}")
        else:
            print(f"❌ Failed to submit query: {response.status_code}")
    except Exception as e:
        print(f"❌ Error submitting query: {e}")

    # Test 4: Update session data
    print(f"\n📍 Test 4: Update session data")
    update_data = {
        "queryID": query_id,
        "custom_field": "test_value",
        "analysis_results": {"test": "data"}
    }

    try:
        response = requests.post(f"{BASE_URL}/session/update", json=update_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Session updated successfully")
            print(f"   Status: {result['status']}")
        else:
            print(f"❌ Failed to update session: {response.status_code}")
    except Exception as e:
        print(f"❌ Error updating session: {e}")

    # Test 5: Create session for another user
    print(f"\n📍 Test 5: Create session for different user")
    create_data2 = {"user_id": "test_user_456"}

    try:
        response = requests.post(f"{BASE_URL}/session/create", json=create_data2)
        if response.status_code == 200:
            session_result2 = response.json()
            query_id2 = session_result2['queryID']
            print(f"✅ Created second session: {query_id2}")
        else:
            print(f"❌ Failed to create second session: {response.status_code}")
    except Exception as e:
        print(f"❌ Error creating second session: {e}")
        query_id2 = None

    # Test 6: Get user sessions
    print(f"\n📍 Test 6: Get all sessions for user")
    try:
        response = requests.get(f"{BASE_URL}/sessions?user_id=test_user_123")
        if response.status_code == 200:
            user_sessions = response.json()
            print(f"✅ Retrieved user sessions")
            print(f"   User: {user_sessions['user_id']}")
            print(f"   Session Count: {user_sessions['session_count']}")
            for i, session in enumerate(user_sessions['sessions'], 1):
                print(f"   Session {i}: {session['queryID']} - {session['processing_status']}")
        else:
            print(f"❌ Failed to get user sessions: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting user sessions: {e}")

    # Test 7: Get session statistics
    print(f"\n📍 Test 7: Get session statistics")
    try:
        response = requests.get(f"{BASE_URL}/session/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Retrieved session statistics")
            print(f"   Active Sessions: {stats['total_active_sessions']}")
            print(f"   Unique Users: {stats['unique_users']}")
            print(f"   Avg Session Age: {stats['average_session_age_hours']:.2f} hours")
        else:
            print(f"❌ Failed to get session stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting session stats: {e}")

    # Test 8: Test session without user_id (anonymous)
    print(f"\n📍 Test 8: Create anonymous session")
    try:
        response = requests.post(f"{BASE_URL}/session/create", json={})
        if response.status_code == 200:
            anon_session = response.json()
            print(f"✅ Created anonymous session: {anon_session['queryID']}")
            print(f"   User: {anon_session['user_id']}")
        else:
            print(f"❌ Failed to create anonymous session: {response.status_code}")
    except Exception as e:
        print(f"❌ Error creating anonymous session: {e}")

    # Test 9: Submit query without existing session (should create new one)
    print(f"\n📍 Test 9: Submit query without session (auto-create)")
    new_query_data = {
        "query": "deep learning computer vision",
        "user_id": "test_user_789"
    }

    try:
        response = requests.post(f"{BASE_URL}/query", json=new_query_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query with auto-session creation successful")
            print(f"   Auto-created QueryID: {result['queryID']}")
            print(f"   SystemID: {result['systemid']}")
        else:
            print(f"❌ Failed query with auto-session: {response.status_code}")
    except Exception as e:
        print(f"❌ Error with auto-session query: {e}")

    # Test 10: Wait and check processing status updates
    print(f"\n📍 Test 10: Monitor processing status updates")
    try:
        # Wait 5 seconds and check status
        time.sleep(5)
        response = requests.get(f"{BASE_URL}/session?queryID={query_id}")
        if response.status_code == 200:
            session_data = response.json()
            status = session_data['session']['processing_status']
            print(f"✅ Processing status after 5s: {status}")
        else:
            print(f"❌ Failed to check status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking status: {e}")

    # Test 11: Expire a session
    print(f"\n📍 Test 11: Expire session")
    if query_id2:
        try:
            expire_data = {"queryID": query_id2}
            response = requests.post(f"{BASE_URL}/session/expire", json=expire_data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Session expired: {result['queryID']}")
                print(f"   Status: {result['status']}")
            else:
                print(f"❌ Failed to expire session: {response.status_code}")
        except Exception as e:
            print(f"❌ Error expiring session: {e}")

    # Test 12: Try to get expired session
    print(f"\n📍 Test 12: Try to get expired session")
    if query_id2:
        try:
            response = requests.get(f"{BASE_URL}/session?queryID={query_id2}")
            if response.status_code == 404:
                print(f"✅ Expired session correctly not found")
            else:
                print(f"⚠️  Expired session still accessible: {response.status_code}")
        except Exception as e:
            print(f"❌ Error checking expired session: {e}")

    print("\n" + "=" * 70)
    print("🎉 Session API Testing Completed!")
    print("\n✅ Features Tested:")
    print("   • Session creation (with and without user_id)")
    print("   • Session retrieval and data access")
    print("   • Session updates with custom data")
    print("   • Query submission with session tracking")
    print("   • User session listing")
    print("   • Session statistics and monitoring")
    print("   • Processing status tracking")
    print("   • Session expiration and cleanup")
    print("   • Anonymous session support")
    print("   • Auto-session creation for queries")

def test_concurrent_sessions():
    """Test concurrent session operations."""
    print("\n🔄 Testing Concurrent Session Operations")
    print("-" * 50)

    def create_session_worker(user_id):
        """Worker function to create sessions concurrently."""
        try:
            response = requests.post(f"{BASE_URL}/session/create",
                                   json={"user_id": f"concurrent_user_{user_id}"})
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Concurrent session {user_id}: {result['queryID'][:8]}...")
            else:
                print(f"❌ Concurrent session {user_id} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Concurrent session {user_id} error: {e}")

    # Create 5 sessions concurrently
    threads = []
    for i in range(5):
        thread = threading.Thread(target=create_session_worker, args=[i])
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("✅ Concurrent session creation test completed")

if __name__ == "__main__":
    test_session_api()
    test_concurrent_sessions()