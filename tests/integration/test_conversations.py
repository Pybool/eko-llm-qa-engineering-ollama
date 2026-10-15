import requests

from tests.integration.base import create_conversation

def test_create_conversation(authenticated_client):
    
    data, payload = create_conversation(authenticated_client)

    assert isinstance(data, dict)
    assert "chat" in data
    assert data["chat"]["id"]
    assert data["chat"]["title"] == payload["title"]
    

def test_authenticated_user_can_retrieve_chats(authenticated_client):
    response = authenticated_client.request("GET", "/chats")

    assert response.status_code == 200

    data = response.json()

    assert "chats" in data
    assert isinstance(data["chats"], list), "'chats' must be an array"
    
    if(len(data["chats"]) > 0):
        assert all(isinstance(chat, dict) for chat in data["chats"])