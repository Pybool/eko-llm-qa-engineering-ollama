def get_ollama_invocations(authenticated_client):
    response = authenticated_client.request(
        "GET",
        "/qa/ollama-invocations",
    )

    assert response.status_code == 200
    return response.json()