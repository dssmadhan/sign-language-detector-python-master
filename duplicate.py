def submit_to_ollama():
    api_endpoint = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    api_payload = {
        "model": "llama3.1",
        "prompt": " ".join(st.session_state.detected_characters),  
        "options": {"temperature": 0},
        "stream": False
    }

    response = requests.post(api_endpoint, headers=headers, json=api_payload)

    if response.status_code == 200:
        response_data = response.json()
        st.session_state.api_response = response_data.get('response', 'No response from Ollama')
        st.success("Response received from Ollama!")

        # Automatically convert Ollama's response to speech
        speak_text(st.session_state.api_response)

        # Display Ollama API response in the Streamlit interface
        st.text("Ollama's Response:")
        st.write(st.session_state.api_response)

    else:
        st.error(f"API request failed with status code {response.status_code}")
