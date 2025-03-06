import os
import requests
import json

class GPTWrapper:
    def __init__(self, api_key=None, model="gpt-3.5-turbo", base_url="https://api.openai.com/v1"):
        """
        Initialize the GPT wrapper with your API key and preferred model.
        
        Args:
            api_key (str): Your OpenAI API key. If None, will try to get from environment.
            model (str): The GPT model to use (default: gpt-3.5-turbo).
            base_url (str): The base URL for the API.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def chat_completion(self, messages, temperature=0.7, max_tokens=1000, **kwargs):
        """
        Send a chat completion request to the GPT API.
        
        Args:
            messages (list): A list of message dictionaries with 'role' and 'content'.
            temperature (float): Controls randomness. Lower is more deterministic.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional parameters to send to the API.
        
        Returns:
            dict: The API response as a dictionary.
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        # Print request information for debugging
        print(f"Making request to: {endpoint}")
        print(f"Using model: {self.model}")
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            print(f"Sending payload: {json.dumps(payload, indent=2)}")
            response = requests.post(endpoint, headers=self.headers, json=payload)
            
            # Print the response status and headers for debugging
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                print(f"Error response body: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error response: {e.response.text}")
            return {"error": str(e)}
    
    def simple_completion(self, prompt, system_message=None):
        """
        A simplified interface to get a completion from a single prompt.
        
        Args:
            prompt (str): The user prompt.
            system_message (str, optional): An optional system message.
        
        Returns:
            str: The generated text response.
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(messages)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            return f"Error parsing response: {e}"


    def list_available_models(self):
        """
        Get a list of available models from the OpenAI API.
        
        Returns:
            list: A list of available model IDs.
        """
        endpoint = f"{self.base_url}/models"
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch models: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize the wrapper
    wrapper = GPTWrapper()
    
    # List available models
    print("Available models:")
    models = wrapper.list_available_models()
    for model in models:
        print(f"- {model}")
    
    # Define a system message
    system_message = "You are a helpful assistant."
    
    # Get a completion
    response = wrapper.simple_completion(
        prompt="Explain what a GPT wrapper is in one paragraph.",
        system_message=system_message
    )
    
    print("\nResponse:")
    print(response)
    