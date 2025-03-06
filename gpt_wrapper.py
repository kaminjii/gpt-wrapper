import os
import requests
import json

class GPTWrapper:
    def __init__(self, api_key=None, model="gpt-4", base_url="https://api.openai.com/v1"):
        """
        Initialize the GPT wrapper with your API key and preferred model.
        
        Args:
            api_key (str): Your OpenAI API key. If None, will try to get from environment.
            model (str): The GPT model to use.
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
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
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


# Example usage
if __name__ == "__main__":
    # Initialize the wrapper
    wrapper = GPTWrapper()
    
    # Define a system message
    system_message = "You are a helpful assistant."
    
    # Get a completion
    response = wrapper.simple_completion(
        prompt="Explain what a GPT wrapper is in one paragraph.",
        system_message=system_message
    )
    
    print(response)
    