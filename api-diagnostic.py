import os
import requests
import json

def check_api_connection():
    """
    Perform basic diagnostics on the OpenAI API connection.
    """
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: No API key found in environment variables.")
        print("   Set your API key with: export OPENAI_API_KEY=your-key-here")
        return
    
    # Mask the API key for display
    masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 10 else "****"
    print(f"✓ API key found: {masked_key}")
    
    # Set up headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Check basic connection
    try:
        print("\nTesting connection to OpenAI API...")
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        
        if response.status_code == 200:
            print(f"✓ Successfully connected to API (Status: {response.status_code})")
            models = response.json()
            print(f"✓ Found {len(models.get('data', []))} models available to your account")
            
            # List a few models
            model_ids = [model['id'] for model in models.get('data', [])]
            common_models = ['gpt-3.5-turbo', 'gpt-4', 'text-embedding-ada-002']
            
            print("\nChecking for common models:")
            for model in common_models:
                if any(model in m for m in model_ids):
                    print(f"✓ {model} (or similar) is available")
                else:
                    print(f"❌ {model} doesn't appear to be available to your account")
            
            # Try a simple completion
            print("\nTesting chat completion endpoint...")
            chat_endpoint = "https://api.openai.com/v1/chat/completions"
            
            # Find an available chat model
            chat_model = next((m for m in model_ids if 'gpt' in m.lower()), 'gpt-3.5-turbo')
            
            payload = {
                "model": chat_model,
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10
            }
            
            chat_response = requests.post(chat_endpoint, headers=headers, json=payload)
            
            if chat_response.status_code == 200:
                print(f"✓ Chat completion successful with model: {chat_model}")
                response_content = chat_response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"   Response: \"{response_content.strip()}\"")
            else:
                print(f"❌ Chat completion failed (Status: {chat_response.status_code})")
                print(f"   Error: {chat_response.text}")
                
                # Suggest solutions
                if chat_response.status_code == 404:
                    print("\nPossible solutions for 404 error:")
                    print("1. Check if you're using a valid model name")
                    print("2. Verify your account has access to the requested model")
                    print("3. Confirm the API endpoint hasn't changed")
                elif chat_response.status_code == 401:
                    print("\nPossible solutions for 401 error:")
                    print("1. Your API key may be invalid or expired")
                    print("2. Your account may have billing issues")
                
        else:
            print(f"❌ Failed to connect (Status: {response.status_code})")
            print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        print("   Check your internet connection and firewall settings")

if __name__ == "__main__":
    print("=== OpenAI API Diagnostic Tool ===")
    check_api_connection()
    print("\nDiagnostics complete!")
    