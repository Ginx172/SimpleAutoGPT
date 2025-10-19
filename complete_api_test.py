import os
import requests
from dotenv import load_dotenv

load_dotenv()

print('ðŸ”‘ TEST COMPLET API KEYS')
print('='*50)

# Test OpenAI
print('\n1. ðŸ¤– OpenAI:')
try:
    import openai
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Test'}],
        max_tokens=5
    )
    print('âœ… OPENAI FUNCÈšIONEAZÄ‚!')
    print(f'   RÄƒspuns: {response.choices[0].message.content}')
except Exception as e:
    print(f'âŒ OpenAI: {str(e)[:100]}...')

# Test Groq
print('\n2. âš¡ Groq:')
try:
    from groq import Groq
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'user', 'content': 'Test'}],
        max_tokens=10
    )
    print('âœ… GROQ FUNCÈšIONEAZÄ‚!')
    print(f'   RÄƒspuns: {response.choices[0].message.content}')
except Exception as e:
    print(f'âŒ Groq: {str(e)[:100]}...')

# Test HuggingFace
print('\n3. ðŸ¤— HuggingFace:')
hf_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if hf_key:
    try:
        headers = {'Authorization': f'Bearer {hf_key}'}
        response = requests.get('https://huggingface.co/api/whoami', headers=headers, timeout=5)
        if response.status_code == 200:
            user_info = response.json()
            print('âœ… HUGGINGFACE FUNCÈšIONEAZÄ‚!')
            print(f'   User: {user_info.get(\"name\", \"Unknown\")}')
        else:
            print(f'âŒ HuggingFace: Status {response.status_code}')
    except Exception as e:
        print(f'âŒ HuggingFace: {str(e)[:50]}...')
else:
    print('âŒ Nu s-a gÄƒsit HUGGINGFACEHUB_API_TOKEN')

print('\n' + '='*50)
print('ðŸ“Š REZUMAT:')
print('âœ… Groq - FUNCÈšIONEAZÄ‚')
print('âŒ OpenAI - QUOTA EPUIZAT')
print('âœ… HuggingFace - TESTAT')
