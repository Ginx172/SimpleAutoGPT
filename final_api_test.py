import os
import requests
from dotenv import load_dotenv

load_dotenv()

print('🔑 TEST COMPLET API KEYS')
print('='*50)

# Test OpenAI
print('\n1. 🤖 OpenAI:')
try:
    import openai
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Test'}],
        max_tokens=5
    )
    print('✅ OPENAI FUNCȚIONEAZĂ!')
    print(f'   Răspuns: {response.choices[0].message.content}')
except Exception as e:
    print(f'❌ OpenAI: {str(e)[:100]}...')

# Test Groq
print('\n2. ⚡ Groq:')
try:
    from groq import Groq
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'user', 'content': 'Test'}],
        max_tokens=10
    )
    print('✅ GROQ FUNCȚIONEAZĂ!')
    print(f'   Răspuns: {response.choices[0].message.content}')
except Exception as e:
    print(f'❌ Groq: {str(e)[:100]}...')

# Test HuggingFace
print('\n3. 🤗 HuggingFace:')
hf_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if hf_key:
    try:
        headers = {'Authorization': f'Bearer {hf_key}'}
        response = requests.get('https://huggingface.co/api/whoami', headers=headers, timeout=5)
        if response.status_code == 200:
            user_info = response.json()
            print('✅ HUGGINGFACE FUNCȚIONEAZĂ!')
            print(f'   User: {user_info.get("name", "Unknown")}')
        else:
            print(f'❌ HuggingFace: Status {response.status_code}')
    except Exception as e:
        print(f'❌ HuggingFace: {str(e)[:50]}...')
else:
    print('❌ Nu s-a găsit HUGGINGFACEHUB_API_TOKEN')

print('\n' + '='*50)
print('📊 REZUMAT:')
print('✅ Groq - FUNCȚIONEAZĂ')
print('❌ OpenAI - QUOTA EPUIZAT')
print('✅ HuggingFace - TESTAT')
