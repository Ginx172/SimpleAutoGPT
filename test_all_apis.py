import os
import requests
from dotenv import load_dotenv

def test_all_apis():
    load_dotenv()
    
    print('🔑 TEST COMPLET API KEYS')
    print('='*50)
    
    # Test OpenAI
    print('\n1. 🤖 OpenAI:')
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[{'role': 'user', 'content': 'Test'}],
                max_tokens=5
            )
            print('✅ OPENAI FUNCȚIONEAZĂ!')
            print(f'   Răspuns: {response.choices[0].message.content}')
        except Exception as e:
            print(f'❌ OpenAI: {str(e)}')
    else:
        print('❌ Nu s-a găsit OPENAI_API_KEY')
    
    # Test Groq
    print('\n2. ⚡ Groq:')
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model='llama-3.1-70b-versatile',
                messages=[{'role': 'user', 'content': 'Test'}],
                max_tokens=10
            )
            print('✅ GROQ FUNCȚIONEAZĂ!')
            print(f'   Răspuns: {response.choices[0].message.content}')
        except Exception as e:
            print(f'❌ Groq: {str(e)}')
    else:
        print('❌ Nu s-a găsit GROQ_API_KEY')
    
    # Test Google
    print('\n3. 🔍 Google:')
    google_key = os.getenv('GOOGLE_API_KEY')
    if google_key:
        try:
            url = f'https://www.googleapis.com/customsearch/v1?key={google_key}&cx=017576662512468239146:omuauf_lfve&q=test'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print('✅ GOOGLE FUNCȚIONEAZĂ!')
            else:
                print(f'❌ Google: Status {response.status_code}')
        except Exception as e:
            print(f'❌ Google: {str(e)}')
    else:
        print('❌ Nu s-a găsit GOOGLE_API_KEY')
    
    # Test HuggingFace
    print('\n4. 🤗 HuggingFace:')
    hf_key = os.getenvabl('HUGGINGFACEHUB_API_TOKEN')
    if hf_key:
        try:
            headers = {'Authorization': f'Bearer {hf_key}'}
            response = requests.get('https://huggingface.co/api/whoami', headers=headers, timeout=10)
            if response.status_code == 200:
                user_info = response.json()
                print('✅ HUGGINGFACE FUNCȚIONEAZĂ!')
                print(f'   User: {user_info.get('name', 'Unknown')}')
            else:
                print(f'❌ HuggingFace: Status {response.status_code}')
        except Exception as e:
            print(f'❌ HuggingFace: {str(e)}')
    else:
        print('❌ Nu s-a găsit HUGGINGFACEHUB_API_TOKEN')
    
    print('\n' + '='*50)

if __name__ == '__main__':
    test_all_apis()
