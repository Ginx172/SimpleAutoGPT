import os
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv('GROQ_API_KEY')
print(f'Groq Key: {groq_key[:10]}...{groq_key[-10:] if groq_key else "None"}')

if groq_key:
    try:
        from groq import Groq
        groq_client = Groq(api_key=groq_key)
        response = groq_client.chat.completions.create(
            model='llama2-70b-4096',
            messages=[{'role': 'user', 'content': 'Test simplu - funcționează?'}],
            max_tokens=20
        )
        print('✅ GROQ FUNCȚIONEAZĂ PERFECT!')
        print(f'Răspuns: {response.choices[0].message.content}')
    except Exception as e:
        print(f'❌ Eroare Groq: {str(e)}')
else:
    print('❌ Nu s-a găsit GROQ_API_KEY în .env')
