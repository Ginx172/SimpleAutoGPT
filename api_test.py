import os
from dotenv import load_dotenv

load_dotenv()

print('ðŸ”‘ TEST API KEYS - STATUS')
print('='*40)

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
        model='llama-3.1-70b-versatile',
        messages=[{'role': 'user', 'content': 'Test'}],
        max_tokens=10
    )
    print('âœ… GROQ FUNCÈšIONEAZÄ‚!')
    print(f'   RÄƒspuns: {response.choices[0].message.content}')
except Exception as e:
    print(f'âŒ Groq: {str(e)[:100]}...')

print('\n' + '='*40)
