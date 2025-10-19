#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('niche_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NicheProcessor:
    def __init__(self):
        load_dotenv()
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.niches = {
            'AI Tools': {
                'keywords': ['AI tools', 'artificial intelligence', 'automation', 'productivity'],
                'competitors': [],
                'trends': [],
                'gaps': []
            },
            'AI Courses': {
                'keywords': ['AI courses', 'machine learning', 'deep learning', 'AI education'],
                'competitors': [],
                'trends': [],
                'gaps': []
            },
            'Tech News': {
                'keywords': ['tech news', 'startup', 'innovation', 'technology'],
                'competitors': [],
                'trends': [],
                'gaps': []
            }
        }
        logger.info('ðŸŽ¯ Niche Processor iniÈ›ializat cu GROQ')
    
    def generate_with_groq(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            response = self.groq_client.chat.completions.create(
                model='llama-3.1-8b-instant',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f'Eroare GROQ: {str(e)}')
            return f'Eroare la generarea conÈ›inutului: {str(e)}'
    
    def analyze_niche(self, niche_name: str) -> Dict[str, Any]:
        logger.info(f'ðŸ” Analizez niÈ™a: {niche_name}')
        
        if niche_name not in self.niches:
            return {'error': f'NiÈ™a {niche_name} nu este definitÄƒ'}
        
        niche_data = self.niches[niche_name]
        
        # GenereazÄƒ analiza cu GROQ
        prompt = f'''
        AnalizeazÄƒ niÈ™a "{niche_name}" pentru social media growth.
        Keywords: {', '.join(niche_data['keywords'])}
        
        GenereazÄƒ:
        1. Top 5 competitori cu engagement ridicat
        2. 5 trending topics actuale
        3. 5 gaps Ã®n piaÈ›Äƒ exploatabile
        4. 5 idei de conÈ›inut viral
        5. Calendar editorial pentru 7 zile
        6. KPI-uri È›intÄƒ
        
        RÄƒspunde Ã®n format JSON structurat.
        '''
        
        groq_response = self.generate_with_groq(prompt, max_tokens=1000)
        
        analysis = {
            'niche': niche_name,
            'keywords': niche_data['keywords'],
            'groq_analysis': groq_response,
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict[str, Any]):
        print('\n' + '='*60)
        print('ðŸŽ¯ ANALIZA NIÈ˜Ä‚ SOCIAL MEDIA')
        print('='*60)
        print(f'\nðŸ“Š NiÈ™a: {analysis["niche"]}')
        print(f'\nðŸ”‘ Keywords: {", ".join(analysis["keywords"])}')
        print(f'\nðŸ¤– Analiza GROQ:')
        print(analysis['groq_analysis'])
        print('\n' + '='*60)

def main():
    processor = NicheProcessor()
    
    print('ðŸŽ¯ NICHE PROCESSOR - Sistem de AnalizÄƒ NiÈ™e Social Media')
    print('='*60)
    
    available_niches = list(processor.niches.keys())
    print(f'\nðŸ“Š NiÈ™e disponibile: {", ".join(available_niches)}')
    
    while True:
        print('\nðŸ’¡ Comenzi disponibile:')
        print('  1. "analyze <niche>" - AnalizeazÄƒ o niÈ™Äƒ')
        print('  2. "list" - ListeazÄƒ niÈ™ele disponibile')
        print('  3. "quit" - IeÈ™ire')
        
        command = input('\nðŸ”µ ComandÄƒ: ').strip().lower()
        
        if command.startswith('analyze '):
            niche = command[8:].strip()
            if niche in processor.niches:
                analysis = processor.analyze_niche(niche)
                processor.print_analysis(analysis)
                
                save = input('\nðŸ’¾ Salvezi analiza Ã®n fiÈ™ier? (y/n): ').strip().lower()
                if save == 'y':
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'niche_analysis_{niche.replace(" ", "_")}_{timestamp}.json'
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(analysis, f, indent=2, ensure_ascii=False)
                    print(f'âœ… Analiza salvatÄƒ Ã®n: {filename}')
            else:
                print(f'âŒ NiÈ™a {niche} nu este disponibilÄƒ. NiÈ™e disponibile: {", ".join(available_niches)}')
        
        elif command == 'list':
            print(f'\nðŸ“Š NiÈ™e disponibile:')
            for i, niche in enumerate(available_niches, 1):
                print(f'  {i}. {niche}')
        
        elif command in ['quit', 'exit', 'q']:
            print('ðŸ‘‹ La revedere!')
            break
        
        else:
            print('âŒ ComandÄƒ invalidÄƒ. ÃŽncearcÄƒ din nou.')

if __name__ == '__main__':
    main()
