from niche_processor import NicheProcessor
import json
from datetime import datetime

print('ðŸŽ¯ TEST AUTOMAT NICHE PROCESSOR')
print('='*50)

processor = NicheProcessor()

# Test cu niÈ™a AI Tools
print('\nðŸ” Testez analiza pentru niÈ™a: AI Tools')
analysis = processor.analyze_niche('AI Tools')
processor.print_analysis(analysis)

# SalveazÄƒ rezultatul
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'niche_analysis_AI_Tools_{timestamp}.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print(f'âœ… Analiza salvatÄƒ Ã®n: {filename}')
