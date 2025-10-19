#!/usr/bin/env python3
"""
Simple AutoGPT - Versiune simplificatÄƒ funcÈ›ionalÄƒ
AutoGPT care funcÈ›ioneazÄƒ direct pe PC-ul tÄƒu fÄƒrÄƒ dependenÈ›e complexe
"""

import os
import json
import time
import requests
import openai
from datetime import datetime
import logging
from typing import List, Dict, Any
import argparse

# Configurare logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autogpt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleAutoGPT:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.conversation_history = []
        self.tasks_completed = []
        self.current_goal = None
        
        if self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info("âœ… OpenAI API configurat cu succes")
        else:
            logger.warning("âš ï¸ Nu ai setat OpenAI API key.")
            self.client = None
    
    def set_api_key(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        logger.info("âœ… OpenAI API key setat cu succes")
    
    def set_goal(self, goal: str):
        self.current_goal = goal
        logger.info(f"ðŸŽ¯ Obiectiv setat: {goal}")
        
        self.conversation_history.append({
            "role": "system",
            "content": f"Obiectivul tÄƒu este: {goal}. LucreazÄƒ pas cu pas pentru a-l atinge."
        })
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.client:
            return "âŒ OpenAI API key nu este configurat."
        
        try:
            self.conversation_history.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            error_msg = f"âŒ Eroare la generarea rÄƒspunsului: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def run_autonomous_mode(self, max_steps: int = 10):
        if not self.current_goal:
            logger.error("âŒ Nu ai setat un obiectiv.")
            return
        
        logger.info(f"ðŸš€ Pornesc modul autonom pentru: {self.current_goal}")
        
        prompt = f"AnalizeazÄƒ obiectivul '{self.current_goal}' È™i creeazÄƒ 5 paÈ™i concreÈ›i pentru a-l atinge."
        steps_response = self.generate_response(prompt)
        
        logger.info(f"ðŸ“‹ PaÈ™i identificaÈ›i: {steps_response}")
        
        # ExecutÄƒ paÈ™ii
        for i in range(max_steps):
            logger.info(f"ðŸ”„ ExecutÃ¢nd pasul {i+1}...")
            result = self.generate_response(f"ExecutÄƒ pasul {i+1} pentru obiectivul '{self.current_goal}'")
            logger.info(f"âœ… Rezultat: {result}")
            time.sleep(1)
        
        logger.info("ðŸŽ‰ AutoGPT a terminat execuÈ›ia!")
    
    def interactive_mode(self):
        logger.info("ðŸ¤– Modul interactiv pornit. Scrie 'quit' pentru a ieÈ™i.")
        
        while True:
            try:
                user_input = input("\nðŸ”µ Tu: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("ðŸ‘‹ La revedere!")
                    break
                
                if user_input.startswith('goal:'):
                    goal = user_input[5:].strip()
                    self.set_goal(goal)
                    continue
                
                if user_input.lower() == 'run':
                    if self.current_goal:
                        self.run_autonomous_mode()
                    else:
                        logger.info("âŒ Nu ai setat un obiectiv.")
                    continue
                
                response = self.generate_response(user_input)
                logger.info(f"ðŸ¤– AutoGPT: {response}")
                
            except KeyboardInterrupt:
                logger.info("\nðŸ‘‹ La revedere!")
                break
            except Exception as e:
                logger.error(f"âŒ Eroare: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Simple AutoGPT")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--goal", help="Obiectivul pentru AutoGPT")
    
    args = parser.parse_args()
    
    autogpt = SimpleAutoGPT(api_key=args.api_key)
    
    if not autogpt.api_key:
        api_key = input("ðŸ”‘ Introdu OpenAI API key-ul tÄƒu: ").strip()
        if api_key:
            autogpt.set_api_key(api_key)
        else:
            logger.error("âŒ Nu ai introdus API key.")
            return
    
    if args.goal:
        autogpt.set_goal(args.goal)
        autogpt.run_autonomous_mode()
    else:
        autogpt.interactive_mode()

if __name__ == "__main__":
    main()
