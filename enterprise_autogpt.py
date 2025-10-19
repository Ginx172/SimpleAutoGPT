#!/usr/bin/env python3
"""
Enterprise AutoGPT - Versiune avansatÄƒ cu capabilitÄƒÈ›i Enterprise
- Coding automatizat cu multiple AI providers
- MCP (Model Context Protocol) support
- Advanced debugging È™i profiling
- AI model benchmarking
- Multi-provider integration
"""

import os
import json
import time
import requests
import subprocess
import threading
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
import argparse
import asyncio
import aiohttp
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_autogpt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    provider: str
    model: str
    prompt: str
    response_time: float
    token_count: int
    quality_score: float
    cost_estimate: float
    timestamp: str

class MCPClient:
    def __init__(self):
        self.connections = {}
        logger.info("ðŸ¤– MCP Client iniÈ›ializat")
    
    async def connect_to_model(self, provider: str, model: str, api_key: str):
        try:
            connection = {
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "connected_at": datetime.now().isoformat()
            }
            self.connections[f"{provider}_{model}"] = connection
            logger.info(f"âœ… Conectat la {provider}/{model} prin MCP")
            return True
        except Exception as e:
            logger.error(f"âŒ Eroare conectare MCP {provider}/{model}: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        return list(self.connections.keys())

class AIProviderManager:
    def __init__(self):
        self.providers = {}
        self.load_api_keys()
        self.initialize_providers()
    
    def load_api_keys(self):
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            logger.info("âœ… Chei API Ã®ncÄƒrcate din .env")
        except Exception as e:
            logger.warning(f"âš ï¸ Nu s-au putut Ã®ncÄƒrca cheile API: {str(e)}")
    
    def initialize_providers(self):
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                self.providers["openai"] = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("âœ… OpenAI provider iniÈ›ializat")
            except ImportError:
                logger.warning("âš ï¸ OpenAI library nu este instalat")
        
        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from anthropic import Anthropic
                self.providers["anthropic"] = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                logger.info("âœ… Anthropic provider iniÈ›ializat")
            except ImportError:
                logger.warning("âš ï¸ Anthropic library nu este instalat")
        
        # Groq
        if os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                self.providers["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logger.info("âœ… Groq provider iniÈ›ializat")
            except ImportError:
                logger.warning("âš ï¸ Groq library nu este instalat")
    
    def get_response(self, provider: str, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            if provider == "openai":
                response = self.providers["openai"].chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens if response.usage else 0
            
            elif provider == "anthropic":
                response = self.providers["anthropic"].messages.create(
                    model=model,
                    max_tokens=kwargs.get("max_tokens", 1000),
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
            
            elif provider == "groq":
                response = self.providers["groq"].chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens if response.usage else 0
            
            else:
                return {"error": f"Provider {provider} nu este suportat"}
            
            response_time = time.time() - start_time
            
            return {
                "content": content,
                "provider": provider,
                "model": model,
                "response_time": response_time,
                "tokens": tokens,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"âŒ Eroare {provider}/{model}: {str(e)}")
            return {
                "error": str(e),
                "provider": provider,
                "model": model,
                "success": False
            }

class CodeGenerator:
    def __init__(self, ai_manager: AIProviderManager):
        self.ai_manager = ai_manager
        logger.info("ðŸ’» Code Generator iniÈ›ializat")
    
    def generate_code(self, description: str, language: str = "python", 
                     provider: str = "openai", model: str = "gpt-4") -> Dict[str, Any]:
        prompt = f"""
GenereazÄƒ cod {language} pentru: {description}

CerinÈ›e:
- Cod curat È™i bine documentat
- Include comentarii explicative
- RespectÄƒ best practices pentru {language}
- Include error handling unde este necesar
- ReturneazÄƒ doar codul, fÄƒrÄƒ explicaÈ›ii suplimentare

Cod:
"""
        
        result = self.ai_manager.get_response(provider, model, prompt)
        
        if result["success"]:
            code_analysis = self.analyze_code(result["content"])
            result.update(code_analysis)
        
        return result
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        lines = code.split('\n')
        
        analysis = {
            "lines_count": len(lines),
            "functions_count": code.count('def '),
            "classes_count": code.count('class '),
            "comments_count": sum(1 for line in lines if line.strip().startswith('#')),
            "complexity_score": self.calculate_complexity(code),
            "has_docstrings": '"""' in code or "'''" in code,
            "has_error_handling": 'try:' in code or 'except' in code
        }
        
        return analysis
    
    def calculate_complexity(self, code: str) -> float:
        complexity_indicators = [
            code.count('if '),
            code.count('for '),
            code.count('while '),
            code.count('try:'),
            code.count('except'),
            code.count('def '),
            code.count('class ')
        ]
        return sum(complexity_indicators) / 10.0

class Debugger:
    def __init__(self):
        self.debug_sessions = {}
        logger.info("ðŸ› Debugger iniÈ›ializat")
    
    def analyze_error(self, error: str, code: str) -> Dict[str, Any]:
        analysis = {
            "error_type": self.classify_error(error),
            "severity": self.assess_severity(error),
            "suggested_fixes": [],
            "related_lines": [],
            "debugging_steps": []
        }
        
        if "SyntaxError" in error:
            analysis["suggested_fixes"].append("VerificÄƒ sintaxa codului")
            analysis["debugging_steps"].append("FoloseÈ™te un linter pentru Python")
        
        elif "NameError" in error:
            analysis["suggested_fixes"].append("VerificÄƒ dacÄƒ variabilele sunt definite")
            analysis["debugging_steps"].append("VerificÄƒ scope-ul variabilelor")
        
        elif "TypeError" in error:
            analysis["suggested_fixes"].append("VerificÄƒ tipurile de date")
            analysis["debugging_steps"].append("AdaugÄƒ type hints")
        
        return analysis
    
    def classify_error(self, error: str) -> str:
        error_types = {
            "SyntaxError": "Eroare de sintaxÄƒ",
            "NameError": "VariabilÄƒ nedefinitÄƒ",
            "TypeError": "Eroare de tip",
            "ValueError": "Valoare invalidÄƒ",
            "IndexError": "Index invalid",
            "KeyError": "Cheie invalidÄƒ",
            "AttributeError": "Atribut inexistent",
            "ImportError": "Eroare de import"
        }
        
        for error_type, description in error_types.items():
            if error_type in error:
                return description
        
        return "Eroare necunoscutÄƒ"
    
    def assess_severity(self, error: str) -> str:
        critical_errors = ["SyntaxError", "ImportError"]
        high_errors = ["TypeError", "ValueError", "AttributeError"]
        
        for error_type in critical_errors:
            if error_type in error:
                return "Critical"
        
        for error_type in high_errors:
            if error_type in error:
                return "High"
        
        return "Medium"

class AIBenchmarker:
    def __init__(self, ai_manager: AIProviderManager):
        self.ai_manager = ai_manager
        self.benchmark_results = []
        self.test_prompts = self.load_test_prompts()
        logger.info("ðŸ“Š AI Benchmarker iniÈ›ializat")
    
    def load_test_prompts(self) -> List[Dict[str, Any]]:
        return [
            {
                "category": "coding",
                "prompt": "Scrie o funcÈ›ie Python care calculeazÄƒ factorialul unui numÄƒr",
                "expected_keywords": ["def", "factorial", "return", "if"]
            },
            {
                "category": "reasoning",
                "prompt": "DacÄƒ o pisicÄƒ are 4 picioare È™i un cÃ¢ine are 4 picioare, cÃ¢te picioare au Ã®mpreunÄƒ?",
                "expected_keywords": ["8", "picioare"]
            },
            {
                "category": "creative",
                "prompt": "Scrie o povestire scurtÄƒ despre un robot care Ã®nvaÈ›Äƒ sÄƒ iubeascÄƒ",
                "expected_keywords": ["robot", "iubire", "Ã®nvaÈ›Äƒ"]
            }
        ]
    
    async def benchmark_model(self, provider: str, model: str) -> List[BenchmarkResult]:
        logger.info(f"ðŸ”„ Benchmark pentru {provider}/{model}")
        results = []
        
        for test in self.test_prompts:
            result = self.ai_manager.get_response(
                provider, model, test["prompt"], max_tokens=500
            )
            
            if result["success"]:
                quality_score = self.calculate_quality_score(
                    result["content"], test["expected_keywords"]
                )
                
                benchmark_result = BenchmarkResult(
                    provider=provider,
                    model=model,
                    prompt=test["prompt"],
                    response_time=result["response_time"],
                    token_count=result["tokens"],
                    quality_score=quality_score,
                    cost_estimate=self.estimate_cost(provider, result["tokens"]),
                    timestamp=datetime.now().isoformat()
                )
                
                results.append(benchmark_result)
                self.benchmark_results.append(benchmark_result)
        
        return results
    
    def calculate_quality_score(self, response: str, expected_keywords: List[str]) -> float:
        if not response:
            return 0.0
        
        response_lower = response.lower()
        found_keywords = sum(1 for keyword in expected_keywords 
                           if keyword.lower() in response_lower)
        
        return (found_keywords / len(expected_keywords)) * 100
    
    def estimate_cost(self, provider: str, tokens: int) -> float:
        pricing = {
            "openai": {"gpt-4": 0.03, "gpt-3.5-turbo": 0.002},
            "anthropic": {"claude-3": 0.015},
            "groq": {"llama2": 0.001}
        }
        
        if provider in pricing:
            for model, price in pricing[provider].items():
                return (tokens / 1000) * price
        
        return 0.0
    
    def generate_benchmark_report(self) -> str:
        if not self.benchmark_results:
            return "Nu existÄƒ rezultate de benchmark"
        
        report = "# ðŸ“Š RAPORT BENCHMARK AI MODELS\n\n"
        
        by_provider = {}
        for result in self.benchmark_results:
            if result.provider not in by_provider:
                by_provider[result.provider] = []
            by_provider[result.provider].append(result)
        
        for provider, results in by_provider.items():
            report += f"## ðŸ¤– {provider.upper()}\n\n"
            
            avg_response_time = sum([r.response_time for r in results]) / len(results)
            avg_quality = sum([r.quality_score for r in results]) / len(results)
            total_cost = sum([r.cost_estimate for r in results])
            
            report += f"- **Timp mediu de rÄƒspuns:** {avg_response_time:.2f}s\n"
            report += f"- **Calitate medie:** {avg_quality:.1f}%\n"
            report += f"- **Cost total estimat:** \n\n"
        
        return report

class EnterpriseAutoGPT:
    def __init__(self):
        self.ai_manager = AIProviderManager()
        self.mcp_client = MCPClient()
        self.code_generator = CodeGenerator(self.ai_manager)
        self.debugger = Debugger()
        self.benchmarker = AIBenchmarker(self.ai_manager)
        self.conversation_history = []
        self.current_goal = None
        logger.info("ðŸš€ Enterprise AutoGPT iniÈ›ializat cu succes")
    
    async def initialize_enterprise_features(self):
        logger.info("ðŸ”§ IniÈ›ializez funcÈ›ionalitÄƒÈ›i Enterprise...")
        
        for provider in self.ai_manager.providers.keys():
            await self.mcp_client.connect_to_model(
                provider, "default", os.getenv(f"{provider.upper()}_API_KEY")
            )
        
        logger.info("âœ… FuncÈ›ionalitÄƒÈ›i Enterprise iniÈ›ializate")
    
    def generate_code_enterprise(self, description: str, language: str = "python") -> Dict[str, Any]:
        logger.info(f"ðŸ’» Generez cod {language} pentru: {description}")
        
        best_result = None
        best_score = 0
        
        for provider in self.ai_manager.providers.keys():
            result = self.code_generator.generate_code(description, language, provider)
            if result["success"] and result.get("quality_score", 0) > best_score:
                best_result = result
                best_score = result.get("quality_score", 0)
        
        return best_result or {"error": "Nu s-a putut genera cod"}
    
    def debug_code_enterprise(self, code: str, error: str = None) -> Dict[str, Any]:
        logger.info("ðŸ› ÃŽncep debug enterprise pentru cod")
        
        if error:
            error_analysis = self.debugger.analyze_error(error, code)
        else:
            error_analysis = self.simulate_code_execution(code)
        
        return {
            "code_analysis": self.code_generator.analyze_code(code),
            "error_analysis": error_analysis,
            "suggestions": self.generate_debug_suggestions(code, error_analysis)
        }
    
    def simulate_code_execution(self, code: str) -> Dict[str, Any]:
        potential_issues = []
        
        if "import " in code and "os." in code:
            potential_issues.append("Codul foloseÈ™te operaÈ›ii OS - poate fi nesigur")
        
        if "eval(" in code or "exec(" in code:
            potential_issues.append("Codul foloseÈ™te eval/exec - risc de securitate")
        
        return {
            "potential_issues": potential_issues,
            "severity": "Low" if not potential_issues else "Medium"
        }
    
    def generate_debug_suggestions(self, code: str, error_analysis: Dict[str, Any]) -> List[str]:
        suggestions = []
        
        if error_analysis.get("severity") == "Critical":
            suggestions.append("VerificÄƒ sintaxa codului cu un linter")
            suggestions.append("TesteazÄƒ codul Ã®ntr-un mediu izolat")
        
        if error_analysis.get("severity") == "High":
            suggestions.append("AdaugÄƒ logging pentru a urmÄƒri execuÈ›ia")
            suggestions.append("FoloseÈ™te un debugger interactiv")
        
        suggestions.extend([
            "AdaugÄƒ unit tests pentru funcÈ›ionalitate",
            "FoloseÈ™te type hints pentru claritate",
            "DocumenteazÄƒ funcÈ›iile È™i clasele"
        ])
        
        return suggestions
    
    async def run_benchmark_suite(self) -> str:
        logger.info("ðŸ“Š Pornesc benchmark suite complet")
        
        benchmark_results = []
        
        for provider in self.ai_manager.providers.keys():
            results = await self.benchmarker.benchmark_model(provider, "default")
            benchmark_results.extend(results)
        
        report = self.benchmarker.generate_benchmark_report()
        self.save_benchmark_results(benchmark_results)
        
        return report
    
    def save_benchmark_results(self, results: List[BenchmarkResult]):
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = []
        for result in results:
            data.append({
                "provider": result.provider,
                "model": result.model,
                "prompt": result.prompt,
                "response_time": result.response_time,
                "token_count": result.token_count,
                "quality_score": result.quality_score,
                "cost_estimate": result.cost_estimate,
                "timestamp": result.timestamp
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"ðŸ’¾ Rezultate benchmark salvate Ã®n {filename}")
    
    def interactive_enterprise_mode(self):
        logger.info("ðŸ¤– Modul interactiv Enterprise pornit")
        logger.info("ðŸ’¡ Comenzi disponibile:")
        logger.info("  - 'code: <descriere>' - GenereazÄƒ cod")
        logger.info("  - 'debug: <cod>' - Debug cod")
        logger.info("  - 'benchmark' - RuleazÄƒ benchmark")
        logger.info("  - 'mcp status' - Status MCP")
        logger.info("  - 'providers' - ListeazÄƒ provideri")
        logger.info("  - 'quit' - IeÈ™ire")
        
        while True:
            try:
                user_input = input("\nðŸ”µ Enterprise AutoGPT: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("ðŸ‘‹ La revedere!")
                    break
                
                if user_input.startswith('code:'):
                    description = user_input[5:].strip()
                    result = self.generate_code_enterprise(description)
                    if result.get("success"):
                        logger.info(f"âœ… Cod generat cu {result['provider']}/{result['model']}")
                        logger.info(f"ðŸ“Š Complexitate: {result.get('complexity_score', 0):.2f}")
                        print(f"\nðŸ’» COD GENERAT:\n{result['content']}")
                    else:
                        logger.error(f"âŒ Eroare: {result.get('error')}")
                    continue
                
                if user_input.startswith('debug:'):
                    code = user_input[6:].strip()
                    result = self.debug_code_enterprise(code)
                    logger.info("ðŸ› AnalizÄƒ debug completatÄƒ")
                    print(f"\nðŸ” ANALIZÄ‚ COD:\n{json.dumps(result['code_analysis'], indent=2)}")
                    if result['error_analysis']:
                        print(f"\nâš ï¸ ANALIZÄ‚ EROARE:\n{json.dumps(result['error_analysis'], indent=2)}")
                    print(f"\nðŸ’¡ SUGESTII:\n{chr(10).join(result['suggestions'])}")
                    continue
                
                if user_input.lower() == 'benchmark':
                    logger.info("ðŸ“Š Pornesc benchmark...")
                    asyncio.run(self.run_benchmark_suite())
                    continue
                
                if user_input.lower() == 'mcp status':
                    models = self.mcp_client.get_available_models()
                    logger.info(f"ðŸ”— Modele MCP conectate: {models}")
                    continue
                
                if user_input.lower() == 'providers':
                    providers = list(self.ai_manager.providers.keys())
                    logger.info(f"ðŸ¤– Provideri disponibili: {providers}")
                    continue
                
                # RÄƒspunde la Ã®ntrebarea utilizatorului cu cel mai bun provider
                best_response = None
                for provider in self.ai_manager.providers.keys():
                    response = self.ai_manager.get_response(provider, "default", user_input)
                    if response["success"] and (not best_response or response["response_time"] < best_response["response_time"]):
                        best_response = response
                
                if best_response:
                    logger.info(f"ðŸ¤– RÄƒspuns de la {best_response['provider']} ({best_response['response_time']:.2f}s):")
                    print(f"\n{best_response['content']}")
                else:
                    logger.error("âŒ Nu s-a putut obÈ›ine rÄƒspuns")
                
            except KeyboardInterrupt:
                logger.info("\nðŸ‘‹ La revedere!")
                break
            except Exception as e:
                logger.error(f"âŒ Eroare: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description="Enterprise AutoGPT")
    parser.add_argument("--benchmark", action="store_true", help="RuleazÄƒ benchmark complet")
    parser.add_argument("--code", help="GenereazÄƒ cod pentru descrierea datÄƒ")
    parser.add_argument("--debug", help="Debug codul dat")
    parser.add_argument("--interactive", action="store_true", help="Modul interactiv")
    
    args = parser.parse_args()
    
    enterprise_autogpt = EnterpriseAutoGPT()
    await enterprise_autogpt.initialize_enterprise_features()
    
    if args.benchmark:
        report = await enterprise_autogpt.run_benchmark_suite()
        print(report)
    
    elif args.code:
        result = enterprise_autogpt.generate_code_enterprise(args.code)
        if result.get("success"):
            print(f"ðŸ’» Cod generat cu {result['provider']}/{result['model']}:")
            print(result['content'])
        else:
            print(f"âŒ Eroare: {result.get('error')}")
    
    elif args.debug:
        result = enterprise_autogpt.debug_code_enterprise(args.debug)
        print(json.dumps(result, indent=2))
    
    elif args.interactive:
        enterprise_autogpt.interactive_enterprise_mode()
    
    else:
        enterprise_autogpt.interactive_enterprise_mode()

if __name__ == "__main__":
    asyncio.run(main())
