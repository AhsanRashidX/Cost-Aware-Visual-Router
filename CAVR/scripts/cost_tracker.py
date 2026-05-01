"""
Real-time cost tracking for API calls and GPU usage
"""

import time
import json
import psutil
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import GPUtil

class RealCostTracker:
    """
    Track real costs from API calls and GPU usage
    """
    
    def __init__(self, log_file: str = "./logs/real_costs.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Real API pricing (as of 2024-2025)
        self.api_pricing = {
            # OpenAI
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},  # per 1K tokens
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            
            # Embedding models
            'text-embedding-3-small': 0.00002,  # per 1K tokens
            'text-embedding-3-large': 0.00013,
            'ada-v2': 0.0001,
            
            # GPU pricing (cloud providers)
            'gpu_hourly': {
                'A100_40GB': 1.50,   # AWS/Azure per hour
                'A100_80GB': 2.50,
                'V100': 1.00,
                'T4': 0.50,
                'L4': 0.60,
            },
            
            # ColPali specific
            'colpali_per_query': 0.01,
        }
        
        self.session_costs = []
        self.current_session = {
            'start_time': datetime.now().isoformat(),
            'api_calls': [],
            'gpu_usage': [],
            'total_cost': 0.0
        }
    
    def log_api_call(self, model: str, input_tokens: int, output_tokens: int, 
                     api_type: str = 'openai'):
        """
        Log an actual API call with token usage
        
        Args:
            model: Model name (e.g., 'gpt-3.5-turbo', 'text-embedding-3-small')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            api_type: 'openai', 'anthropic', 'together', 'replicate'
        """
        if api_type == 'openai':
            if model in self.api_pricing:
                if isinstance(self.api_pricing[model], dict):
                    input_cost = (input_tokens / 1000) * self.api_pricing[model]['input']
                    output_cost = (output_tokens / 1000) * self.api_pricing[model]['output']
                else:
                    # Embedding model (flat rate per 1K tokens)
                    input_cost = (input_tokens / 1000) * self.api_pricing[model]
                    output_cost = 0
            else:
                input_cost = 0.001  # fallback
                output_cost = 0.003
        else:
            input_cost = 0.001
            output_cost = 0.003
        
        call_cost = input_cost + output_cost
        
        api_call = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'api_type': api_type,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': call_cost,
            'latency_ms': 0  # Will be filled separately
        }
        
        self.current_session['api_calls'].append(api_call)
        self.current_session['total_cost'] += call_cost
        
        return call_cost
    
    def log_gpu_usage(self, duration_seconds: float, gpu_type: str = 'A100_40GB'):
        """
        Log GPU usage cost
        
        Args:
            duration_seconds: GPU usage duration in seconds
            gpu_type: Type of GPU (A100_40GB, V100, T4, etc.)
        """
        hourly_rate = self.api_pricing['gpu_hourly'].get(gpu_type, 1.0)
        gpu_cost = (duration_seconds / 3600) * hourly_rate
        
        gpu_record = {
            'timestamp': datetime.now().isoformat(),
            'gpu_type': gpu_type,
            'duration_seconds': duration_seconds,
            'hourly_rate': hourly_rate,
            'total_cost': gpu_cost
        }
        
        self.current_session['gpu_usage'].append(gpu_record)
        self.current_session['total_cost'] += gpu_cost
        
        return gpu_cost
    
    def log_query_cost(self, query: str, path_used: str, 
                       actual_latency_ms: float,
                       gpu_time_seconds: float = None):
        """
        Log complete cost for a single query
        
        Args:
            query: The query text
            path_used: 'parametric', 'text', 'visual', 'hybrid'
            actual_latency_ms: Actual API/GPU latency
            gpu_time_seconds: GPU processing time (if applicable)
        """
        query_record = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:200],
            'path_used': path_used,
            'latency_ms': actual_latency_ms,
            'components': []
        }
        
        # Cost based on path
        if path_used == 'parametric':
            # LLM call (e.g., GPT-3.5)
            cost = self.log_api_call('gpt-3.5-turbo', 
                                    input_tokens=len(query.split()), 
                                    output_tokens=50)
            query_record['components'].append({'type': 'llm', 'cost': cost})
            query_record['total_cost'] = cost
            
        elif path_used == 'text':
            # Embedding + LLM
            embed_cost = self.log_api_call('text-embedding-3-small',
                                          input_tokens=len(query.split()),
                                          output_tokens=0)
            llm_cost = self.log_api_call('gpt-3.5-turbo',
                                        input_tokens=500,
                                        output_tokens=100)
            query_record['components'].append({'type': 'embedding', 'cost': embed_cost})
            query_record['components'].append({'type': 'llm', 'cost': llm_cost})
            query_record['total_cost'] = embed_cost + llm_cost
            
        elif path_used == 'visual':
            # ColPali GPU cost
            if gpu_time_seconds:
                gpu_cost = self.log_gpu_usage(gpu_time_seconds, 'A100_40GB')
                query_record['components'].append({'type': 'gpu', 'cost': gpu_cost})
                query_record['total_cost'] = gpu_cost
            else:
                query_record['total_cost'] = 0.01  # Estimate fallback
                
        elif path_used == 'hybrid':
            # Text + Visual
            embed_cost = self.log_api_call('text-embedding-3-small',
                                          input_tokens=len(query.split()),
                                          output_tokens=0)
            if gpu_time_seconds:
                gpu_cost = self.log_gpu_usage(gpu_time_seconds, 'A100_40GB')
                query_record['components'].append({'type': 'embedding', 'cost': embed_cost})
                query_record['components'].append({'type': 'gpu', 'cost': gpu_cost})
                query_record['total_cost'] = embed_cost + gpu_cost
            else:
                query_record['total_cost'] = 0.0105
        
        self.current_session['api_calls'].append(query_record)
        self.current_session['total_cost'] += query_record['total_cost']
        
        return query_record
    
    def get_real_gpu_metrics(self):
        """Get real GPU metrics using nvidia-smi"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'gpu_utilization_percent': gpu.load * 100,
                    'gpu_memory_used_mb': gpu.memoryUsed,
                    'gpu_memory_total_mb': gpu.memoryTotal,
                    'gpu_temperature_celsius': gpu.temperature
                }
        except:
            pass
        return {
            'gpu_utilization_percent': 0,
            'gpu_memory_used_mb': 0,
            'gpu_memory_total_mb': 0,
            'gpu_temperature_celsius': 0
        }
    
    def save_session(self):
        """Save current session costs to file"""
        self.current_session['end_time'] = datetime.now().isoformat()
        
        # Calculate metrics
        total_queries = len([c for c in self.current_session['api_calls'] if 'query' in c])
        total_api_cost = sum(c.get('total_cost', 0) for c in self.current_session['api_calls'] 
                            if 'components' not in c)
        total_query_cost = sum(c.get('total_cost', 0) for c in self.current_session['api_calls'] 
                              if 'components' in c)
        
        self.current_session['summary'] = {
            'total_queries': total_queries,
            'total_api_cost': total_api_cost,
            'total_query_cost': total_query_cost,
            'total_cost': self.current_session['total_cost'],
            'avg_cost_per_query': self.current_session['total_cost'] / max(total_queries, 1)
        }
        
        # Save to file
        existing = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                existing = json.load(f)
        
        existing.append(self.current_session)
        
        with open(self.log_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print(f"\n💰 Cost Session Saved:")
        print(f"   Total Queries: {total_queries}")
        print(f"   Total Cost: ${self.current_session['total_cost']:.4f}")
        print(f"   Avg Cost/Query: ${self.current_session['summary']['avg_cost_per_query']:.4f}")
        
        return self.current_session
    
    def reset_session(self):
        """Start a new cost tracking session"""
        self.current_session = {
            'start_time': datetime.now().isoformat(),
            'api_calls': [],
            'gpu_usage': [],
            'total_cost': 0.0
        }