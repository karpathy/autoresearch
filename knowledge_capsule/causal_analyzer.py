"""
Causal Analyzer - 实验因果关系分析
"""
from collections import defaultdict

class CausalAnalyzer:
    def __init__(self):
        self.experiments = []
        self.relations = []
    
    def add_experiment(self, config, result):
        exp = {"config": config, "result": result}
        self.experiments.append(exp)
        
        for var, val in config.items():
            self.relations.append({"variable": var, "value": val, "result": result})
    
    def find_key_variables(self):
        var_results = defaultdict(list)
        for r in self.relations:
            var_results[r["variable"]].append(r["result"])
        
        key_vars = []
        for var, results in var_results.items():
            if len(results) > 1:
                variance = max(results) - min(results)
                key_vars.append((var, variance))
        
        return sorted(key_vars, key=lambda x: x[1], reverse=True)
    
    def report(self):
        top_vars = self.find_key_variables()
        return {
            "total_experiments": len(self.experiments),
            "key_variables": [v[0] for v in top_vars[:5]],
            "recommendations": [f"Focus on {v[0]}" for v in top_vars[:3]]
        }
