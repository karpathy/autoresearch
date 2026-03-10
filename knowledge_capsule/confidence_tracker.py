"""
Confidence Tracker - 贝叶斯置信度追踪
"""
import math

class ConfidenceTracker:
    def __init__(self, base_rate=0.3):
        self.base_rate = base_rate
        self.hypotheses = {}
    
    def add_hypothesis(self, hypo_id, description, prior_knowledge=None):
        prior = self.base_rate
        if prior_knowledge:
            for pk in prior_knowledge:
                prior *= pk.get('confidence', 0.5)
            prior = min(1.0, prior * 2)
        
        self.hypotheses[hypo_id] = {
            "id": hypo_id,
            "description": description,
            "confidence": prior,
            "evidence": []
        }
        return self.hypotheses[hypo_id]
    
    def update(self, hypo_id, evidence):
        if hypo_id not in self.hypotheses:
            return None
        
        h = self.hypotheses[hypo_id]
        p_h = h["confidence"]
        
        if evidence.get("success"):
            likelihood = 0.8
        else:
            likelihood = 0.3
        
        p_h_new = (likelihood * p_h) / (likelihood * p_h + (1 - likelihood) * (1 - p_h))
        h["confidence"] = p_h_new
        h["evidence"].append(evidence)
        
        return p_h_new
    
    def get_top(self, n=5):
        return sorted(self.hypotheses.values(), key=lambda x: x["confidence"], reverse=True)[:n]
