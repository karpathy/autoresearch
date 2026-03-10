"""
Knowledge Capsule Lifecycle Tracker
基于记忆树理念，为实验假设添加生命周期管理
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Optional
import json

class LifecycleStage(Enum):
    SPROUT = "sprout"
    GREEN_LEAF = "green_leaf"
    YELLOW_LEAF = "yellow_leaf"
    RED_LEAF = "red_leaf"
    SOIL = "soil"

class ExperimentCapsule:
    def __init__(self, exp_id: str, hypothesis: str):
        self.id = exp_id
        self.hypothesis = hypothesis
        self.stage = LifecycleStage.SPROUT.value
        self.confidence = 0.7
        self.created_at = datetime.now().isoformat()
        self.validations = []
    
    def validate(self, success: bool):
        if success:
            self.confidence = min(1.0, self.confidence + 0.05)
            if self.confidence >= 0.8:
                self.stage = LifecycleStage.GREEN_LEAF.value
        else:
            self.confidence = max(0.0, self.confidence - 0.1)
            if self.confidence < 0.5:
                self.stage = LifecycleStage.YELLOW_LEAF.value
    
    def to_dict(self):
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "stage": self.stage,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "validations": self.validations
        }

class LifecycleTracker:
    def __init__(self):
        self.experiments: Dict[str, ExperimentCapsule] = {}
    
    def add(self, exp_id: str, hypothesis: str) -> ExperimentCapsule:
        capsule = ExperimentCapsule(exp_id, hypothesis)
        self.experiments[exp_id] = capsule
        return capsule
    
    def validate(self, exp_id: str, success: bool):
        if exp_id in self.experiments:
            self.experiments[exp_id].validate(success)
            self.experiments[exp_id].validations.append({
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_active(self):
        return [e.to_dict() for e in self.experiments.values() 
                if e.stage == LifecycleStage.GREEN_LEAF.value]
    
    def export(self):
        return {k: v.to_dict() for k, v in self.experiments.items()}
