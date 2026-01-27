# Medical Diagnosis with Bayesian Networks

## Overview

Medical diagnosis is one of the most successful applications of Probabilistic Graphical Models. Bayesian Networks are particularly well-suited for this domain because:

1. **Uncertainty is inherent**: Symptoms don't deterministically indicate diseases
2. **Causal structure**: Diseases cause symptoms, not vice versa
3. **Expert knowledge**: Doctors can often specify local conditional probabilities
4. **Interpretability**: Reasoning can be explained to patients and reviewed by physicians

## The Medical Diagnosis Problem

Given:
- **Symptoms** $S = \{s_1, \ldots, s_m\}$: Observable findings (fever, cough, test results)
- **Diseases** $D = \{d_1, \ldots, d_k\}$: Hidden conditions to diagnose
- **Risk factors** $R = \{r_1, \ldots, r_p\}$: Patient characteristics (age, smoking)

Goal: Compute $P(\text{Disease} \mid \text{Symptoms}, \text{Risk Factors})$

## Network Structure

A typical medical diagnostic network follows this pattern:

```
     Risk Factors
        /     \
       ↓       ↓
    Disease₁  Disease₂  ...
     /|\        /|\
    ↓ ↓ ↓      ↓ ↓ ↓
   S₁ S₂ S₃   S₄ S₅ S₆
```

**Key structural patterns:**
- Risk factors influence disease probability
- Diseases cause symptoms (causal direction)
- Multiple diseases may share symptoms
- Symptoms are often conditionally independent given diseases

```python
import torch
import numpy as np
from typing import Dict, List, Tuple
from itertools import product as cartesian_product

class MedicalBayesianNetwork:
    """
    A Bayesian Network specialized for medical diagnosis.
    
    Structure: Risk factors → Diseases → Symptoms
    """
    
    def __init__(self):
        self.risk_factors: List[str] = []
        self.diseases: List[str] = []
        self.symptoms: List[str] = []
        
        self.cardinalities: Dict[str, int] = {}
        self.cpts: Dict[str, Tuple[List[str], torch.Tensor]] = {}
    
    def add_risk_factor(self, name: str, cardinality: int = 2):
        """Add a risk factor (e.g., Age, Smoking)."""
        self.risk_factors.append(name)
        self.cardinalities[name] = cardinality
    
    def add_disease(self, name: str, 
                    risk_factor_parents: List[str] = None,
                    base_probability: float = 0.01):
        """Add a disease node."""
        self.diseases.append(name)
        self.cardinalities[name] = 2  # Present or absent
        
        if risk_factor_parents:
            parents = risk_factor_parents
            shape = tuple(self.cardinalities[p] for p in parents) + (2,)
            cpt = torch.ones(shape) * base_probability
            cpt[..., 0] = 1 - cpt[..., 1]
        else:
            parents = []
            cpt = torch.tensor([1 - base_probability, base_probability])
        
        self.cpts[name] = (parents, cpt)
    
    def add_symptom(self, name: str,
                    disease_parents: List[str],
                    sensitivity: Dict[str, float] = None,
                    specificity: float = 0.95):
        """
        Add a symptom node using Noisy-OR model.
        
        Args:
            sensitivity: P(Symptom=1 | Disease=1) for each disease
            specificity: P(Symptom=0 | all Diseases=0)
        """
        self.symptoms.append(name)
        self.cardinalities[name] = 2
        
        parents = disease_parents
        n_diseases = len(disease_parents)
        shape = (2,) * n_diseases + (2,)
        
        cpt = torch.zeros(shape)
        
        if sensitivity is None:
            sensitivity = {d: 0.8 for d in disease_parents}
        
        for idx in cartesian_product(*[range(2) for _ in disease_parents]):
            disease_states = dict(zip(disease_parents, idx))
            
            # Noisy-OR: probability of NO symptom
            prob_no_symptom = specificity
            for d, state in disease_states.items():
                if state == 1:
                    prob_no_symptom *= (1 - sensitivity[d])
            
            cpt[idx + (0,)] = prob_no_symptom
            cpt[idx + (1,)] = 1 - prob_no_symptom
        
        self.cpts[name] = (parents, cpt)
    
    def set_cpt(self, variable: str, parents: List[str], values: torch.Tensor):
        """Manually set a CPT."""
        self.cpts[variable] = (parents, values)
    
    def compute_joint_probability(self, assignment: Dict[str, int]) -> float:
        """Compute P(assignment) using chain rule."""
        prob = 1.0
        all_vars = self.risk_factors + self.diseases + self.symptoms
        
        for var in all_vars:
            parents, cpt = self.cpts.get(var, ([], torch.tensor([0.5, 0.5])))
            
            if parents:
                index = tuple(assignment[p] for p in parents) + (assignment[var],)
            else:
                index = (assignment[var],)
            
            prob *= cpt[index].item()
        
        return prob
    
    def diagnose(self, evidence: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """Compute P(Disease | Evidence) for all diseases."""
        results = {}
        
        for disease in self.diseases:
            probs = torch.zeros(2)
            
            hidden_vars = [d for d in self.diseases if d != disease and d not in evidence]
            hidden_vars += [s for s in self.symptoms if s not in evidence]
            hidden_vars += [r for r in self.risk_factors if r not in evidence]
            
            hidden_cards = [self.cardinalities[v] for v in hidden_vars]
            
            for disease_val in [0, 1]:
                total = 0.0
                
                for hidden_vals in cartesian_product(*[range(c) for c in hidden_cards]):
                    assignment = dict(evidence)
                    assignment[disease] = disease_val
                    assignment.update(dict(zip(hidden_vars, hidden_vals)))
                    total += self.compute_joint_probability(assignment)
                
                probs[disease_val] = total
            
            probs = probs / probs.sum()
            results[disease] = probs
        
        return results
    
    def differential_diagnosis(self, evidence: Dict[str, int], 
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank diseases by posterior probability."""
        results = self.diagnose(evidence)
        disease_probs = [(d, probs[1].item()) for d, probs in results.items()]
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        return disease_probs[:top_k]


def build_respiratory_diagnosis_network() -> MedicalBayesianNetwork:
    """Build a Bayesian Network for respiratory illness diagnosis."""
    bn = MedicalBayesianNetwork()
    
    # Risk factors
    bn.add_risk_factor('Smoking', 2)
    bn.add_risk_factor('Age', 3)  # 0=Young, 1=Middle, 2=Old
    
    bn.cpts['Smoking'] = ([], torch.tensor([0.7, 0.3]))
    bn.cpts['Age'] = ([], torch.tensor([0.3, 0.4, 0.3]))
    
    # Diseases
    bn.add_disease('Cold', base_probability=0.1)
    bn.add_disease('Flu', base_probability=0.05)
    bn.add_disease('Pneumonia', risk_factor_parents=['Age'], base_probability=0.01)
    bn.add_disease('LungCancer', risk_factor_parents=['Smoking', 'Age'], base_probability=0.001)
    bn.add_disease('TB', base_probability=0.001)
    
    # Refined CPTs
    bn.set_cpt('Pneumonia', ['Age'], torch.tensor([
        [0.99, 0.01],   # Young
        [0.98, 0.02],   # Middle
        [0.95, 0.05]    # Old
    ]))
    
    bn.set_cpt('LungCancer', ['Smoking', 'Age'], torch.tensor([
        [[0.9999, 0.0001], [0.999, 0.001], [0.995, 0.005]],  # Non-smoker
        [[0.999, 0.001], [0.99, 0.01], [0.95, 0.05]]         # Smoker
    ]))
    
    # Symptoms
    bn.add_symptom('Fever', 
                   disease_parents=['Cold', 'Flu', 'Pneumonia', 'TB'],
                   sensitivity={'Cold': 0.3, 'Flu': 0.9, 'Pneumonia': 0.8, 'TB': 0.7},
                   specificity=0.95)
    
    bn.add_symptom('Cough',
                   disease_parents=['Cold', 'Flu', 'Pneumonia', 'LungCancer', 'TB'],
                   sensitivity={'Cold': 0.7, 'Flu': 0.8, 'Pneumonia': 0.9, 
                               'LungCancer': 0.6, 'TB': 0.8},
                   specificity=0.9)
    
    bn.add_symptom('Fatigue',
                   disease_parents=['Flu', 'Pneumonia', 'LungCancer', 'TB'],
                   sensitivity={'Flu': 0.8, 'Pneumonia': 0.7, 
                               'LungCancer': 0.8, 'TB': 0.9},
                   specificity=0.85)
    
    bn.add_symptom('WeightLoss',
                   disease_parents=['LungCancer', 'TB'],
                   sensitivity={'LungCancer': 0.6, 'TB': 0.7},
                   specificity=0.98)
    
    return bn


def demonstrate_medical_diagnosis():
    """Demonstrate medical diagnosis with a Bayesian Network."""
    
    print("=" * 70)
    print("Medical Diagnosis with Bayesian Networks")
    print("=" * 70)
    
    bn = build_respiratory_diagnosis_network()
    
    print("\nNetwork Structure:")
    print(f"  Risk Factors: {bn.risk_factors}")
    print(f"  Diseases: {bn.diseases}")
    print(f"  Symptoms: {bn.symptoms}")
    
    # Case 1: Patient with fever and cough
    print("\n" + "-" * 70)
    print("Case 1: Patient presents with Fever and Cough")
    
    evidence = {'Fever': 1, 'Cough': 1}
    diagnosis = bn.differential_diagnosis(evidence)
    
    print("\nDifferential Diagnosis:")
    for disease, prob in diagnosis:
        print(f"  {disease}: {prob*100:.2f}%")
    
    # Case 2: Elderly smoker with concerning symptoms
    print("\n" + "-" * 70)
    print("Case 2: 65-year-old smoker with Cough, Fatigue, Weight Loss")
    
    evidence = {
        'Age': 2, 'Smoking': 1,
        'Cough': 1, 'Fatigue': 1, 'WeightLoss': 1
    }
    diagnosis = bn.differential_diagnosis(evidence)
    
    print("\nDifferential Diagnosis:")
    for disease, prob in diagnosis:
        print(f"  {disease}: {prob*100:.2f}%")

demonstrate_medical_diagnosis()
```

## The Noisy-OR Model

A powerful simplification for medical networks:

$$P(\text{Symptom} = 0 \mid D_1, \ldots, D_n) = (1 - \lambda_0) \prod_{i: D_i = 1} (1 - \lambda_i)$$

**Advantages:**
- Reduces parameters from $O(2^n)$ to $O(n)$
- Intuitive: each disease independently "tries" to cause symptom
- Often medically reasonable assumption

## Real-World Medical Bayesian Networks

| System | Domain | Scale |
|--------|--------|-------|
| QMR-DT | General medicine | ~600 diseases, ~4000 symptoms |
| PATHFINDER | Lymph-node pathology | ~140 variables |
| HEPAR II | Liver disorders | Expert-constructed |

## Summary

Medical Bayesian Networks provide:
1. **Natural causal structure** for disease-symptom relationships
2. **Probabilistic diagnosis** with uncertainty quantification
3. **Interpretable reasoning** for clinical decision support
4. **Efficient computation** with Noisy-OR and other simplifications
