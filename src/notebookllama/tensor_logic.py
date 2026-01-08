"""
Tensor-Logic Orchestration Module for RAGSwarm Neural Notebook-LM.

This module implements a neuro-symbolic reasoning framework that combines
tensor-based neural operations with logical symbolic reasoning.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from datetime import datetime

from .opencog_atomspace import Atom, AtomType, AtomSpace


class TensorType(Enum):
    """Types of tensors in the neuro-symbolic system."""
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    ACTIVATION = "activation"
    LOGIC_STATE = "logic_state"
    KNOWLEDGE_VECTOR = "knowledge_vector"


@dataclass
class Tensor:
    """
    Represents a tensor in the neuro-symbolic fabric.
    Simplified tensor implementation without heavy dependencies.
    """
    shape: Tuple[int, ...]
    data: List[float]
    tensor_type: TensorType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate tensor dimensions."""
        expected_size = 1
        for dim in self.shape:
            expected_size *= dim
        
        if len(self.data) != expected_size:
            raise ValueError(f"Data size {len(self.data)} doesn't match shape {self.shape}")
    
    def dot(self, other: 'Tensor') -> float:
        """Compute dot product with another tensor (must be 1D)."""
        if len(self.shape) != 1 or len(other.shape) != 1:
            raise ValueError("Dot product only supported for 1D tensors")
        if self.shape[0] != other.shape[0]:
            raise ValueError("Tensor dimensions don't match")
        
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def norm(self) -> float:
        """Compute L2 norm of the tensor."""
        return math.sqrt(sum(x * x for x in self.data))
    
    def cosine_similarity(self, other: 'Tensor') -> float:
        """Compute cosine similarity with another tensor."""
        dot_product = self.dot(other)
        norm_product = self.norm() * other.norm()
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product
    
    def add(self, other: 'Tensor') -> 'Tensor':
        """Element-wise addition."""
        if self.shape != other.shape:
            raise ValueError("Tensor shapes don't match")
        
        return Tensor(
            shape=self.shape,
            data=[a + b for a, b in zip(self.data, other.data)],
            tensor_type=self.tensor_type,
        )
    
    def scale(self, scalar: float) -> 'Tensor':
        """Scale tensor by a scalar value."""
        return Tensor(
            shape=self.shape,
            data=[x * scalar for x in self.data],
            tensor_type=self.tensor_type,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tensor to dictionary."""
        return {
            "shape": list(self.shape),
            "data": self.data,
            "tensor_type": self.tensor_type.value,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tensor':
        """Create tensor from dictionary."""
        return cls(
            shape=tuple(data["shape"]),
            data=data["data"],
            tensor_type=TensorType(data["tensor_type"]),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], tensor_type: TensorType = TensorType.EMBEDDING) -> 'Tensor':
        """Create a zero tensor."""
        size = 1
        for dim in shape:
            size *= dim
        return cls(shape=shape, data=[0.0] * size, tensor_type=tensor_type)
    
    @classmethod
    def ones(cls, shape: Tuple[int, ...], tensor_type: TensorType = TensorType.EMBEDDING) -> 'Tensor':
        """Create a tensor filled with ones."""
        size = 1
        for dim in shape:
            size *= dim
        return cls(shape=shape, data=[1.0] * size, tensor_type=tensor_type)


@dataclass
class LogicalProposition:
    """Represents a logical proposition in the symbolic reasoning system."""
    predicate: str
    arguments: List[str]
    truth_value: float  # Fuzzy truth value [0, 1]
    confidence: float  # Confidence in the truth value [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicate": self.predicate,
            "arguments": self.arguments,
            "truth_value": self.truth_value,
            "confidence": self.confidence,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        args = ", ".join(self.arguments)
        return f"{self.predicate}({args}) [TV: {self.truth_value:.2f}, Conf: {self.confidence:.2f}]"


class LogicalReasoner:
    """
    Implements symbolic logical reasoning with fuzzy logic support.
    """
    
    def __init__(self):
        self.propositions: Dict[str, LogicalProposition] = {}
        self.rules: List[Tuple[List[str], str]] = []  # (premises, conclusion)
    
    def add_proposition(self, prop: LogicalProposition) -> str:
        """Add a logical proposition."""
        prop_id = f"{prop.predicate}_{len(self.propositions)}"
        self.propositions[prop_id] = prop
        return prop_id
    
    def add_rule(self, premises: List[str], conclusion: str):
        """Add an inference rule."""
        self.rules.append((premises, conclusion))
    
    def fuzzy_and(self, truth_values: List[float]) -> float:
        """Fuzzy AND operation (minimum)."""
        return min(truth_values) if truth_values else 0.0
    
    def fuzzy_or(self, truth_values: List[float]) -> float:
        """Fuzzy OR operation (maximum)."""
        return max(truth_values) if truth_values else 0.0
    
    def fuzzy_not(self, truth_value: float) -> float:
        """Fuzzy NOT operation."""
        return 1.0 - truth_value
    
    def infer(self, prop_id: str) -> Optional[LogicalProposition]:
        """Apply inference rules to derive new propositions."""
        if prop_id not in self.propositions:
            return None
        
        # Simple forward chaining
        for premises, conclusion in self.rules:
            if prop_id in premises:
                # Check if all premises are satisfied
                premise_tvs = []
                for premise in premises:
                    if premise in self.propositions:
                        premise_tvs.append(self.propositions[premise].truth_value)
                
                if len(premise_tvs) == len(premises):
                    # All premises present, apply rule
                    conclusion_tv = self.fuzzy_and(premise_tvs)
                    
                    # Create conclusion proposition
                    premise_propositions = [self.propositions[pid] for pid in premises]
                    conclusion_prop = LogicalProposition(
                        predicate=conclusion,
                        arguments=[],
                        truth_value=conclusion_tv,
                        confidence=min(p.confidence for p in premise_propositions),
                    )
                    
                    return conclusion_prop
        
        return None


class NeuroSymbolicBridge:
    """
    Bridges neural tensor operations with symbolic logical reasoning.
    Implements the core neuro-symbolic integration.
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.logical_reasoner = LogicalReasoner()
        self.atom_embeddings: Dict[str, Tensor] = {}
        self.concept_lattice: Dict[str, List[str]] = {}  # Concept hierarchy
    
    def embed_atom(self, atom: Atom) -> Tensor:
        """
        Create a neural embedding for a symbolic atom.
        Uses a simple hash-based embedding for now.
        """
        if atom.atom_id in self.atom_embeddings:
            return self.atom_embeddings[atom.atom_id]
        
        # Simple embedding based on atom properties
        embedding = []
        
        # Encode atom type (first 10 dimensions)
        type_encoding = [0.0] * 10
        type_idx = list(AtomType).index(atom.atom_type) % 10
        type_encoding[type_idx] = 1.0
        embedding.extend(type_encoding)
        
        # Encode truth value (10 dimensions)
        tv = atom.truth_value
        embedding.extend([tv] * 10)
        
        # Encode attention value (10 dimensions)
        av = atom.attention_value
        embedding.extend([av] * 10)
        
        # Hash-based features for name (remaining dimensions)
        name_hash = hash(atom.name)
        remaining_dims = self.embedding_dim - len(embedding)
        for i in range(remaining_dims):
            val = math.sin(name_hash * (i + 1) * 0.01)
            embedding.append(val)
        
        tensor = Tensor(
            shape=(self.embedding_dim,),
            data=embedding,
            tensor_type=TensorType.EMBEDDING,
            metadata={"atom_id": atom.atom_id, "atom_name": atom.name},
        )
        
        self.atom_embeddings[atom.atom_id] = tensor
        return tensor
    
    def tensor_to_logic(self, tensor: Tensor, threshold: float = 0.5) -> LogicalProposition:
        """
        Convert a neural tensor representation to a symbolic logical proposition.
        """
        # Extract truth value from tensor activation
        avg_activation = sum(tensor.data) / len(tensor.data)
        truth_value = max(0.0, min(1.0, avg_activation))
        
        # Extract confidence from variance
        variance = sum((x - avg_activation) ** 2 for x in tensor.data) / len(tensor.data)
        confidence = 1.0 - min(1.0, variance)
        
        atom_id = tensor.metadata.get("atom_id", "unknown")
        
        return LogicalProposition(
            predicate="activated",
            arguments=[atom_id],
            truth_value=truth_value,
            confidence=confidence,
        )
    
    def logic_to_tensor(self, prop: LogicalProposition) -> Tensor:
        """
        Convert a symbolic logical proposition to a neural tensor.
        """
        # Create a tensor encoding the proposition
        data = [prop.truth_value] * (self.embedding_dim // 2)
        data.extend([prop.confidence] * (self.embedding_dim - len(data)))
        
        return Tensor(
            shape=(self.embedding_dim,),
            data=data,
            tensor_type=TensorType.LOGIC_STATE,
            metadata={"predicate": prop.predicate, "arguments": prop.arguments},
        )
    
    def compute_semantic_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """
        Compute semantic similarity between two atoms using their embeddings.
        """
        emb1 = self.embed_atom(atom1)
        emb2 = self.embed_atom(atom2)
        
        return emb1.cosine_similarity(emb2)
    
    def neural_inference(self, atoms: List[Atom]) -> List[Tuple[Atom, float]]:
        """
        Perform neural inference over a set of atoms.
        Returns atoms ranked by relevance score.
        """
        if not atoms:
            return []
        
        # Compute centroid of atom embeddings
        embeddings = [self.embed_atom(atom) for atom in atoms]
        centroid_data = [0.0] * self.embedding_dim
        
        for emb in embeddings:
            for i, val in enumerate(emb.data):
                centroid_data[i] += val / len(embeddings)
        
        centroid = Tensor(
            shape=(self.embedding_dim,),
            data=centroid_data,
            tensor_type=TensorType.EMBEDDING,
        )
        
        # Rank atoms by similarity to centroid
        ranked = []
        for atom in atoms:
            emb = self.embed_atom(atom)
            similarity = emb.cosine_similarity(centroid)
            ranked.append((atom, similarity))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class TensorLogicOrchestrator:
    """
    Orchestrates the integration of tensor operations and logical reasoning.
    Provides the main interface for neuro-symbolic processing.
    """
    
    def __init__(self, atomspace: AtomSpace, embedding_dim: int = 128):
        self.atomspace = atomspace
        self.bridge = NeuroSymbolicBridge(embedding_dim=embedding_dim)
        self.reasoning_history: List[Dict[str, Any]] = []
    
    async def neuro_symbolic_query(self, query: str, atoms: List[Atom]) -> Dict[str, Any]:
        """
        Process a query using both neural and symbolic reasoning.
        
        Args:
            query: The query string
            atoms: Relevant atoms from the knowledge base
        
        Returns:
            Combined neuro-symbolic reasoning results
        """
        # Neural phase: Rank atoms by relevance
        neural_results = self.bridge.neural_inference(atoms)
        
        # Symbolic phase: Apply logical reasoning
        symbolic_results = []
        for atom, score in neural_results[:10]:  # Top 10
            # Convert to logical proposition
            tensor = self.bridge.embed_atom(atom)
            prop = self.bridge.tensor_to_logic(tensor)
            
            # Add to logical reasoner
            prop_id = self.bridge.logical_reasoner.add_proposition(prop)
            
            # Try to infer new knowledge
            inferred = self.bridge.logical_reasoner.infer(prop_id)
            
            symbolic_results.append({
                "atom": atom.to_dict(),
                "neural_score": score,
                "proposition": prop.to_dict(),
                "inferred": inferred.to_dict() if inferred else None,
            })
        
        # Record reasoning history
        reasoning_record = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "neural_results_count": len(neural_results),
            "symbolic_results_count": len(symbolic_results),
            "top_result": symbolic_results[0] if symbolic_results else None,
        }
        self.reasoning_history.append(reasoning_record)
        
        return {
            "query": query,
            "neural_rankings": [(a.to_dict(), s) for a, s in neural_results[:5]],
            "symbolic_reasoning": symbolic_results,
            "reasoning_type": "neuro-symbolic",
        }
    
    async def tensor_attention(self, focus_atom: Atom, context_atoms: List[Atom]) -> List[Tuple[Atom, float]]:
        """
        Apply neural attention mechanism to focus on relevant atoms.
        
        Args:
            focus_atom: The atom to focus attention on
            context_atoms: Context atoms to compute attention over
        
        Returns:
            Atoms with attention scores
        """
        focus_embedding = self.bridge.embed_atom(focus_atom)
        
        attention_scores = []
        for atom in context_atoms:
            atom_embedding = self.bridge.embed_atom(atom)
            
            # Compute attention as scaled dot product
            attention = focus_embedding.dot(atom_embedding) / math.sqrt(self.bridge.embedding_dim)
            attention = 1.0 / (1.0 + math.exp(-attention))  # Sigmoid activation
            
            attention_scores.append((atom, attention))
        
        # Normalize attention scores
        total = sum(score for _, score in attention_scores)
        if total > 0:
            attention_scores = [(atom, score / total) for atom, score in attention_scores]
        
        attention_scores.sort(key=lambda x: x[1], reverse=True)
        return attention_scores
    
    async def knowledge_consolidation(self, atoms: List[Atom]) -> Tensor:
        """
        Consolidate multiple atoms into a unified knowledge representation.
        
        Args:
            atoms: Atoms to consolidate
        
        Returns:
            Consolidated knowledge tensor
        """
        if not atoms:
            return Tensor.zeros((self.bridge.embedding_dim,), TensorType.KNOWLEDGE_VECTOR)
        
        # Weighted sum of embeddings based on attention values
        consolidated_data = [0.0] * self.bridge.embedding_dim
        total_attention = sum(atom.attention_value for atom in atoms)
        
        for atom in atoms:
            embedding = self.bridge.embed_atom(atom)
            weight = atom.attention_value / total_attention if total_attention > 0 else 1.0 / len(atoms)
            
            for i, val in enumerate(embedding.data):
                consolidated_data[i] += val * weight
        
        return Tensor(
            shape=(self.bridge.embedding_dim,),
            data=consolidated_data,
            tensor_type=TensorType.KNOWLEDGE_VECTOR,
            metadata={
                "consolidated_atoms": [atom.atom_id for atom in atoms],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get the history of reasoning operations."""
        return self.reasoning_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tensor-logic operations."""
        return {
            "total_embeddings": len(self.bridge.atom_embeddings),
            "embedding_dimension": self.bridge.embedding_dim,
            "reasoning_operations": len(self.reasoning_history),
            "logical_propositions": len(self.bridge.logical_reasoner.propositions),
            "inference_rules": len(self.bridge.logical_reasoner.rules),
        }
