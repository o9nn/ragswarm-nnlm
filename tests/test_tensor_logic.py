"""
Tests for tensor-logic orchestration module.
"""

import pytest
from notebookllama.tensor_logic import (
    Tensor, TensorType, LogicalProposition, LogicalReasoner,
    NeuroSymbolicBridge, TensorLogicOrchestrator
)
from notebookllama.opencog_atomspace import Atom, AtomType, AtomSpace


def test_tensor_creation():
    """Test tensor creation and validation."""
    tensor = Tensor(
        shape=(3,),
        data=[1.0, 2.0, 3.0],
        tensor_type=TensorType.EMBEDDING
    )
    
    assert tensor.shape == (3,)
    assert tensor.data == [1.0, 2.0, 3.0]
    assert tensor.tensor_type == TensorType.EMBEDDING


def test_tensor_invalid_shape():
    """Test tensor with invalid shape/data mismatch."""
    with pytest.raises(ValueError):
        Tensor(
            shape=(3,),
            data=[1.0, 2.0],  # Wrong size
            tensor_type=TensorType.EMBEDDING
        )


def test_tensor_dot_product():
    """Test tensor dot product."""
    t1 = Tensor(shape=(3,), data=[1.0, 2.0, 3.0], tensor_type=TensorType.EMBEDDING)
    t2 = Tensor(shape=(3,), data=[2.0, 3.0, 4.0], tensor_type=TensorType.EMBEDDING)
    
    dot = t1.dot(t2)
    assert dot == 20.0  # 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20


def test_tensor_norm():
    """Test tensor L2 norm."""
    tensor = Tensor(shape=(3,), data=[3.0, 4.0, 0.0], tensor_type=TensorType.EMBEDDING)
    norm = tensor.norm()
    assert norm == 5.0  # sqrt(9 + 16) = 5


def test_tensor_cosine_similarity():
    """Test cosine similarity between tensors."""
    t1 = Tensor(shape=(2,), data=[1.0, 0.0], tensor_type=TensorType.EMBEDDING)
    t2 = Tensor(shape=(2,), data=[0.0, 1.0], tensor_type=TensorType.EMBEDDING)
    
    similarity = t1.cosine_similarity(t2)
    assert similarity == 0.0  # Orthogonal vectors


def test_tensor_operations():
    """Test tensor addition and scaling."""
    t1 = Tensor(shape=(2,), data=[1.0, 2.0], tensor_type=TensorType.EMBEDDING)
    t2 = Tensor(shape=(2,), data=[3.0, 4.0], tensor_type=TensorType.EMBEDDING)
    
    # Addition
    t3 = t1.add(t2)
    assert t3.data == [4.0, 6.0]
    
    # Scaling
    t4 = t1.scale(2.0)
    assert t4.data == [2.0, 4.0]


def test_tensor_serialization():
    """Test tensor to/from dict."""
    tensor = Tensor(
        shape=(2,), 
        data=[1.0, 2.0], 
        tensor_type=TensorType.EMBEDDING,
        metadata={"test": "value"}
    )
    
    d = tensor.to_dict()
    assert d["shape"] == [2]
    assert d["data"] == [1.0, 2.0]
    assert d["metadata"]["test"] == "value"
    
    # Reconstruct
    tensor2 = Tensor.from_dict(d)
    assert tensor2.shape == tensor.shape
    assert tensor2.data == tensor.data


def test_logical_proposition():
    """Test logical proposition creation."""
    prop = LogicalProposition(
        predicate="is_concept",
        arguments=["machine_learning"],
        truth_value=0.9,
        confidence=0.85
    )
    
    assert prop.predicate == "is_concept"
    assert prop.truth_value == 0.9
    assert prop.confidence == 0.85


def test_logical_reasoner():
    """Test logical reasoning operations."""
    reasoner = LogicalReasoner()
    
    # Add propositions
    prop1 = LogicalProposition(
        predicate="is_animal",
        arguments=["dog"],
        truth_value=1.0,
        confidence=1.0
    )
    
    prop_id = reasoner.add_proposition(prop1)
    assert prop_id in reasoner.propositions


def test_fuzzy_logic_operations():
    """Test fuzzy logic operations."""
    reasoner = LogicalReasoner()
    
    # Test fuzzy AND (minimum)
    assert reasoner.fuzzy_and([0.7, 0.8, 0.9]) == 0.7
    
    # Test fuzzy OR (maximum)
    assert reasoner.fuzzy_or([0.7, 0.8, 0.9]) == 0.9
    
    # Test fuzzy NOT
    assert reasoner.fuzzy_not(0.3) == 0.7


@pytest.mark.asyncio
async def test_neurosymbolic_bridge():
    """Test neuro-symbolic bridge."""
    atomspace = AtomSpace()
    bridge = NeuroSymbolicBridge(embedding_dim=64)
    
    # Create test atom
    atom = Atom(
        atom_type=AtomType.CONCEPT,
        name="test_concept",
        truth_value=0.8
    )
    await atomspace.add_atom(atom)
    
    # Test embedding
    embedding = bridge.embed_atom(atom)
    assert embedding.shape == (64,)
    assert len(embedding.data) == 64


@pytest.mark.asyncio
async def test_tensor_to_logic_conversion():
    """Test conversion between tensor and logic representations."""
    bridge = NeuroSymbolicBridge(embedding_dim=32)
    
    # Create a tensor
    tensor = Tensor(
        shape=(32,),
        data=[0.5] * 32,
        tensor_type=TensorType.ACTIVATION,
        metadata={"atom_id": "test_123"}
    )
    
    # Convert to logical proposition
    prop = bridge.tensor_to_logic(tensor)
    assert isinstance(prop, LogicalProposition)
    assert 0.0 <= prop.truth_value <= 1.0
    assert 0.0 <= prop.confidence <= 1.0


@pytest.mark.asyncio
async def test_logic_to_tensor_conversion():
    """Test conversion from logic to tensor."""
    bridge = NeuroSymbolicBridge(embedding_dim=32)
    
    # Create a logical proposition
    prop = LogicalProposition(
        predicate="test",
        arguments=["arg1"],
        truth_value=0.9,
        confidence=0.85
    )
    
    # Convert to tensor
    tensor = bridge.logic_to_tensor(prop)
    assert tensor.shape == (32,)
    assert tensor.tensor_type == TensorType.LOGIC_STATE


@pytest.mark.asyncio
async def test_semantic_similarity():
    """Test semantic similarity computation."""
    atomspace = AtomSpace()
    bridge = NeuroSymbolicBridge(embedding_dim=64)
    
    # Create test atoms
    atom1 = Atom(atom_type=AtomType.CONCEPT, name="concept1")
    atom2 = Atom(atom_type=AtomType.CONCEPT, name="concept2")
    
    await atomspace.add_atom(atom1)
    await atomspace.add_atom(atom2)
    
    # Compute similarity
    similarity = bridge.compute_semantic_similarity(atom1, atom2)
    assert -1.0 <= similarity <= 1.0


@pytest.mark.asyncio
async def test_neural_inference():
    """Test neural inference over atoms."""
    atomspace = AtomSpace()
    bridge = NeuroSymbolicBridge(embedding_dim=64)
    
    # Create test atoms
    atoms = []
    for i in range(5):
        atom = Atom(
            atom_type=AtomType.CONCEPT,
            name=f"concept_{i}",
            attention_value=0.5 + i * 0.1
        )
        await atomspace.add_atom(atom)
        atoms.append(atom)
    
    # Perform neural inference
    ranked = bridge.neural_inference(atoms)
    
    assert len(ranked) == 5
    # Results should be ranked
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i+1][1]


@pytest.mark.asyncio
async def test_tensor_logic_orchestrator():
    """Test tensor-logic orchestrator."""
    atomspace = AtomSpace()
    orchestrator = TensorLogicOrchestrator(atomspace, embedding_dim=64)
    
    # Create test atoms
    atoms = []
    for i in range(3):
        atom = Atom(
            atom_type=AtomType.CONCEPT,
            name=f"test_concept_{i}",
            truth_value=0.8
        )
        await atomspace.add_atom(atom)
        atoms.append(atom)
    
    # Test neuro-symbolic query
    result = await orchestrator.neuro_symbolic_query("test query", atoms)
    
    assert "query" in result
    assert "neural_rankings" in result
    assert "symbolic_reasoning" in result
    assert result["reasoning_type"] == "neuro-symbolic"


@pytest.mark.asyncio
async def test_tensor_attention():
    """Test tensor attention mechanism."""
    atomspace = AtomSpace()
    orchestrator = TensorLogicOrchestrator(atomspace, embedding_dim=32)
    
    # Create focus and context atoms
    focus = Atom(atom_type=AtomType.CONCEPT, name="focus")
    await atomspace.add_atom(focus)
    
    context = []
    for i in range(5):
        atom = Atom(atom_type=AtomType.CONCEPT, name=f"context_{i}")
        await atomspace.add_atom(atom)
        context.append(atom)
    
    # Apply attention
    attention_scores = await orchestrator.tensor_attention(focus, context)
    
    assert len(attention_scores) == 5
    # Scores should sum to approximately 1.0 (normalized)
    total = sum(score for _, score in attention_scores)
    assert 0.99 <= total <= 1.01


@pytest.mark.asyncio
async def test_knowledge_consolidation():
    """Test knowledge consolidation."""
    atomspace = AtomSpace()
    orchestrator = TensorLogicOrchestrator(atomspace, embedding_dim=32)
    
    # Create atoms with varying attention
    atoms = []
    for i in range(5):
        atom = Atom(
            atom_type=AtomType.CONCEPT,
            name=f"concept_{i}",
            attention_value=0.2 + i * 0.2
        )
        await atomspace.add_atom(atom)
        atoms.append(atom)
    
    # Consolidate knowledge
    consolidated = await orchestrator.knowledge_consolidation(atoms)
    
    assert consolidated.shape == (32,)
    assert consolidated.tensor_type == TensorType.KNOWLEDGE_VECTOR


@pytest.mark.asyncio
async def test_orchestrator_statistics():
    """Test getting statistics from orchestrator."""
    atomspace = AtomSpace()
    orchestrator = TensorLogicOrchestrator(atomspace, embedding_dim=64)
    
    # Add some data
    atom = Atom(atom_type=AtomType.CONCEPT, name="test")
    await atomspace.add_atom(atom)
    
    orchestrator.bridge.embed_atom(atom)
    
    # Get statistics
    stats = orchestrator.get_statistics()
    
    assert "total_embeddings" in stats
    assert "embedding_dimension" in stats
    assert stats["embedding_dimension"] == 64
    assert stats["total_embeddings"] >= 1


@pytest.mark.asyncio
async def test_reasoning_history():
    """Test reasoning history tracking."""
    atomspace = AtomSpace()
    orchestrator = TensorLogicOrchestrator(atomspace, embedding_dim=32)
    
    # Create test atoms
    atoms = [
        Atom(atom_type=AtomType.CONCEPT, name="concept1"),
        Atom(atom_type=AtomType.CONCEPT, name="concept2")
    ]
    for atom in atoms:
        await atomspace.add_atom(atom)
    
    # Perform multiple queries
    await orchestrator.neuro_symbolic_query("query1", atoms)
    await orchestrator.neuro_symbolic_query("query2", atoms)
    
    # Check history
    history = orchestrator.get_reasoning_history()
    assert len(history) == 2
    assert history[0]["query"] == "query1"
    assert history[1]["query"] == "query2"
