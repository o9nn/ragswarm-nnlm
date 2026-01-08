"""
Cognitive Architecture Framework for RAGSwarm Neural Notebook-LM.

Implements a modular cognitive architecture with perception, reasoning,
memory, and action components inspired by cognitive science principles.
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

from .opencog_atomspace import Atom, AtomType, AtomSpace
from .tensor_logic import TensorLogicOrchestrator, Tensor, LogicalProposition


class CognitiveModule(Enum):
    """Types of cognitive modules in the architecture."""
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTION = "action"
    LEARNING = "learning"
    METACOGNITION = "metacognition"


class MemoryType(Enum):
    """Types of memory in the cognitive system."""
    WORKING = "working"  # Short-term working memory
    EPISODIC = "episodic"  # Memory of specific events
    SEMANTIC = "semantic"  # Factual knowledge
    PROCEDURAL = "procedural"  # Skills and procedures


@dataclass
class CognitiveState:
    """Represents the current state of the cognitive system."""
    current_focus: Optional[str] = None
    active_goals: List[str] = field(default_factory=list)
    working_memory: List[Atom] = field(default_factory=list)
    attention_distribution: Dict[str, float] = field(default_factory=dict)
    arousal_level: float = 0.5  # Overall activation level [0, 1]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "current_focus": self.current_focus,
            "active_goals": self.active_goals,
            "working_memory_size": len(self.working_memory),
            "attention_distribution": self.attention_distribution,
            "arousal_level": self.arousal_level,
            "timestamp": self.timestamp,
        }


@dataclass
class PerceptualInput:
    """Represents input from the perceptual system."""
    modality: str  # e.g., "text", "code", "structure"
    content: Any
    salience: float = 0.5  # How attention-grabbing [0, 1]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveGoal:
    """Represents a goal in the cognitive system."""
    goal_id: str
    description: str
    priority: float  # [0, 1]
    status: str  # "active", "completed", "failed"
    subgoals: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "subgoals": self.subgoals,
            "created_at": self.created_at,
        }


class PerceptionModule:
    """
    Handles perception and preprocessing of external stimuli.
    Converts raw input into internal representations.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.perceptual_buffer: List[PerceptualInput] = []
        self.feature_extractors: Dict[str, Callable] = {}
    
    def register_feature_extractor(self, modality: str, extractor: Callable):
        """Register a feature extractor for a specific modality."""
        self.feature_extractors[modality] = extractor
    
    async def perceive(self, input_data: Any, modality: str = "text") -> PerceptualInput:
        """
        Process raw input and create perceptual representation.
        
        Args:
            input_data: Raw input data
            modality: Type of input (text, code, etc.)
        
        Returns:
            Processed perceptual input
        """
        # Extract features if extractor available
        if modality in self.feature_extractors:
            features = self.feature_extractors[modality](input_data)
        else:
            features = {"raw": str(input_data)}
        
        # Compute salience based on features
        salience = self._compute_salience(features)
        
        perception = PerceptualInput(
            modality=modality,
            content=input_data,
            salience=salience,
            metadata={"features": features},
        )
        
        self.perceptual_buffer.append(perception)
        
        # Convert to atom for storage
        atom = Atom(
            atom_type=AtomType.NODE,
            name=f"perception_{len(self.perceptual_buffer)}",
            attention_value=salience,
            metadata={
                "modality": modality,
                "features": features,
            },
        )
        
        await self.atomspace.add_atom(atom)
        
        return perception
    
    def _compute_salience(self, features: Dict[str, Any]) -> float:
        """Compute how salient/attention-grabbing the input is."""
        # Simple heuristic: longer content is more salient
        if "raw" in features:
            content_length = len(str(features["raw"]))
            return min(1.0, content_length / 1000.0)
        return 0.5


class AttentionModule:
    """
    Manages attention allocation across cognitive processes.
    Determines what the system should focus on.
    """
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity  # Working memory capacity
        self.focus_items: List[str] = []
        self.attention_weights: Dict[str, float] = {}
    
    def allocate_attention(self, items: List[Atom], relevance_scores: Dict[str, float]):
        """
        Allocate attention to items based on relevance.
        
        Args:
            items: Candidate items for attention
            relevance_scores: Relevance score for each item
        """
        # Sort by relevance and attention value
        scored_items = [
            (atom, relevance_scores.get(atom.atom_id, 0.5) * atom.attention_value)
            for atom in items
        ]
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Select top items within capacity
        self.focus_items = [atom.atom_id for atom, _ in scored_items[:self.capacity]]
        
        # Compute attention weights
        total_score = sum(score for _, score in scored_items[:self.capacity])
        self.attention_weights = {
            atom.atom_id: score / total_score if total_score > 0 else 1.0 / self.capacity
            for atom, score in scored_items[:self.capacity]
        }
    
    def get_focus(self) -> List[str]:
        """Get current focus items."""
        return self.focus_items
    
    def shift_attention(self, new_focus: str):
        """Shift attention to a new item."""
        if new_focus not in self.focus_items:
            if len(self.focus_items) >= self.capacity:
                # Remove least attended item
                min_item = min(self.attention_weights.items(), key=lambda x: x[1])[0]
                self.focus_items.remove(min_item)
                del self.attention_weights[min_item]
            
            self.focus_items.append(new_focus)
            self.attention_weights[new_focus] = 1.0


class MemoryModule:
    """
    Manages different types of memory in the cognitive system.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.memory_stores: Dict[MemoryType, List[Atom]] = {
            MemoryType.WORKING: [],
            MemoryType.EPISODIC: [],
            MemoryType.SEMANTIC: [],
            MemoryType.PROCEDURAL: [],
        }
        self.working_memory_capacity = 7
    
    async def store(self, atom: Atom, memory_type: MemoryType):
        """Store an atom in the specified memory type."""
        self.memory_stores[memory_type].append(atom)
        
        # Manage working memory capacity
        if memory_type == MemoryType.WORKING:
            if len(self.memory_stores[MemoryType.WORKING]) > self.working_memory_capacity:
                # Remove oldest/least relevant item
                oldest = min(
                    self.memory_stores[MemoryType.WORKING],
                    key=lambda a: a.attention_value
                )
                self.memory_stores[MemoryType.WORKING].remove(oldest)
    
    async def retrieve(self, query: str, memory_type: Optional[MemoryType] = None) -> List[Atom]:
        """
        Retrieve memories matching the query.
        
        Args:
            query: Search query
            memory_type: Specific memory type to search, or None for all
        
        Returns:
            Matching atoms
        """
        if memory_type:
            stores_to_search = [memory_type]
        else:
            stores_to_search = list(MemoryType)
        
        results = []
        for mem_type in stores_to_search:
            for atom in self.memory_stores[mem_type]:
                if query.lower() in atom.name.lower():
                    results.append(atom)
        
        return results
    
    async def consolidate(self):
        """
        Consolidate working memory into long-term memory.
        Moves important items from working to semantic/episodic memory.
        """
        for atom in self.memory_stores[MemoryType.WORKING]:
            if atom.attention_value > 0.7:
                # High attention -> semantic memory
                if atom not in self.memory_stores[MemoryType.SEMANTIC]:
                    self.memory_stores[MemoryType.SEMANTIC].append(atom)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            mem_type.value: len(atoms)
            for mem_type, atoms in self.memory_stores.items()
        }


class ReasoningModule:
    """
    Handles reasoning and inference using neuro-symbolic methods.
    """
    
    def __init__(self, orchestrator: TensorLogicOrchestrator):
        self.orchestrator = orchestrator
        self.inference_chains: List[Dict[str, Any]] = []
    
    async def reason(self, query: str, context_atoms: List[Atom]) -> Dict[str, Any]:
        """
        Perform reasoning over context to answer query.
        
        Args:
            query: Query to reason about
            context_atoms: Context for reasoning
        
        Returns:
            Reasoning results
        """
        # Use neuro-symbolic reasoning
        result = await self.orchestrator.neuro_symbolic_query(query, context_atoms)
        
        # Record inference chain
        self.inference_chains.append({
            "query": query,
            "context_size": len(context_atoms),
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return result
    
    async def analogical_reasoning(self, source: Atom, target_domain: List[Atom]) -> Optional[Atom]:
        """
        Perform analogical reasoning: find similar structures in target domain.
        
        Args:
            source: Source atom to find analogy for
            target_domain: Target atoms to search
        
        Returns:
            Most analogous atom if found
        """
        if not target_domain:
            return None
        
        # Use semantic similarity from neuro-symbolic bridge
        similarities = []
        for atom in target_domain:
            similarity = self.orchestrator.bridge.compute_semantic_similarity(source, atom)
            similarities.append((atom, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[0][0] if similarities else None


class PlanningModule:
    """
    Handles goal decomposition and planning.
    """
    
    def __init__(self):
        self.goals: Dict[str, CognitiveGoal] = {}
        self.plans: Dict[str, List[str]] = {}  # goal_id -> list of actions
    
    def create_goal(self, description: str, priority: float = 0.5) -> CognitiveGoal:
        """Create a new goal."""
        goal_id = f"goal_{len(self.goals)}"
        goal = CognitiveGoal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            status="active",
        )
        self.goals[goal_id] = goal
        return goal
    
    async def decompose_goal(self, goal_id: str) -> List[str]:
        """
        Decompose a high-level goal into subgoals.
        
        Args:
            goal_id: ID of goal to decompose
        
        Returns:
            List of subgoal IDs
        """
        if goal_id not in self.goals:
            return []
        
        goal = self.goals[goal_id]
        
        # Simple decomposition based on keywords
        subgoals = []
        if "analyze" in goal.description.lower():
            subgoals.extend([
                self.create_goal("Extract key concepts", priority=goal.priority).goal_id,
                self.create_goal("Identify relationships", priority=goal.priority).goal_id,
                self.create_goal("Synthesize findings", priority=goal.priority).goal_id,
            ])
        
        goal.subgoals = subgoals
        return subgoals
    
    def update_goal_status(self, goal_id: str, status: str):
        """Update the status of a goal."""
        if goal_id in self.goals:
            self.goals[goal_id].status = status
    
    def get_active_goals(self) -> List[CognitiveGoal]:
        """Get all active goals sorted by priority."""
        active = [g for g in self.goals.values() if g.status == "active"]
        active.sort(key=lambda g: g.priority, reverse=True)
        return active


class CognitiveArchitecture:
    """
    Main cognitive architecture integrating all modules.
    Provides a unified interface for cognitive processing.
    """
    
    def __init__(self, atomspace: AtomSpace, embedding_dim: int = 128):
        self.atomspace = atomspace
        
        # Initialize orchestrator
        self.orchestrator = TensorLogicOrchestrator(atomspace, embedding_dim)
        
        # Initialize modules
        self.perception = PerceptionModule(atomspace)
        self.attention = AttentionModule()
        self.memory = MemoryModule(atomspace)
        self.reasoning = ReasoningModule(self.orchestrator)
        self.planning = PlanningModule()
        
        # Current state
        self.state = CognitiveState()
        
        # Processing history
        self.processing_history: List[Dict[str, Any]] = []
    
    async def perceive_and_process(self, input_data: Any, modality: str = "text") -> Dict[str, Any]:
        """
        Main cognitive processing loop.
        
        Args:
            input_data: Input to process
            modality: Type of input
        
        Returns:
            Processing results
        """
        # 1. Perception
        perception = await self.perception.perceive(input_data, modality)
        
        # 2. Attention allocation
        relevant_atoms = await self.atomspace.find_atoms_by_name(str(input_data)[:100])
        relevance_scores = {atom.atom_id: atom.attention_value for atom in relevant_atoms}
        self.attention.allocate_attention(relevant_atoms, relevance_scores)
        
        # 3. Update working memory
        focus_atoms = [
            atom for atom in relevant_atoms 
            if atom.atom_id in self.attention.get_focus()
        ]
        
        for atom in focus_atoms:
            await self.memory.store(atom, MemoryType.WORKING)
        
        # 4. Reasoning
        reasoning_result = await self.reasoning.reason(
            str(input_data), 
            self.memory.memory_stores[MemoryType.WORKING]
        )
        
        # 5. Update state
        self.state.current_focus = input_data if len(str(input_data)) < 100 else str(input_data)[:100]
        self.state.working_memory = self.memory.memory_stores[MemoryType.WORKING]
        self.state.attention_distribution = self.attention.attention_weights
        self.state.timestamp = datetime.utcnow().isoformat()
        
        # Record processing
        record = {
            "input": str(input_data)[:200],
            "modality": modality,
            "perception_salience": perception.salience,
            "attention_focus": len(self.attention.get_focus()),
            "reasoning_result": reasoning_result,
            "state": self.state.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.processing_history.append(record)
        
        return record
    
    async def goal_directed_processing(self, goal_description: str) -> Dict[str, Any]:
        """
        Process with a specific goal in mind.
        
        Args:
            goal_description: Description of the goal
        
        Returns:
            Results of goal-directed processing
        """
        # Create goal
        goal = self.planning.create_goal(goal_description, priority=0.8)
        self.state.active_goals.append(goal.goal_id)
        
        # Decompose into subgoals
        subgoals = await self.planning.decompose_goal(goal.goal_id)
        
        # Process each subgoal
        results = []
        for subgoal_id in subgoals:
            subgoal = self.planning.goals[subgoal_id]
            
            # Retrieve relevant knowledge
            relevant = await self.memory.retrieve(subgoal.description)
            
            # Reason about subgoal
            reasoning = await self.reasoning.reason(subgoal.description, relevant)
            
            results.append({
                "subgoal": subgoal.to_dict(),
                "reasoning": reasoning,
            })
            
            # Mark subgoal as completed
            self.planning.update_goal_status(subgoal_id, "completed")
        
        # Mark main goal as completed
        self.planning.update_goal_status(goal.goal_id, "completed")
        
        return {
            "goal": goal.to_dict(),
            "subgoal_results": results,
            "completed": True,
        }
    
    async def metacognitive_reflection(self) -> Dict[str, Any]:
        """
        Reflect on own cognitive processes (metacognition).
        
        Returns:
            Analysis of cognitive performance
        """
        # Analyze processing history
        if not self.processing_history:
            return {"message": "No processing history available"}
        
        avg_salience = sum(
            r.get("perception_salience", 0) for r in self.processing_history
        ) / len(self.processing_history)
        
        # Analyze reasoning effectiveness
        reasoning_stats = self.orchestrator.get_statistics()
        
        # Memory statistics
        memory_stats = self.memory.get_statistics()
        
        # Goal completion rate
        total_goals = len(self.planning.goals)
        completed_goals = sum(
            1 for g in self.planning.goals.values() if g.status == "completed"
        )
        completion_rate = completed_goals / total_goals if total_goals > 0 else 0
        
        return {
            "processing_cycles": len(self.processing_history),
            "average_salience": avg_salience,
            "reasoning_statistics": reasoning_stats,
            "memory_statistics": memory_stats,
            "goal_completion_rate": completion_rate,
            "current_state": self.state.to_dict(),
            "recommendations": self._generate_recommendations(
                avg_salience, completion_rate
            ),
        }
    
    def _generate_recommendations(self, avg_salience: float, completion_rate: float) -> List[str]:
        """Generate recommendations for improving cognitive performance."""
        recommendations = []
        
        if avg_salience < 0.3:
            recommendations.append(
                "Low salience detected - consider processing more relevant inputs"
            )
        
        if completion_rate < 0.5:
            recommendations.append(
                "Low goal completion rate - consider simplifying goals or allocating more resources"
            )
        
        if len(self.state.working_memory) > 10:
            recommendations.append(
                "Working memory overload - consider memory consolidation"
            )
        
        if not recommendations:
            recommendations.append("Cognitive performance is optimal")
        
        return recommendations
    
    def get_state(self) -> CognitiveState:
        """Get current cognitive state."""
        return self.state
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get history of cognitive processing."""
        return self.processing_history
