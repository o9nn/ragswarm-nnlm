"""
Neuro-Symbolic Fabric Integration for RAGSwarm Neural Notebook-LM.

Integrates tensor-logic orchestration, cognitive architecture, neural notebook generation,
and knowledge constellations into a unified neuro-symbolic framework.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from .opencog_atomspace import AtomSpace, DistributedAtomSpace, Atom, AtomType
from .tensor_logic import TensorLogicOrchestrator
from .cognitive_architecture import CognitiveArchitecture
from .neural_notebook_generator import NotebookOrchestrator
from .knowledge_constellations import KnowledgeConstellationOrchestrator


@dataclass
class NeuroSymbolicConfig:
    """Configuration for the neuro-symbolic fabric."""
    embedding_dim: int = 128
    node_id: str = "primary"
    enable_tensor_logic: bool = True
    enable_cognitive_arch: bool = True
    enable_constellations: bool = True
    enable_notebook_gen: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedding_dim": self.embedding_dim,
            "node_id": self.node_id,
            "enable_tensor_logic": self.enable_tensor_logic,
            "enable_cognitive_arch": self.enable_cognitive_arch,
            "enable_constellations": self.enable_constellations,
            "enable_notebook_gen": self.enable_notebook_gen,
        }


class NeuroSymbolicFabric:
    """
    Main integration layer for the neuro-symbolic framework.
    Provides unified interface to all neuro-symbolic capabilities.
    """
    
    def __init__(self, config: Optional[NeuroSymbolicConfig] = None):
        self.config = config or NeuroSymbolicConfig()
        
        # Initialize core components
        self.distributed_atomspace = DistributedAtomSpace(node_id=self.config.node_id)
        self.atomspace = self.distributed_atomspace.local_space
        
        # Initialize specialized modules
        self.tensor_logic: Optional[TensorLogicOrchestrator] = None
        self.cognitive_arch: Optional[CognitiveArchitecture] = None
        self.notebook_orchestrator: Optional[NotebookOrchestrator] = None
        self.constellation_orchestrator: Optional[KnowledgeConstellationOrchestrator] = None
        
        self._initialized = False
        self.processing_history: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialize all enabled modules."""
        if self._initialized:
            return
        
        # Initialize tensor-logic orchestration
        if self.config.enable_tensor_logic:
            self.tensor_logic = TensorLogicOrchestrator(
                self.atomspace, 
                embedding_dim=self.config.embedding_dim
            )
        
        # Initialize cognitive architecture
        if self.config.enable_cognitive_arch:
            self.cognitive_arch = CognitiveArchitecture(
                self.atomspace,
                embedding_dim=self.config.embedding_dim
            )
        
        # Initialize notebook orchestrator
        if self.config.enable_notebook_gen:
            self.notebook_orchestrator = NotebookOrchestrator(
                self.atomspace,
                embedding_dim=self.config.embedding_dim
            )
        
        # Initialize constellation orchestrator
        if self.config.enable_constellations:
            self.constellation_orchestrator = KnowledgeConstellationOrchestrator(
                self.atomspace
            )
        
        self._initialized = True
    
    async def process_with_neuro_symbolic_reasoning(
        self, 
        query: str, 
        context_atoms: Optional[List[Atom]] = None
    ) -> Dict[str, Any]:
        """
        Process a query using full neuro-symbolic reasoning pipeline.
        
        Args:
            query: Query to process
            context_atoms: Optional context atoms
        
        Returns:
            Comprehensive reasoning results
        """
        if not self._initialized:
            await self.initialize()
        
        result = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }
        
        # Get context if not provided
        if context_atoms is None:
            context_atoms = await self.atomspace.find_atoms_by_name(query)
        
        # Tensor-logic reasoning
        if self.tensor_logic:
            tensor_result = await self.tensor_logic.neuro_symbolic_query(
                query, 
                context_atoms
            )
            result["components"]["tensor_logic"] = tensor_result
        
        # Cognitive processing
        if self.cognitive_arch:
            cognitive_result = await self.cognitive_arch.perceive_and_process(
                query, 
                modality="text"
            )
            result["components"]["cognitive"] = cognitive_result
        
        # Constellation analysis
        if self.constellation_orchestrator and context_atoms:
            await self.constellation_orchestrator.build_hypergraph_from_atoms(
                context_atoms
            )
            constellations = await self.constellation_orchestrator.discover_constellations()
            result["components"]["constellations"] = {
                ctype: [c.to_dict() for c in clist]
                for ctype, clist in constellations.items()
            }
        
        # Record processing
        self.processing_history.append(result)
        
        return result
    
    async def generate_neural_notebook(
        self, 
        repo_path: str, 
        focus: Optional[str] = None
    ) -> Any:
        """
        Generate a neural notebook from repository analysis.
        
        Args:
            repo_path: Path to repository
            focus: Optional focus area
        
        Returns:
            Generated neural notebook
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.notebook_orchestrator:
            raise RuntimeError("Notebook generation not enabled")
        
        notebook = await self.notebook_orchestrator.generate_from_repository(
            repo_path, 
            focus
        )
        
        return notebook
    
    async def add_knowledge(
        self, 
        content: str, 
        atom_type: AtomType = AtomType.CONCEPT,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add knowledge to the neuro-symbolic fabric.
        
        Args:
            content: Content to add
            atom_type: Type of atom to create
            metadata: Optional metadata
        
        Returns:
            Atom ID
        """
        if not self._initialized:
            await self.initialize()
        
        # Create atom
        atom = Atom(
            atom_type=atom_type,
            name=content[:200],  # Limit name length
            metadata=metadata or {"full_content": content},
        )
        
        atom_id = await self.atomspace.add_atom(atom)
        
        # Distribute to network
        await self.distributed_atomspace.broadcast_atom(atom)
        
        # Update hypergraph if constellations enabled
        if self.constellation_orchestrator:
            await self.constellation_orchestrator.build_hypergraph_from_atoms([atom])
        
        return atom_id
    
    async def cognitive_goal_processing(
        self, 
        goal_description: str
    ) -> Dict[str, Any]:
        """
        Process a goal using cognitive architecture.
        
        Args:
            goal_description: Description of the goal
        
        Returns:
            Goal processing results
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.cognitive_arch:
            raise RuntimeError("Cognitive architecture not enabled")
        
        return await self.cognitive_arch.goal_directed_processing(goal_description)
    
    async def discover_knowledge_patterns(self) -> Dict[str, Any]:
        """
        Discover emergent patterns in the knowledge base.
        
        Returns:
            Discovered constellation patterns
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.constellation_orchestrator:
            raise RuntimeError("Constellation detection not enabled")
        
        # Get all atoms
        atoms = list(self.atomspace.atoms.values())
        
        # Build hypergraph
        await self.constellation_orchestrator.build_hypergraph_from_atoms(atoms)
        
        # Discover constellations
        constellations = await self.constellation_orchestrator.discover_constellations()
        
        return {
            "constellations": {
                ctype: [c.to_dict() for c in clist]
                for ctype, clist in constellations.items()
            },
            "statistics": self.constellation_orchestrator.get_statistics(),
        }
    
    async def tensor_attention_analysis(
        self, 
        focus_concept: str, 
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform tensor-based attention analysis.
        
        Args:
            focus_concept: Concept to focus on
            context: Optional context
        
        Returns:
            Attention analysis results
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.tensor_logic:
            raise RuntimeError("Tensor-logic not enabled")
        
        # Find focus atom
        focus_atoms = await self.atomspace.find_atoms_by_name(focus_concept)
        if not focus_atoms:
            return {"error": f"Concept not found: {focus_concept}"}
        
        focus_atom = focus_atoms[0]
        
        # Get context atoms
        if context:
            context_atoms = await self.atomspace.find_atoms_by_name(context)
        else:
            # Use activation spreading
            context_atoms = await self.atomspace.spread_activation(
                focus_atom.atom_id, 
                depth=2
            )
        
        # Apply tensor attention
        attention_results = await self.tensor_logic.tensor_attention(
            focus_atom, 
            context_atoms
        )
        
        return {
            "focus": focus_atom.to_dict(),
            "attention_distribution": [
                (atom.to_dict(), score) 
                for atom, score in attention_results[:10]
            ],
        }
    
    async def metacognitive_analysis(self) -> Dict[str, Any]:
        """
        Perform metacognitive analysis of the system's own processing.
        
        Returns:
            Metacognitive insights
        """
        if not self._initialized:
            await self.initialize()
        
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }
        
        # Cognitive architecture metacognition
        if self.cognitive_arch:
            cognitive_reflection = await self.cognitive_arch.metacognitive_reflection()
            analysis["components"]["cognitive"] = cognitive_reflection
        
        # Tensor-logic statistics
        if self.tensor_logic:
            tensor_stats = self.tensor_logic.get_statistics()
            analysis["components"]["tensor_logic"] = tensor_stats
        
        # Constellation statistics
        if self.constellation_orchestrator:
            constellation_stats = self.constellation_orchestrator.get_statistics()
            analysis["components"]["constellations"] = constellation_stats
        
        # Processing history analysis
        analysis["processing_history"] = {
            "total_queries": len(self.processing_history),
            "recent_queries": [
                {"query": h["query"], "timestamp": h["timestamp"]}
                for h in self.processing_history[-5:]
            ],
        }
        
        # Knowledge base statistics
        analysis["knowledge_base"] = {
            "total_atoms": len(self.atomspace),
            "atom_types": self._count_atom_types(),
        }
        
        return analysis
    
    def _count_atom_types(self) -> Dict[str, int]:
        """Count atoms by type."""
        type_counts = {}
        for atom in self.atomspace.atoms.values():
            atom_type = atom.atom_type.value
            type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
        return type_counts
    
    async def consolidate_knowledge(self):
        """
        Consolidate knowledge across all modules.
        Performs memory consolidation, attention updates, and pattern discovery.
        """
        if not self._initialized:
            await self.initialize()
        
        # Cognitive memory consolidation
        if self.cognitive_arch:
            await self.cognitive_arch.memory.consolidate()
        
        # Update attention values based on usage
        for atom in self.atomspace.atoms.values():
            # Decay attention over time
            await self.atomspace.update_attention(atom.atom_id, -0.01)
        
        # Rediscover constellations
        if self.constellation_orchestrator:
            atoms = list(self.atomspace.atoms.values())
            await self.constellation_orchestrator.build_hypergraph_from_atoms(atoms)
            await self.constellation_orchestrator.discover_constellations()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "initialized": self._initialized,
            "config": self.config.to_dict(),
            "modules": {
                "tensor_logic": self.tensor_logic is not None,
                "cognitive_arch": self.cognitive_arch is not None,
                "notebook_gen": self.notebook_orchestrator is not None,
                "constellations": self.constellation_orchestrator is not None,
            },
            "knowledge_base": {
                "total_atoms": len(self.atomspace),
                "node_id": self.distributed_atomspace.node_id,
            },
            "processing_history_size": len(self.processing_history),
        }


# Global instance for convenience
_fabric_instance: Optional[NeuroSymbolicFabric] = None


def get_fabric(config: Optional[NeuroSymbolicConfig] = None) -> NeuroSymbolicFabric:
    """Get or create the global neuro-symbolic fabric instance."""
    global _fabric_instance
    
    if _fabric_instance is None:
        _fabric_instance = NeuroSymbolicFabric(config)
    
    return _fabric_instance


async def initialize_fabric(config: Optional[NeuroSymbolicConfig] = None):
    """Initialize the global fabric instance."""
    fabric = get_fabric(config)
    await fabric.initialize()
    return fabric
