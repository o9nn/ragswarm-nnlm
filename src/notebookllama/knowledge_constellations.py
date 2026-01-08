"""
Hypergraph Knowledge Constellations for RAGSwarm.

Advanced hypergraph-based knowledge representation with constellation patterns
for discovering emergent structures in knowledge networks.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

from .opencog_atomspace import Atom, AtomType, AtomSpace


class ConstellationType(Enum):
    """Types of knowledge constellations that can emerge."""
    CLUSTER = "cluster"  # Dense group of related concepts
    PATH = "path"  # Sequential chain of reasoning
    HUB = "hub"  # Central concept connecting many others
    BRIDGE = "bridge"  # Concept connecting disparate clusters
    CYCLE = "cycle"  # Circular reasoning or reference pattern
    HIERARCHY = "hierarchy"  # Parent-child taxonomic structure
    LATTICE = "lattice"  # Multi-dimensional concept space


@dataclass
class Constellation:
    """Represents a discovered knowledge constellation."""
    constellation_id: str
    constellation_type: ConstellationType
    atoms: List[str]  # Atom IDs in this constellation
    strength: float  # How strong/cohesive the constellation is [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "constellation_id": self.constellation_id,
            "constellation_type": self.constellation_type.value,
            "atoms": self.atoms,
            "strength": self.strength,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at,
        }


@dataclass
class HyperedgeData:
    """Represents a hyperedge connecting multiple atoms."""
    hyperedge_id: str
    atom_ids: List[str]  # Atoms connected by this hyperedge
    edge_type: str  # Type of relationship
    weight: float = 1.0  # Strength of connection
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hyperedge_id": self.hyperedge_id,
            "atom_ids": self.atom_ids,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class Hypergraph:
    """
    Enhanced hypergraph structure for complex knowledge representation.
    Extends basic AtomSpace with hyperedges and constellation detection.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.hyperedges: Dict[str, HyperedgeData] = {}
        self.incidence_matrix: Dict[str, Set[str]] = {}  # atom_id -> set of hyperedge_ids
    
    def add_hyperedge(self, atom_ids: List[str], edge_type: str, weight: float = 1.0) -> str:
        """
        Add a hyperedge connecting multiple atoms.
        
        Args:
            atom_ids: IDs of atoms to connect
            edge_type: Type of relationship
            weight: Strength of connection
        
        Returns:
            Hyperedge ID
        """
        hyperedge_id = f"he_{len(self.hyperedges)}"
        
        hyperedge = HyperedgeData(
            hyperedge_id=hyperedge_id,
            atom_ids=atom_ids,
            edge_type=edge_type,
            weight=weight,
        )
        
        self.hyperedges[hyperedge_id] = hyperedge
        
        # Update incidence matrix
        for atom_id in atom_ids:
            if atom_id not in self.incidence_matrix:
                self.incidence_matrix[atom_id] = set()
            self.incidence_matrix[atom_id].add(hyperedge_id)
        
        return hyperedge_id
    
    def get_connected_atoms(self, atom_id: str, edge_type: Optional[str] = None) -> Set[str]:
        """
        Get all atoms connected to the given atom via hyperedges.
        
        Args:
            atom_id: Source atom ID
            edge_type: Optional filter by edge type
        
        Returns:
            Set of connected atom IDs
        """
        if atom_id not in self.incidence_matrix:
            return set()
        
        connected = set()
        
        for hyperedge_id in self.incidence_matrix[atom_id]:
            hyperedge = self.hyperedges[hyperedge_id]
            
            if edge_type is None or hyperedge.edge_type == edge_type:
                # Add all other atoms in this hyperedge
                for other_atom_id in hyperedge.atom_ids:
                    if other_atom_id != atom_id:
                        connected.add(other_atom_id)
        
        return connected
    
    def get_hyperedge_degree(self, atom_id: str) -> int:
        """Get the number of hyperedges connected to an atom."""
        return len(self.incidence_matrix.get(atom_id, set()))
    
    def compute_clustering_coefficient(self, atom_id: str) -> float:
        """
        Compute local clustering coefficient for an atom.
        Measures how connected the neighbors are to each other.
        
        Args:
            atom_id: Atom to compute coefficient for
        
        Returns:
            Clustering coefficient [0, 1]
        """
        neighbors = self.get_connected_atoms(atom_id)
        
        if len(neighbors) < 2:
            return 0.0
        
        # Count connections between neighbors
        connections = 0
        neighbor_list = list(neighbors)
        
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] in self.get_connected_atoms(neighbor_list[i]):
                    connections += 1
        
        # Maximum possible connections
        max_connections = len(neighbors) * (len(neighbors) - 1) / 2
        
        return connections / max_connections if max_connections > 0 else 0.0


class ConstellationDetector:
    """
    Detects emergent constellation patterns in the hypergraph.
    """
    
    def __init__(self, hypergraph: Hypergraph):
        self.hypergraph = hypergraph
        self.detected_constellations: List[Constellation] = []
    
    async def detect_clusters(self, min_size: int = 3, min_density: float = 0.5) -> List[Constellation]:
        """
        Detect cluster constellations (dense groups of related concepts).
        
        Args:
            min_size: Minimum cluster size
            min_density: Minimum density (ratio of actual to possible connections)
        
        Returns:
            List of detected cluster constellations
        """
        clusters = []
        visited = set()
        
        # Get all atoms
        all_atoms = list(self.hypergraph.incidence_matrix.keys())
        
        for atom_id in all_atoms:
            if atom_id in visited:
                continue
            
            # Grow cluster from this seed
            cluster = self._grow_cluster(atom_id, visited, min_density)
            
            if len(cluster) >= min_size:
                # Compute cluster strength
                strength = self._compute_cluster_strength(cluster)
                
                constellation = Constellation(
                    constellation_id=f"cluster_{len(clusters)}",
                    constellation_type=ConstellationType.CLUSTER,
                    atoms=cluster,
                    strength=strength,
                    metadata={"size": len(cluster)},
                )
                
                clusters.append(constellation)
                self.detected_constellations.append(constellation)
        
        return clusters
    
    def _grow_cluster(self, seed_id: str, visited: Set[str], min_density: float) -> List[str]:
        """Grow a cluster from a seed atom."""
        cluster = [seed_id]
        visited.add(seed_id)
        
        candidates = list(self.hypergraph.get_connected_atoms(seed_id))
        
        for candidate_id in candidates:
            if candidate_id in visited:
                continue
            
            # Check if adding this candidate maintains density
            test_cluster = cluster + [candidate_id]
            if self._compute_cluster_density(test_cluster) >= min_density:
                cluster.append(candidate_id)
                visited.add(candidate_id)
                
                # Add its neighbors as candidates
                new_candidates = self.hypergraph.get_connected_atoms(candidate_id)
                candidates.extend([c for c in new_candidates if c not in visited])
        
        return cluster
    
    def _compute_cluster_density(self, atom_ids: List[str]) -> float:
        """Compute density of a cluster (ratio of actual to possible connections)."""
        if len(atom_ids) < 2:
            return 1.0
        
        # Count actual connections
        connections = 0
        for i, atom_id in enumerate(atom_ids):
            connected = self.hypergraph.get_connected_atoms(atom_id)
            for other_id in atom_ids[i+1:]:
                if other_id in connected:
                    connections += 1
        
        # Maximum possible connections
        max_connections = len(atom_ids) * (len(atom_ids) - 1) / 2
        
        return connections / max_connections if max_connections > 0 else 0.0
    
    def _compute_cluster_strength(self, atom_ids: List[str]) -> float:
        """Compute overall strength of a cluster."""
        if not atom_ids:
            return 0.0
        
        # Average clustering coefficient of atoms in cluster
        coefficients = []
        for atom_id in atom_ids:
            coef = self.hypergraph.compute_clustering_coefficient(atom_id)
            coefficients.append(coef)
        
        return sum(coefficients) / len(coefficients)
    
    async def detect_hubs(self, min_degree: int = 5) -> List[Constellation]:
        """
        Detect hub constellations (central concepts with many connections).
        
        Args:
            min_degree: Minimum number of connections to be considered a hub
        
        Returns:
            List of detected hub constellations
        """
        hubs = []
        
        for atom_id, hyperedges in self.hypergraph.incidence_matrix.items():
            degree = len(hyperedges)
            
            if degree >= min_degree:
                # Get all connected atoms
                connected = self.hypergraph.get_connected_atoms(atom_id)
                
                # Strength based on degree and attention
                strength = min(1.0, degree / 20.0)  # Normalize to [0, 1]
                
                constellation = Constellation(
                    constellation_id=f"hub_{len(hubs)}",
                    constellation_type=ConstellationType.HUB,
                    atoms=[atom_id] + list(connected),
                    strength=strength,
                    metadata={"degree": degree, "hub_atom": atom_id},
                )
                
                hubs.append(constellation)
                self.detected_constellations.append(constellation)
        
        return hubs
    
    async def detect_bridges(self) -> List[Constellation]:
        """
        Detect bridge constellations (atoms connecting disparate clusters).
        
        Returns:
            List of detected bridge constellations
        """
        bridges = []
        
        # First detect clusters
        clusters = await self.detect_clusters()
        
        if len(clusters) < 2:
            return bridges
        
        # Find atoms that connect different clusters
        for atom_id in self.hypergraph.incidence_matrix.keys():
            connected_clusters = set()
            
            # Check which clusters this atom connects to
            connected_atoms = self.hypergraph.get_connected_atoms(atom_id)
            
            for cluster in clusters:
                if any(a in connected_atoms for a in cluster.atoms):
                    connected_clusters.add(cluster.constellation_id)
            
            # If connects 2+ clusters, it's a bridge
            if len(connected_clusters) >= 2:
                strength = len(connected_clusters) / len(clusters)
                
                constellation = Constellation(
                    constellation_id=f"bridge_{len(bridges)}",
                    constellation_type=ConstellationType.BRIDGE,
                    atoms=[atom_id],
                    strength=strength,
                    metadata={
                        "connected_clusters": list(connected_clusters),
                        "bridge_atom": atom_id,
                    },
                )
                
                bridges.append(constellation)
                self.detected_constellations.append(constellation)
        
        return bridges
    
    async def detect_hierarchies(self) -> List[Constellation]:
        """
        Detect hierarchical constellations (parent-child structures).
        
        Returns:
            List of detected hierarchy constellations
        """
        hierarchies = []
        
        # Look for atoms with "is_a" or "subclass_of" type relationships
        hierarchy_edges = [
            he for he in self.hypergraph.hyperedges.values()
            if he.edge_type in ["is_a", "subclass_of", "parent_of"]
        ]
        
        if not hierarchy_edges:
            return hierarchies
        
        # Build hierarchy trees
        visited = set()
        
        for edge in hierarchy_edges:
            if edge.hyperedge_id in visited:
                continue
            
            # Build hierarchy from this edge
            hierarchy_atoms = self._build_hierarchy_tree(edge, visited)
            
            if len(hierarchy_atoms) >= 2:
                strength = min(1.0, len(hierarchy_atoms) / 10.0)
                
                constellation = Constellation(
                    constellation_id=f"hierarchy_{len(hierarchies)}",
                    constellation_type=ConstellationType.HIERARCHY,
                    atoms=hierarchy_atoms,
                    strength=strength,
                    metadata={"levels": self._count_hierarchy_levels(hierarchy_atoms)},
                )
                
                hierarchies.append(constellation)
                self.detected_constellations.append(constellation)
        
        return hierarchies
    
    def _build_hierarchy_tree(self, start_edge: HyperedgeData, visited: Set[str]) -> List[str]:
        """Build a hierarchy tree from a starting edge."""
        atoms = list(start_edge.atom_ids)
        visited.add(start_edge.hyperedge_id)
        
        # Find connected hierarchy edges
        for atom_id in start_edge.atom_ids:
            if atom_id not in self.hypergraph.incidence_matrix:
                continue
            
            for he_id in self.hypergraph.incidence_matrix[atom_id]:
                if he_id in visited:
                    continue
                
                edge = self.hypergraph.hyperedges[he_id]
                if edge.edge_type in ["is_a", "subclass_of", "parent_of"]:
                    child_atoms = self._build_hierarchy_tree(edge, visited)
                    atoms.extend([a for a in child_atoms if a not in atoms])
        
        return atoms
    
    def _count_hierarchy_levels(self, atom_ids: List[str]) -> int:
        """Count the number of levels in a hierarchy."""
        # Simple heuristic: use number of atoms as proxy for depth
        return min(len(atom_ids), 10)
    
    async def detect_all_constellations(self) -> Dict[str, List[Constellation]]:
        """
        Detect all types of constellations.
        
        Returns:
            Dictionary mapping constellation types to detected instances
        """
        results = {
            "clusters": await self.detect_clusters(),
            "hubs": await self.detect_hubs(),
            "bridges": await self.detect_bridges(),
            "hierarchies": await self.detect_hierarchies(),
        }
        
        return results
    
    def get_constellation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of detected constellations."""
        type_counts = {}
        for constellation in self.detected_constellations:
            ctype = constellation.constellation_type.value
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        
        avg_strength = (
            sum(c.strength for c in self.detected_constellations) / len(self.detected_constellations)
            if self.detected_constellations else 0.0
        )
        
        return {
            "total_constellations": len(self.detected_constellations),
            "by_type": type_counts,
            "average_strength": avg_strength,
            "timestamp": datetime.utcnow().isoformat(),
        }


class KnowledgeConstellationOrchestrator:
    """
    Orchestrates hypergraph construction and constellation detection.
    Provides high-level interface for working with knowledge constellations.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.hypergraph = Hypergraph(atomspace)
        self.detector = ConstellationDetector(self.hypergraph)
        self.constellation_history: List[Dict[str, Any]] = []
    
    async def build_hypergraph_from_atoms(self, atoms: List[Atom]):
        """
        Build hypergraph structure from atoms.
        Creates hyperedges based on atom relationships.
        
        Args:
            atoms: List of atoms to build hypergraph from
        """
        # Create hyperedges based on outgoing links
        for atom in atoms:
            if atom.outgoing:
                # Create hyperedge for this atom and its outgoing connections
                self.hypergraph.add_hyperedge(
                    atom_ids=[atom.atom_id] + atom.outgoing,
                    edge_type="links_to",
                    weight=atom.truth_value,
                )
        
        # Create semantic hyperedges based on metadata
        for i, atom1 in enumerate(atoms):
            for atom2 in atoms[i+1:]:
                # Check for shared metadata keys
                shared_keys = set(atom1.metadata.keys()) & set(atom2.metadata.keys())
                
                if shared_keys:
                    # Create semantic relationship
                    self.hypergraph.add_hyperedge(
                        atom_ids=[atom1.atom_id, atom2.atom_id],
                        edge_type="shares_metadata",
                        weight=len(shared_keys) / max(len(atom1.metadata), len(atom2.metadata)),
                    )
    
    async def discover_constellations(self) -> Dict[str, List[Constellation]]:
        """
        Discover all constellation patterns in the hypergraph.
        
        Returns:
            Dictionary of detected constellations by type
        """
        constellations = await self.detector.detect_all_constellations()
        
        # Record discovery
        self.constellation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "summary": self.detector.get_constellation_summary(),
            "constellations": {
                ctype: [c.to_dict() for c in clist]
                for ctype, clist in constellations.items()
            },
        })
        
        return constellations
    
    def get_constellation_by_id(self, constellation_id: str) -> Optional[Constellation]:
        """Get a specific constellation by ID."""
        for constellation in self.detector.detected_constellations:
            if constellation.constellation_id == constellation_id:
                return constellation
        return None
    
    def visualize_constellation(self, constellation: Constellation) -> Dict[str, Any]:
        """
        Create visualization data for a constellation.
        
        Args:
            constellation: Constellation to visualize
        
        Returns:
            Visualization data with nodes and edges
        """
        nodes = []
        edges = []
        
        # Add nodes for each atom in constellation
        for i, atom_id in enumerate(constellation.atoms):
            nodes.append({
                "id": atom_id,
                "label": atom_id,
                "group": constellation.constellation_type.value,
            })
        
        # Add edges based on hypergraph connections
        for i, atom_id in enumerate(constellation.atoms):
            connected = self.hypergraph.get_connected_atoms(atom_id)
            
            for other_id in constellation.atoms[i+1:]:
                if other_id in connected:
                    edges.append({
                        "source": atom_id,
                        "target": other_id,
                    })
        
        return {
            "constellation_id": constellation.constellation_id,
            "type": constellation.constellation_type.value,
            "nodes": nodes,
            "edges": edges,
            "metadata": constellation.metadata,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hypergraph and constellations."""
        return {
            "hypergraph": {
                "total_atoms": len(self.hypergraph.incidence_matrix),
                "total_hyperedges": len(self.hypergraph.hyperedges),
                "average_degree": (
                    sum(len(edges) for edges in self.hypergraph.incidence_matrix.values()) /
                    len(self.hypergraph.incidence_matrix)
                    if self.hypergraph.incidence_matrix else 0
                ),
            },
            "constellations": self.detector.get_constellation_summary(),
            "history_records": len(self.constellation_history),
        }
