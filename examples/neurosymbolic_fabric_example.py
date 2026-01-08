"""
Example: Neuro-Symbolic Fabric with RAGSwarm Neural Notebook-LM

This example demonstrates the complete neuro-symbolic fabric including:
- Tensor-logic orchestration
- Cognitive architecture
- Neural notebook generation
- Knowledge constellation discovery
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from notebookllama.neurosymbolic_fabric import (
    NeuroSymbolicFabric,
    NeuroSymbolicConfig,
    initialize_fabric
)
from notebookllama.opencog_atomspace import Atom, AtomType


async def demonstrate_tensor_logic():
    """Demonstrate tensor-logic reasoning."""
    print("\n" + "="*70)
    print("TENSOR-LOGIC ORCHESTRATION DEMONSTRATION")
    print("="*70 + "\n")
    
    # Configure with tensor-logic only for this demo
    config = NeuroSymbolicConfig(
        embedding_dim=64,
        enable_tensor_logic=True,
        enable_cognitive_arch=False,
        enable_constellations=False,
        enable_notebook_gen=False
    )
    
    fabric = await initialize_fabric(config)
    
    # Add knowledge atoms
    print("Adding knowledge atoms...")
    concepts = [
        ("machine_learning", "A field of AI focused on learning from data"),
        ("neural_networks", "Computing systems inspired by biological brains"),
        ("deep_learning", "ML technique using multi-layer neural networks"),
        ("supervised_learning", "Learning from labeled training data"),
        ("unsupervised_learning", "Learning patterns without labels")
    ]
    
    atoms = []
    for name, description in concepts:
        atom_id = await fabric.add_knowledge(
            content=name,
            atom_type=AtomType.CONCEPT,
            metadata={"description": description}
        )
        # Get the atom back
        atom = await fabric.atomspace.get_atom(atom_id)
        atoms.append(atom)
        print(f"  ✓ Added: {name}")
    
    # Perform neuro-symbolic reasoning
    print("\nPerforming neuro-symbolic query...")
    result = await fabric.process_with_neuro_symbolic_reasoning(
        "Explain the relationship between machine learning and neural networks",
        atoms
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Components used: {list(result['components'].keys())}")
    
    if 'tensor_logic' in result['components']:
        tl = result['components']['tensor_logic']
        print(f"\nTop neural rankings:")
        for i, (atom_dict, score) in enumerate(tl['neural_rankings'][:3], 1):
            print(f"  {i}. {atom_dict['name']} (score: {score:.4f})")
    
    # Tensor attention analysis
    print("\nTensor attention analysis...")
    focus_concept = "machine_learning"
    attention_result = await fabric.tensor_attention_analysis(
        focus_concept=focus_concept
    )
    
    print(f"\nFocus: {focus_concept}")
    print("Attention distribution:")
    for i, (atom_dict, score) in enumerate(attention_result['attention_distribution'][:3], 1):
        print(f"  {i}. {atom_dict['name']}: {score:.4f}")


async def demonstrate_cognitive_architecture():
    """Demonstrate cognitive architecture."""
    print("\n" + "="*70)
    print("COGNITIVE ARCHITECTURE DEMONSTRATION")
    print("="*70 + "\n")
    
    # Reset global instance for fresh config
    from notebookllama import neurosymbolic_fabric
    neurosymbolic_fabric._fabric_instance = None
    
    config = NeuroSymbolicConfig(
        embedding_dim=64,
        enable_tensor_logic=False,
        enable_cognitive_arch=True,
        enable_constellations=False,
        enable_notebook_gen=False
    )
    
    fabric = await initialize_fabric(config)
    
    # Perceptual processing
    print("Processing perceptual input...")
    inputs = [
        "This is a complex software system with multiple components",
        "The architecture follows a modular design pattern",
        "Code quality metrics indicate good maintainability"
    ]
    
    for inp in inputs:
        result = await fabric.cognitive_arch.perceive_and_process(inp, modality="text")
        print(f"  ✓ Processed: {inp[:50]}...")
        print(f"    Salience: {result['perception_salience']:.3f}")
    
    # Goal-directed processing
    print("\nGoal-directed cognitive processing...")
    goal = "Analyze system architecture and identify improvement opportunities"
    goal_result = await fabric.cognitive_goal_processing(goal)
    
    print(f"Goal: {goal}")
    print(f"Subgoals completed: {len(goal_result['subgoal_results'])}")
    for i, subgoal_result in enumerate(goal_result['subgoal_results'], 1):
        subgoal = subgoal_result['subgoal']
        print(f"  {i}. {subgoal['description']} - {subgoal['status']}")
    
    # Metacognitive reflection
    print("\nMetacognitive reflection...")
    reflection = await fabric.metacognitive_analysis()
    
    if 'components' in reflection and 'cognitive' in reflection['components']:
        cog = reflection['components']['cognitive']
        print(f"Processing cycles: {cog.get('processing_cycles', 0)}")
        print(f"Goal completion rate: {cog.get('goal_completion_rate', 0):.2%}")
        print("\nRecommendations:")
        for rec in cog.get('recommendations', []):
            print(f"  • {rec}")


async def demonstrate_knowledge_constellations():
    """Demonstrate knowledge constellation discovery."""
    print("\n" + "="*70)
    print("KNOWLEDGE CONSTELLATION DEMONSTRATION")
    print("="*70 + "\n")
    
    # Reset global instance for fresh config
    from notebookllama import neurosymbolic_fabric
    neurosymbolic_fabric._fabric_instance = None
    
    config = NeuroSymbolicConfig(
        embedding_dim=64,
        enable_tensor_logic=False,
        enable_cognitive_arch=False,
        enable_constellations=True,
        enable_notebook_gen=False
    )
    
    fabric = await initialize_fabric(config)
    
    # Create a knowledge network
    print("Building knowledge network...")
    concepts = [
        # Programming concepts
        ("python", AtomType.CONCEPT, {"domain": "programming"}),
        ("javascript", AtomType.CONCEPT, {"domain": "programming"}),
        ("java", AtomType.CONCEPT, {"domain": "programming"}),
        
        # ML concepts
        ("machine_learning", AtomType.CONCEPT, {"domain": "AI"}),
        ("neural_networks", AtomType.CONCEPT, {"domain": "AI"}),
        ("deep_learning", AtomType.CONCEPT, {"domain": "AI"}),
        
        # Data concepts
        ("database", AtomType.CONCEPT, {"domain": "data"}),
        ("sql", AtomType.CONCEPT, {"domain": "data"}),
        ("nosql", AtomType.CONCEPT, {"domain": "data"}),
    ]
    
    for name, atom_type, metadata in concepts:
        await fabric.add_knowledge(name, atom_type, metadata)
        print(f"  ✓ Added: {name}")
    
    # Discover patterns
    print("\nDiscovering knowledge constellations...")
    patterns = await fabric.discover_knowledge_patterns()
    
    print("\nDiscovered constellations:")
    for ctype, constellations in patterns['constellations'].items():
        if constellations:
            print(f"\n{ctype.upper()}:")
            for const in constellations[:3]:  # Show first 3 of each type
                print(f"  • {const['constellation_id']}: {len(const['atoms'])} atoms, "
                      f"strength={const['strength']:.3f}")
    
    # Statistics
    stats = patterns['statistics']
    print(f"\nHypergraph statistics:")
    print(f"  Total atoms: {stats['hypergraph']['total_atoms']}")
    print(f"  Total hyperedges: {stats['hypergraph']['total_hyperedges']}")
    print(f"  Average degree: {stats['hypergraph']['average_degree']:.2f}")


async def demonstrate_neural_notebook():
    """Demonstrate neural notebook generation."""
    print("\n" + "="*70)
    print("NEURAL NOTEBOOK GENERATION DEMONSTRATION")
    print("="*70 + "\n")
    
    # Reset global instance for fresh config
    from notebookllama import neurosymbolic_fabric
    neurosymbolic_fabric._fabric_instance = None
    
    config = NeuroSymbolicConfig(
        embedding_dim=64,
        enable_tensor_logic=False,
        enable_cognitive_arch=True,
        enable_constellations=False,
        enable_notebook_gen=True
    )
    
    fabric = await initialize_fabric(config)
    
    # Generate notebook for current repository
    print("Generating neural notebook for repository...")
    print("(This may take a moment...)\n")
    
    # Use current directory or examples directory
    repo_path = os.path.dirname(os.path.dirname(__file__))
    
    try:
        notebook = await fabric.generate_neural_notebook(repo_path)
        
        print(f"Generated notebook: {notebook.title}")
        print(f"Description: {notebook.description}")
        print(f"Total cells: {len(notebook.cells)}")
        
        print("\nNotebook structure:")
        for i, cell in enumerate(notebook.cells[:5], 1):  # Show first 5 cells
            content_preview = cell.content[:60].replace('\n', ' ')
            print(f"  {i}. [{cell.cell_type.value}] {content_preview}...")
        
        if len(notebook.cells) > 5:
            print(f"  ... and {len(notebook.cells) - 5} more cells")
        
        # Knowledge graph info
        kg = notebook.knowledge_graph
        if kg.get('nodes'):
            print(f"\nKnowledge graph:")
            print(f"  Nodes: {len(kg['nodes'])}")
            print(f"  Edges: {len(kg['edges'])}")
        
        # Save notebook
        output_path = "/tmp/neural_notebook_demo.json"
        await fabric.notebook_orchestrator.save_notebook(notebook, output_path)
        print(f"\nNotebook saved to: {output_path}")
        
    except Exception as e:
        print(f"Note: Could not generate notebook - {e}")
        print("This is expected if running in a limited environment")


async def demonstrate_full_integration():
    """Demonstrate full neuro-symbolic integration."""
    print("\n" + "="*70)
    print("FULL NEURO-SYMBOLIC INTEGRATION DEMONSTRATION")
    print("="*70 + "\n")
    
    # Reset global instance for fresh config
    from notebookllama import neurosymbolic_fabric
    neurosymbolic_fabric._fabric_instance = None
    
    # Enable all components
    config = NeuroSymbolicConfig(
        embedding_dim=64,
        enable_tensor_logic=True,
        enable_cognitive_arch=True,
        enable_constellations=True,
        enable_notebook_gen=True
    )
    
    fabric = await initialize_fabric(config)
    
    print("Neuro-symbolic fabric initialized with all components\n")
    
    # Add diverse knowledge
    print("Building knowledge base...")
    knowledge_items = [
        ("reinforcement_learning", "ML paradigm for learning through interaction"),
        ("computer_vision", "AI field focused on visual understanding"),
        ("natural_language_processing", "AI field for language understanding"),
        ("robotics", "Integration of AI with physical systems"),
    ]
    
    atoms = []
    for name, description in knowledge_items:
        atom_id = await fabric.add_knowledge(
            name,
            AtomType.CONCEPT,
            {"description": description}
        )
        atom = await fabric.atomspace.get_atom(atom_id)
        atoms.append(atom)
    
    print(f"  ✓ Added {len(knowledge_items)} concepts\n")
    
    # Full neuro-symbolic query
    print("Performing full neuro-symbolic query...")
    query = "How do these AI concepts relate to each other?"
    
    result = await fabric.process_with_neuro_symbolic_reasoning(query, atoms)
    
    print(f"\nQuery: {query}")
    print(f"Components involved: {', '.join(result['components'].keys())}")
    
    # System status
    print("\nSystem status:")
    status = fabric.get_system_status()
    print(f"  Knowledge base size: {status['knowledge_base']['total_atoms']} atoms")
    print(f"  Active modules: {sum(status['modules'].values())}/{len(status['modules'])}")
    print(f"  Processing history: {status['processing_history_size']} queries")
    
    # Metacognitive overview
    print("\nMetacognitive analysis...")
    analysis = await fabric.metacognitive_analysis()
    
    print("  Component statistics:")
    for component, stats in analysis.get('components', {}).items():
        if isinstance(stats, dict):
            print(f"    {component}: {len(stats)} metrics")


async def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("RAGSwarm Neural Notebook-LM: Neuro-Symbolic Fabric Examples")
    print("="*70)
    
    demos = [
        ("Tensor-Logic Orchestration", demonstrate_tensor_logic),
        ("Cognitive Architecture", demonstrate_cognitive_architecture),
        ("Knowledge Constellations", demonstrate_knowledge_constellations),
        ("Neural Notebook Generation", demonstrate_neural_notebook),
        ("Full Integration", demonstrate_full_integration),
    ]
    
    for title, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\nError in {title}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
