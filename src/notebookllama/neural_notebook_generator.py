"""
Autonomous Neural Notebook Generator for RAGSwarm.

Generates intelligent notebooks from repository analysis using
cognitive architecture and neuro-symbolic reasoning.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import os
import json
import re

from .opencog_atomspace import Atom, AtomType, AtomSpace
from .cognitive_architecture import CognitiveArchitecture
from .tensor_logic import TensorLogicOrchestrator


class NotebookCellType(Enum):
    """Types of cells in a neural notebook."""
    MARKDOWN = "markdown"
    CODE = "code"
    ANALYSIS = "analysis"
    INSIGHT = "insight"
    QUERY = "query"
    VISUALIZATION = "visualization"


@dataclass
class NotebookCell:
    """Represents a cell in the neural notebook."""
    cell_type: NotebookCellType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "cell_type": self.cell_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "execution_order": self.execution_order,
        }


@dataclass
class NeuralNotebook:
    """Represents a complete neural notebook."""
    title: str
    description: str
    cells: List[NotebookCell] = field(default_factory=list)
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def add_cell(self, cell: NotebookCell):
        """Add a cell to the notebook."""
        cell.execution_order = len(self.cells) + 1
        self.cells.append(cell)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "title": self.title,
            "description": self.description,
            "cells": [cell.to_dict() for cell in self.cells],
            "knowledge_graph": self.knowledge_graph,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
    
    def to_jupyter_format(self) -> Dict[str, Any]:
        """Convert to Jupyter notebook format."""
        jupyter_cells = []
        
        for cell in self.cells:
            if cell.cell_type in [NotebookCellType.MARKDOWN, NotebookCellType.ANALYSIS, 
                                  NotebookCellType.INSIGHT, NotebookCellType.QUERY]:
                jupyter_cells.append({
                    "cell_type": "markdown",
                    "metadata": cell.metadata,
                    "source": [cell.content],
                })
            elif cell.cell_type == NotebookCellType.CODE:
                jupyter_cells.append({
                    "cell_type": "code",
                    "execution_count": cell.execution_order,
                    "metadata": cell.metadata,
                    "source": [cell.content],
                    "outputs": [],
                })
        
        return {
            "cells": jupyter_cells,
            "metadata": {
                **self.metadata,
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.13.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }


class RepositoryAnalyzer:
    """
    Analyzes repository structure and content to extract knowledge.
    """
    
    def __init__(self, cognitive_arch: CognitiveArchitecture):
        self.cognitive_arch = cognitive_arch
        self.atomspace = cognitive_arch.atomspace
    
    async def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Analyze a repository and extract structural knowledge.
        
        Args:
            repo_path: Path to the repository
        
        Returns:
            Analysis results including files, structure, and insights
        """
        if not os.path.exists(repo_path):
            return {"error": f"Repository path does not exist: {repo_path}"}
        
        analysis = {
            "repo_path": repo_path,
            "files": [],
            "structure": {},
            "language_distribution": {},
            "key_files": [],
            "dependencies": [],
            "insights": [],
        }
        
        # Scan repository
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and common build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                      ['node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                # Analyze file
                file_info = await self._analyze_file(file_path, rel_path)
                analysis["files"].append(file_info)
                
                # Update language distribution
                lang = file_info.get("language", "unknown")
                analysis["language_distribution"][lang] = \
                    analysis["language_distribution"].get(lang, 0) + 1
        
        # Identify key files
        analysis["key_files"] = self._identify_key_files(analysis["files"])
        
        # Generate insights using cognitive architecture
        insights = await self._generate_insights(analysis)
        analysis["insights"] = insights
        
        return analysis
    
    async def _analyze_file(self, file_path: str, rel_path: str) -> Dict[str, Any]:
        """Analyze a single file."""
        file_info = {
            "path": rel_path,
            "language": self._detect_language(file_path),
            "size": os.path.getsize(file_path),
            "is_key_file": False,
        }
        
        # Read file content (for text files)
        try:
            if file_info["size"] < 1_000_000:  # Skip very large files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    file_info["lines"] = len(content.split('\n'))
                    
                    # Extract key information
                    if file_info["language"] == "python":
                        file_info["imports"] = self._extract_python_imports(content)
                        file_info["functions"] = self._extract_python_functions(content)
                        file_info["classes"] = self._extract_python_classes(content)
                    
                    # Create atom for this file
                    atom = Atom(
                        atom_type=AtomType.DOCUMENT,
                        name=f"file:{rel_path}",
                        metadata={
                            "path": rel_path,
                            "language": file_info["language"],
                            "content_preview": content[:500],
                        },
                    )
                    await self.atomspace.add_atom(atom)
        except Exception as e:
            file_info["error"] = str(e)
        
        return file_info
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.html': 'html',
            '.css': 'css',
        }
        return language_map.get(ext, 'unknown')
    
    def _extract_python_imports(self, content: str) -> List[str]:
        """Extract Python import statements."""
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports[:20]  # Limit to first 20
    
    def _extract_python_functions(self, content: str) -> List[str]:
        """Extract Python function definitions."""
        pattern = r'^\s*def\s+(\w+)\s*\('
        functions = re.findall(pattern, content, re.MULTILINE)
        return functions[:50]  # Limit to first 50
    
    def _extract_python_classes(self, content: str) -> List[str]:
        """Extract Python class definitions."""
        pattern = r'^\s*class\s+(\w+)\s*[\(:]'
        classes = re.findall(pattern, content, re.MULTILINE)
        return classes[:30]  # Limit to first 30
    
    def _identify_key_files(self, files: List[Dict[str, Any]]) -> List[str]:
        """Identify key files in the repository."""
        key_files = []
        
        # Common key files
        key_names = [
            'readme.md', 'readme.rst', 'readme.txt',
            'setup.py', 'pyproject.toml', 'package.json',
            'requirements.txt', 'cargo.toml', 'go.mod',
            'main.py', 'app.py', 'index.js', 'main.go',
        ]
        
        for file_info in files:
            filename = os.path.basename(file_info["path"]).lower()
            if filename in key_names:
                key_files.append(file_info["path"])
                file_info["is_key_file"] = True
        
        return key_files
    
    async def _generate_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights about the repository using cognitive architecture."""
        insights = []
        
        # Language insights
        if analysis["language_distribution"]:
            main_lang = max(analysis["language_distribution"].items(), key=lambda x: x[1])
            insights.append(f"Primary language: {main_lang[0]} ({main_lang[1]} files)")
        
        # Structure insights
        total_files = len(analysis["files"])
        insights.append(f"Repository contains {total_files} files")
        
        # Key files insight
        if analysis["key_files"]:
            insights.append(f"Found {len(analysis['key_files'])} key configuration/entry files")
        
        # Use cognitive architecture for deeper analysis
        analysis_text = f"Repository with {total_files} files, primarily in {main_lang[0]}"
        cognitive_result = await self.cognitive_arch.perceive_and_process(
            analysis_text, modality="text"
        )
        
        return insights


class NotebookGenerator:
    """
    Generates neural notebooks from repository analysis.
    """
    
    def __init__(self, cognitive_arch: CognitiveArchitecture):
        self.cognitive_arch = cognitive_arch
        self.analyzer = RepositoryAnalyzer(cognitive_arch)
    
    async def generate_notebook(
        self, 
        repo_path: str, 
        focus: Optional[str] = None
    ) -> NeuralNotebook:
        """
        Generate a neural notebook from repository analysis.
        
        Args:
            repo_path: Path to the repository
            focus: Optional focus area (e.g., "architecture", "testing", "documentation")
        
        Returns:
            Generated neural notebook
        """
        # Analyze repository
        analysis = await self.analyzer.analyze_repository(repo_path)
        
        if "error" in analysis:
            # Create error notebook
            notebook = NeuralNotebook(
                title="Repository Analysis Error",
                description=analysis["error"],
            )
            return notebook
        
        # Create notebook
        repo_name = os.path.basename(os.path.abspath(repo_path))
        notebook = NeuralNotebook(
            title=f"Neural Notebook: {repo_name}",
            description=f"Autonomous analysis of repository: {repo_name}",
            metadata={
                "repo_path": repo_path,
                "focus": focus,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        )
        
        # Generate title cell
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.MARKDOWN,
            content=f"# Neural Notebook: {repo_name}\n\n"
                   f"*Generated by RAGSwarm Neural Notebook-LM*\n\n"
                   f"**Repository:** `{repo_path}`\n\n"
                   f"**Analysis Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ))
        
        # Generate overview section
        await self._generate_overview_section(notebook, analysis)
        
        # Generate architecture section
        await self._generate_architecture_section(notebook, analysis)
        
        # Generate key insights section
        await self._generate_insights_section(notebook, analysis)
        
        # Generate code analysis section
        await self._generate_code_analysis_section(notebook, analysis)
        
        # Generate recommendations section
        await self._generate_recommendations_section(notebook, analysis)
        
        # Build knowledge graph
        notebook.knowledge_graph = await self._build_knowledge_graph(analysis)
        
        return notebook
    
    async def _generate_overview_section(
        self, 
        notebook: NeuralNotebook, 
        analysis: Dict[str, Any]
    ):
        """Generate the overview section."""
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.MARKDOWN,
            content="## 📊 Repository Overview",
        ))
        
        # Statistics
        stats_content = "### Statistics\n\n"
        stats_content += f"- **Total Files:** {len(analysis['files'])}\n"
        
        # Language distribution
        if analysis["language_distribution"]:
            stats_content += "\n### Language Distribution\n\n"
            for lang, count in sorted(
                analysis["language_distribution"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                stats_content += f"- **{lang}:** {count} files\n"
        
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.ANALYSIS,
            content=stats_content,
        ))
    
    async def _generate_architecture_section(
        self, 
        notebook: NeuralNotebook, 
        analysis: Dict[str, Any]
    ):
        """Generate the architecture analysis section."""
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.MARKDOWN,
            content="## 🏗️ Architecture Analysis",
        ))
        
        # Key files
        if analysis["key_files"]:
            content = "### Key Files\n\n"
            content += "The following key files were identified:\n\n"
            for file_path in analysis["key_files"]:
                content += f"- `{file_path}`\n"
            
            notebook.add_cell(NotebookCell(
                cell_type=NotebookCellType.ANALYSIS,
                content=content,
            ))
        
        # File structure
        content = "### File Organization\n\n"
        
        # Group files by directory
        directories = {}
        for file_info in analysis["files"]:
            dir_path = os.path.dirname(file_info["path"])
            if not dir_path:
                dir_path = "root"
            
            if dir_path not in directories:
                directories[dir_path] = []
            directories[dir_path].append(file_info)
        
        content += f"Repository is organized into {len(directories)} directories.\n\n"
        
        # Show top directories by file count
        top_dirs = sorted(directories.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for dir_path, files in top_dirs:
            content += f"- `{dir_path}/`: {len(files)} files\n"
        
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.ANALYSIS,
            content=content,
        ))
    
    async def _generate_insights_section(
        self, 
        notebook: NeuralNotebook, 
        analysis: Dict[str, Any]
    ):
        """Generate the insights section using cognitive processing."""
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.MARKDOWN,
            content="## 💡 Cognitive Insights",
        ))
        
        # Use insights from analysis
        if analysis.get("insights"):
            content = "### Automated Analysis\n\n"
            for insight in analysis["insights"]:
                content += f"- {insight}\n"
            
            notebook.add_cell(NotebookCell(
                cell_type=NotebookCellType.INSIGHT,
                content=content,
            ))
        
        # Perform metacognitive reflection
        reflection = await self.cognitive_arch.metacognitive_reflection()
        
        content = "### Metacognitive Analysis\n\n"
        content += f"**Processing Cycles:** {reflection.get('processing_cycles', 0)}\n\n"
        
        if reflection.get("recommendations"):
            content += "**Recommendations:**\n\n"
            for rec in reflection["recommendations"]:
                content += f"- {rec}\n"
        
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.INSIGHT,
            content=content,
            metadata={"source": "metacognition"},
        ))
    
    async def _generate_code_analysis_section(
        self, 
        notebook: NeuralNotebook, 
        analysis: Dict[str, Any]
    ):
        """Generate code analysis section for Python files."""
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.MARKDOWN,
            content="## 🔍 Code Analysis",
        ))
        
        # Analyze Python files
        python_files = [f for f in analysis["files"] if f.get("language") == "python"]
        
        if python_files:
            content = f"### Python Code Structure\n\n"
            content += f"Found {len(python_files)} Python files.\n\n"
            
            # Aggregate functions and classes
            all_functions = []
            all_classes = []
            
            for file_info in python_files:
                all_functions.extend(file_info.get("functions", []))
                all_classes.extend(file_info.get("classes", []))
            
            if all_classes:
                content += f"**Total Classes:** {len(set(all_classes))}\n\n"
                content += "Sample classes:\n"
                for cls in list(set(all_classes))[:10]:
                    content += f"- `{cls}`\n"
                content += "\n"
            
            if all_functions:
                content += f"**Total Functions:** {len(set(all_functions))}\n\n"
                content += "Sample functions:\n"
                for func in list(set(all_functions))[:10]:
                    content += f"- `{func}()`\n"
            
            notebook.add_cell(NotebookCell(
                cell_type=NotebookCellType.ANALYSIS,
                content=content,
            ))
    
    async def _generate_recommendations_section(
        self, 
        notebook: NeuralNotebook, 
        analysis: Dict[str, Any]
    ):
        """Generate recommendations section."""
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.MARKDOWN,
            content="## 📋 Recommendations",
        ))
        
        recommendations = []
        
        # Check for documentation
        has_readme = any('readme' in f["path"].lower() for f in analysis["files"])
        if not has_readme:
            recommendations.append("Consider adding a README.md file for documentation")
        
        # Check for tests
        has_tests = any('test' in f["path"].lower() for f in analysis["files"])
        if not has_tests:
            recommendations.append("Consider adding automated tests")
        
        # Check for configuration
        config_files = [f for f in analysis["files"] if f.get("language") in ["json", "yaml", "toml"]]
        if not config_files:
            recommendations.append("Consider adding configuration files for better project management")
        
        if not recommendations:
            recommendations.append("Repository structure looks good!")
        
        content = ""
        for i, rec in enumerate(recommendations, 1):
            content += f"{i}. {rec}\n"
        
        notebook.add_cell(NotebookCell(
            cell_type=NotebookCellType.INSIGHT,
            content=content,
        ))
    
    async def _build_knowledge_graph(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build a knowledge graph from the analysis."""
        graph = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
            },
        }
        
        # Add file nodes
        for i, file_info in enumerate(analysis["files"][:50]):  # Limit to 50 files
            graph["nodes"].append({
                "id": f"file_{i}",
                "type": "file",
                "label": file_info["path"],
                "language": file_info.get("language", "unknown"),
            })
        
        # Add language nodes
        for lang in analysis["language_distribution"]:
            graph["nodes"].append({
                "id": f"lang_{lang}",
                "type": "language",
                "label": lang,
            })
        
        # Add edges connecting files to languages
        for i, file_info in enumerate(analysis["files"][:50]):
            lang = file_info.get("language", "unknown")
            graph["edges"].append({
                "source": f"file_{i}",
                "target": f"lang_{lang}",
                "type": "written_in",
            })
        
        return graph


class NotebookOrchestrator:
    """
    Orchestrates the autonomous generation of neural notebooks.
    """
    
    def __init__(self, atomspace: AtomSpace, embedding_dim: int = 128):
        self.cognitive_arch = CognitiveArchitecture(atomspace, embedding_dim)
        self.generator = NotebookGenerator(self.cognitive_arch)
        self.notebooks: Dict[str, NeuralNotebook] = {}
    
    async def generate_from_repository(
        self, 
        repo_path: str, 
        focus: Optional[str] = None
    ) -> NeuralNotebook:
        """
        Generate a neural notebook from a repository.
        
        Args:
            repo_path: Path to repository
            focus: Optional focus area
        
        Returns:
            Generated notebook
        """
        notebook = await self.generator.generate_notebook(repo_path, focus)
        
        # Store notebook
        notebook_id = f"notebook_{len(self.notebooks)}"
        self.notebooks[notebook_id] = notebook
        
        return notebook
    
    async def save_notebook(self, notebook: NeuralNotebook, output_path: str):
        """
        Save notebook to file.
        
        Args:
            notebook: Notebook to save
            output_path: Path to save to
        """
        # Determine format from extension
        if output_path.endswith('.ipynb'):
            # Save as Jupyter notebook
            jupyter_format = notebook.to_jupyter_format()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(jupyter_format, f, indent=2)
        else:
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook.to_dict(), f, indent=2)
    
    def get_notebook(self, notebook_id: str) -> Optional[NeuralNotebook]:
        """Get a generated notebook by ID."""
        return self.notebooks.get(notebook_id)
    
    def list_notebooks(self) -> List[str]:
        """List all generated notebook IDs."""
        return list(self.notebooks.keys())
