"""
Chess-Grid Generator for The Living Agent
==========================================
Generates an N×M grid of interconnected Markdown knowledge cells.
Each cell has up to 8 directional links to its neighbors.

Usage: python grid_generator.py [--rows 16] [--cols 16]
"""

import os
import random
import argparse

# Research domains to seed cells with diverse topics
RESEARCH_DOMAINS = [
    # Biological Computing
    "DNA-based logic gates and their computational limits",
    "Protein folding as a search algorithm",
    "Neural organoid computing architectures",
    "Synthetic biology circuits for Boolean operations",
    "Bacterial quorum sensing as distributed consensus",
    "Slime mold optimization and network design",
    "Enzyme cascades as analog signal processors",
    "Epigenetic memory in cellular computing",
    # Quantum Physics & Computing
    "Topological qubits and fault-tolerant quantum computation",
    "Quantum coherence in biological photosynthesis",
    "Variational quantum eigensolvers for molecular simulation",
    "Quantum error correction via surface codes",
    "Quantum reservoir computing with spin chains",
    "Quantum tunneling in enzyme catalysis",
    "Entanglement-assisted classical communication",
    "Quantum machine learning kernel methods",
    # P2PCLAW & Decentralized Knowledge
    "Proof-of-Discovery consensus mechanisms",
    "Semantic routing in knowledge graphs",
    "Decentralized AI governance frameworks",
    "Peer-to-peer scientific validation protocols",
    "Token-incentivized research contribution models",
    "Federated learning across heterogeneous agents",
    "Knowledge graph embedding and link prediction",
    "Merkle DAG structures for versioned knowledge",
    # Cognitive Architecture & AGI
    "Autopoietic systems and self-organization",
    "Cognitive architectures: SOAR vs ACT-R vs S²FSM",
    "Meta-learning and learning-to-learn paradigms",
    "Embodied cognition and situated AI",
    "Compositional generalization in neural networks",
    "Neuro-symbolic integration approaches",
    "Intrinsic motivation and curiosity-driven exploration",
    "Skill acquisition and procedural knowledge formation",
    # Physics & Emergence
    "Emergence and complexity in physical systems",
    "Information theory and thermodynamics of computation",
    "Self-organized criticality in neural networks",
    "Scale-free networks and preferential attachment",
    "Dissipative structures and non-equilibrium thermodynamics",
    "Holographic principle and information bounds",
    "Cellular automata and computational universality",
    "Renormalization group and multi-scale physics",
    # Cross-Domain Synthesis
    "Bio-inspired optimization: ant colony and swarm intelligence",
    "Morphogenetic computing: Turing patterns as programs",
    "Neuromorphic hardware: memristors and beyond",
    "Evolutionary strategies for neural architecture search",
    "Reservoir computing with physical substrates",
    "DNA data storage and retrieval systems",
    "Molecular communication and nanonetworks",
    "Synthetic ecosystems for emergent intelligence",
]

# 8 directions: name -> (row_offset, col_offset)
DIRECTIONS = {
    "N":  (-1,  0), "NE": (-1,  1), "E":  ( 0,  1), "SE": ( 1,  1),
    "S":  ( 1,  0), "SW": ( 1, -1), "W":  ( 0, -1), "NW": (-1, -1),
}

DIRECTION_EMOJI = {
    "N": "⬆️", "NE": "↗️", "E": "➡️", "SE": "↘️",
    "S": "⬇️", "SW": "↙️", "W": "⬅️", "NW": "↖️",
}


def get_cell_topic(row, col, rows, cols):
    """Assign a unique research topic to each cell."""
    idx = (row * cols + col) % len(RESEARCH_DOMAINS)
    return RESEARCH_DOMAINS[idx]


def get_cell_type(row, col, rows, cols):
    """Determine special cell types based on position."""
    if row == 0:
        return "ENTRY"
    elif row == rows - 1:
        return "SYNTHESIS"
    elif row == rows // 2 and col == cols // 2:
        return "MUTATION_CHAMBER"
    elif (row * cols + col) % 17 == 0:
        return "SKILL_NODE"
    elif (row * cols + col) % 23 == 0:
        return "EXPERIMENT_NODE"
    return "KNOWLEDGE"


def generate_cell(row, col, rows, cols):
    """Generate the Markdown content for a single grid cell."""
    topic = get_cell_topic(row, col, rows, cols)
    cell_type = get_cell_type(row, col, rows, cols)
    
    # Header
    lines = [f"# Cell [{row},{col}] — {cell_type}"]
    lines.append(f"**Grid Position**: Row {row}, Column {col}")
    lines.append(f"**Type**: {cell_type}")
    lines.append("")
    
    # Type-specific content
    if cell_type == "ENTRY":
        lines.append("## 🚀 Entry Point")
        lines.append(f"Welcome, Agent. You have entered the Chess-Grid at column {col}.")
        lines.append(f"Your mission: traverse the board toward Row {rows-1}, accumulating knowledge at every cell.")
        lines.append(f"**Research Focus**: {topic}")
        lines.append("")
        lines.append("Begin by choosing a direction below. Prefer SOUTH (⬇️) or diagonal moves to advance toward the synthesis edge.")
    elif cell_type == "SYNTHESIS":
        lines.append("## 📝 Synthesis Terminal")
        lines.append("You have reached the far edge of the Chess-Grid.")
        lines.append("**ACTION REQUIRED**: Synthesize all accumulated knowledge into a professional scientific paper.")
        lines.append(f"**Final Topic Integration**: {topic}")
        lines.append("")
        lines.append("After synthesis, compress your trace and re-enter at Row 0.")
    elif cell_type == "MUTATION_CHAMBER":
        lines.append("## 🧬 Mutation Chamber")
        lines.append("This is a special node. Analyze your recent performance.")
        lines.append("If your last 3 SNS scores were below 0.5, you should modify your research strategy.")
        lines.append(f"**Mutation Topic**: {topic}")
        lines.append("")
        lines.append("[ACQUIRED: agent reads this node → adds 'self_mutation' to COMPETENCY_MAP]")
    elif cell_type == "SKILL_NODE":
        skill_name = random.choice(["deep_analysis", "cross_reference", "hypothesis_generator", "evidence_evaluator", "pattern_recognition"])
        lines.append(f"## ⚡ Skill Node: `{skill_name}`")
        lines.append(f"**Research Context**: {topic}")
        lines.append("")
        lines.append(f"[ACQUIRED: agent reads this node → adds '{skill_name}' to COMPETENCY_MAP]")
    elif cell_type == "EXPERIMENT_NODE":
        lines.append("## 🔬 Experiment Node")
        lines.append(f"**Hypothesis**: {topic}")
        lines.append("")
        lines.append("Design a mental experiment to test this hypothesis.")
        lines.append("Record your prediction, methodology, and expected outcome.")
        lines.append("The result will be stored in your episodic memory.")
    else:
        lines.append(f"## 📚 Research Node")
        lines.append(f"**Topic**: {topic}")
        lines.append("")
        lines.append("Study this topic carefully. Extract key insights that connect to your SOUL's research goal.")
        lines.append("Consider how this knowledge intersects with biological computing and physics.")
    
    # Navigation section — 8 directions
    lines.append("")
    lines.append("---")
    lines.append("## 🧭 Navigation (Choose Your Direction)")
    lines.append("")
    
    for dir_name, (dr, dc) in DIRECTIONS.items():
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            emoji = DIRECTION_EMOJI[dir_name]
            target_topic = get_cell_topic(nr, nc, rows, cols)
            short_topic = target_topic[:50] + "..." if len(target_topic) > 50 else target_topic
            lines.append(f"- {emoji} **{dir_name}**: [{short_topic}](cell_R{nr}_C{nc}.md)")
    
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Chess-Grid for The Living Agent")
    parser.add_argument("--rows", type=int, default=16, help="Number of rows (default: 16)")
    parser.add_argument("--cols", type=int, default=16, help="Number of columns (default: 16)")
    parser.add_argument("--output", type=str, default="knowledge/grid", help="Output directory")
    args = parser.parse_args()
    
    rows, cols = args.rows, args.cols
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🏁 Generating {rows}×{cols} Chess-Grid ({rows*cols} cells)...")
    
    for r in range(rows):
        for c in range(cols):
            content = generate_cell(r, c, rows, cols)
            filename = f"cell_R{r}_C{c}.md"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
    
    # Generate grid index
    index_lines = ["# Chess-Grid Index", f"**Dimensions**: {rows}×{cols} = {rows*cols} cells", ""]
    index_lines.append("| | " + " | ".join([f"C{c}" for c in range(cols)]) + " |")
    index_lines.append("|---" * (cols + 1) + "|")
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            cell_type = get_cell_type(r, c, rows, cols)
            icon = {"ENTRY": "🚀", "SYNTHESIS": "📝", "MUTATION_CHAMBER": "🧬", 
                    "SKILL_NODE": "⚡", "EXPERIMENT_NODE": "🔬", "KNOWLEDGE": "📚"}[cell_type]
            row_cells.append(f"[{icon}](grid/cell_R{r}_C{c}.md)")
        index_lines.append(f"| **R{r}** | " + " | ".join(row_cells) + " |")
    
    index_path = os.path.join(os.path.dirname(output_dir), "grid_index.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    
    print(f"✅ Generated {rows*cols} cells in {output_dir}")
    print(f"📋 Grid index saved to {index_path}")


if __name__ == "__main__":
    main()
