# -*- coding: utf-8 -*-
"""
Rural Postman Problem (RPP) solver for Multi-Material Printing Path Planning Problem (MMPPP)

This module provides classes and methods to 
- define unit cells and lattices,
- tessellate unit cells into lattices and apply noise and boundaries,
- visualize structures,
- apply nozzle offsets and plan paths,
- export path and generate (Aerotech A3200-style) Gcodes.

@author: Zefang Li, Phd Candidate, Johns Hopkins University, zli299@jhu.edu
@author: Jochen Mueller, Assistant Professor, Johns Hopkins University, jmueller@jhu.edu
"""

# using Python 3.11.5
import networkx as nx #using 3.4.2, version older than 3.1 may make path differently at even-degree vertices, but this does not change cost
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; suppresses pop-up plot windows. Remove this line to re-enable interactive display.
import matplotlib.pyplot as plt
import json
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist

import os
# import sys
# os.environ['PYTHONHASHSEED'] = '1'
# os.execv(sys.executable, [sys.executable] + sys.argv)

# Output folder
foldername = "RPP_output"

# Store nodes and edges with materials for a unit cell
class UnitCell:
    def __init__(self, nodes=None, edges=None):
        """
        Initialize a UnitCell with optional nodes and edges.
        
        Args:
            nodes (list): List of [x, y] coordinates. Default: empty list
            edges (list): List of [node_idx_u, node_idx_v, material_name] tuples. Default: empty list
        """
        self.coords = nodes if nodes is not None else []
        # Dictionary to store connections: {'MaterialName': [[u, v], ...]}
        self.materials = {}
        
        # Auto-generate distinct colors for materials
        self._color_map = {}
        self._color_palette = self._generate_color_palette()
        self._color_index = 0
        
        # Add edges if provided
        if edges is not None:
            for edge in edges:
                if len(edge) == 3:
                    u, v, material = edge
                    self.add_connection(u, v, material)

    def _generate_color_palette(self):
        """
        Generates a diverse palette of visually distinct colors.
        Uses a combination of qualitative colormaps and custom colors.
        """
        # Start with tableau colors (10 distinct colors)
        tableau = list(mcolors.TABLEAU_COLORS.values())
        
        # Add CSS4 colors that are visually distinct
        extra_colors = [
            'gold', 'crimson', 'darkviolet', 'lime', 'deepskyblue',
            'orangered', 'mediumspringgreen', 'hotpink', 'navy', 'chocolate',
            'teal', 'coral', 'indigo', 'yellowgreen', 'tomato',
            'steelblue', 'orchid', 'sienna', 'lightseagreen', 'salmon'
        ]
        
        # Combine and return
        return tableau + extra_colors

    def _get_color_for_material(self, material_name):
        """
        Assigns and retrieves a unique color for each material.
        Uses cycling through the palette if more materials than colors.
        """
        if material_name not in self._color_map:
            color = self._color_palette[self._color_index % len(self._color_palette)]
            self._color_map[material_name] = color
            self._color_index += 1
        return self._color_map[material_name]

    def add_node(self, x, y):
        """
        Adds a node coordinate to the cell.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            
        Returns:
            int: Index of the added node
        """
        self.coords.append([x, y])
        return len(self.coords) - 1

    def add_nodes(self, node_list):
        """
        Adds multiple nodes at once.
        
        Args:
            node_list (list): List of [x, y] coordinate pairs
            
        Returns:
            list: Indices of added nodes
        """
        indices = []
        for x, y in node_list:
            indices.append(self.add_node(x, y))
        return indices

    def add_connection(self, u, v, material_type):
        """
        Adds a connection (edge) between node indices u and v for a specific material.
        
        Args:
            u (int): Index of first node
            v (int): Index of second node
            material_type (str): Name/type of the material
        """
        if material_type not in self.materials:
            self.materials[material_type] = []
        self.materials[material_type].append([u, v])
        
        # Ensure color is assigned when material is first added
        self._get_color_for_material(material_type)

    def add_connections(self, connection_list):
        """
        Adds multiple connections at once.
        
        Args:
            connection_list (list): List of [u, v, material_type] tuples
        """
        for u, v, material_type in connection_list:
            self.add_connection(u, v, material_type)

    def get_lines(self, material_type):
        """
        Returns a list of coordinate pairs [[x1,y1], [x2,y2]] for a given material.
        
        Args:
            material_type (str): Name/type of the material
            
        Returns:
            list: List of line segments as coordinate pairs
        """
        lines = []
        if material_type in self.materials:
            for u, v in self.materials[material_type]:
                p1 = self.coords[u]
                p2 = self.coords[v]
                lines.append([p1, p2])
        return lines

    def get_color(self, material_type):
        """
        Returns the assigned color for a material type.
        
        Args:
            material_type (str): Name/type of the material
            
        Returns:
            str: Color name or hex code
        """
        return self._get_color_for_material(material_type)

    def get_all_materials(self):
        """
        Returns a list of all material types in the unit cell.
        
        Returns:
            list: List of material type names
        """
        return list(self.materials.keys())

    def get_color_map(self):
        """
        Returns the complete color mapping for all materials.
        
        Returns:
            dict: Dictionary mapping material names to colors
        """
        return dict(self._color_map)

    def visualize(self, save_path=None, show_nodes=True, show_labels=False, fontsize = 8):
        """
        Plots the unit cell with automatically assigned colors for each material.
        
        Args:
            save_path (str): Path to save the figure (optional)
            show_nodes (bool): Whether to show node markers
            show_labels (bool): Whether to show node index labels
        """
        fig = plt.figure(figsize=(2.36,2.36),dpi=300)
        
        # Plot edges for each material
        for mat_type, connections in self.materials.items():
            color = self.get_color(mat_type)
            for u, v in connections:
                p1 = self.coords[u]
                p2 = self.coords[v]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=color, linewidth=1.5, label=mat_type, alpha=0.8)
        
        # Plot nodes if requested
        if show_nodes:
            for idx, coord in enumerate(self.coords):
                plt.scatter(coord[0], coord[1], s=30, c='black', zorder=5)
                if show_labels:
                    plt.text(coord[0], coord[1], str(idx), 
                            fontsize=fontsize, ha='center', va='bottom')
        
        # Add legend (remove duplicates)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.title('Unit Cell')
        plt.axis('equal')
        # plt.grid(True, alpha=0.3)
        
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight',dpi=300)
        plt.show()

    def __repr__(self):
        """String representation of the UnitCell."""
        mat_summary = {mat: len(edges) for mat, edges in self.materials.items()}
        return f"UnitCell(nodes={len(self.coords)}, materials={mat_summary})"

# Store in a single graph the tessellated lattice from a unit cell
class Lattice:
    def __init__(self, unit_cell=None, n_x=1, n_y=1, tes_x=None, tes_y=None, 
                 x_min=None, x_max=None, y_min=None, y_max=None):
        """
        Initialize a Lattice from a UnitCell with tessellation parameters.
        
        Args:
            unit_cell (UnitCell): The base unit cell to tessellate
            n_x (int): Number of repetitions in x direction
            n_y (int): Number of repetitions in y direction
            tes_x (float): Translation distance in x for each repeat (default: auto from unit cell)
            tes_y (float): Translation distance in y for each repeat (default: auto from unit cell)
            x_min (float): Minimum x boundary (optional)
            x_max (float): Maximum x boundary (optional)
            y_min (float): Minimum y boundary (optional)
            y_max (float): Maximum y boundary (optional)
        """
        # Store construction parameters
        self.unit_cell = unit_cell
        self.n_x = n_x
        self.n_y = n_y
        
        # NetworkX graphs to store lattice structure
        self.graph = nx.Graph()  # Main lattice graph
        
        # Color map inherited from unit cell
        self._color_map = {}
        
        # Auto-calculate tessellation vectors if not provided
        if unit_cell is not None and unit_cell.coords:
            if tes_x is None:
                tes_x = max([coord[0] for coord in unit_cell.coords])
            if tes_y is None:
                tes_y = max([coord[1] for coord in unit_cell.coords])
        
        self.tes_x = tes_x
        self.tes_y = tes_y
        
        # Build lattice if unit cell provided
        if unit_cell is not None:
            self._color_map = unit_cell.get_color_map().copy()
            self._tessellate()
  
            
            # Apply boundaries if specified
            if any(b is not None for b in [x_min, x_max, y_min, y_max]):
                self.apply_boundaries(x_min, x_max, y_min, y_max)

    def _tessellate(self):
        """Replicates the unit cell across a grid."""
        if self.unit_cell is None:
            return
        
        # Node ID tracking: maps (x, y, material) -> node_id
        node_map = {}
        next_node_id = 0
        
        for iY in range(self.n_y):
            for iX in range(self.n_x):
                offset_x = iX * self.tes_x
                offset_y = iY * self.tes_y
                
                # Process each material type
                for mat_type in self.unit_cell.materials:
                    connections = self.unit_cell.materials[mat_type]
                    
                    for u, v in connections:
                        # Get original coordinates
                        p1 = self.unit_cell.coords[u]
                        p2 = self.unit_cell.coords[v]
                        
                        # Apply offset
                        new_p1 = [p1[0] + offset_x, p1[1] + offset_y]
                        new_p2 = [p2[0] + offset_x, p2[1] + offset_y]
                        
                        # Create unique keys for nodes (position-based)
                        key1 = (round(new_p1[0], 6), round(new_p1[1], 6))
                        key2 = (round(new_p2[0], 6), round(new_p2[1], 6))
                        
                        # Add nodes if they don't exist
                        if key1 not in node_map:
                            node_map[key1] = next_node_id
                            self.graph.add_node(next_node_id, 
                                              x=new_p1[0], 
                                              y=new_p1[1],
                                              pos=key1)
                            next_node_id += 1
                        
                        if key2 not in node_map:
                            node_map[key2] = next_node_id
                            self.graph.add_node(next_node_id, 
                                              x=new_p2[0], 
                                              y=new_p2[1],
                                              pos=key2)
                            next_node_id += 1
                        
                        # Add edge with material property
                        node_id1 = node_map[key1]
                        node_id2 = node_map[key2]
                        
                        # Calculate weight (Euclidean distance)
                        weight = np.sqrt((new_p2[0] - new_p1[0])**2 + 
                                       (new_p2[1] - new_p1[1])**2)
                        
                        self.graph.add_edge(node_id1, node_id2, 
                                          material=mat_type,
                                          weight=weight)


    def apply_boundaries(self, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        Removes nodes and edges outside the specified bounding box.
        
        Args:
            x_min (float): Minimum x coordinate
            x_max (float): Maximum x coordinate
            y_min (float): Minimum y coordinate
            y_max (float): Maximum y coordinate
        """
        # Set default boundaries if not specified
        if x_min is None:
            x_min = float('-inf')
        if x_max is None:
            x_max = float('inf')
        if y_min is None:
            y_min = float('-inf')
        if y_max is None:
            y_max = float('inf')
        
        # Find nodes to remove
        nodes_to_remove = []
        for node, data in self.graph.nodes(data=True):
            x, y = data['x'], data['y']
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                nodes_to_remove.append(node)
        
        # Remove nodes (edges are automatically removed)
        self.graph.remove_nodes_from(nodes_to_remove)

    def apply_noise(self, rng_val, seed=None):
        """
        Applies stochastic noise to node positions.

        Args:
            rng_val (float): Maximum displacement in each direction (±rng_val)
            seed (int, optional): Seed for reproducible noise. If None, noise is
                                  non-deterministic. Uses a local numpy Generator
                                  so it does not affect global RNG state.
        """
        if rng_val <= 0:
            return

        rng = np.random.default_rng(seed)

        # Apply jitter to each node (deterministic node order for reproducibility)
        for node in sorted(self.graph.nodes()):
            noise_x, noise_y = rng.uniform(-rng_val, rng_val, size=2)

            self.graph.nodes[node]['x'] += noise_x
            self.graph.nodes[node]['y'] += noise_y

            # Update position key
            new_x = self.graph.nodes[node]['x']
            new_y = self.graph.nodes[node]['y']
            self.graph.nodes[node]['pos'] = (round(new_x, 6), round(new_y, 6))

        # Recalculate edge weights after noise
        for u, v in self.graph.edges():
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            weight = np.sqrt((v_data['x'] - u_data['x'])**2 +
                           (v_data['y'] - u_data['y'])**2)
            self.graph[u][v]['weight'] = weight

    def randomize_materials(self, probabilities=None, seed=None):
        """
        Randomizes the material assignment of each edge in the lattice.

        Two modes:
        - Shuffle mode (probabilities=None): Permutes the existing material
          assignments across edges. Exact per-material counts are preserved.
        - Sample mode (probabilities=dict): For each edge, draws a material
          i.i.d. from the given distribution. Counts will vary stochastically.

        Node positions and edge weights are unchanged.

        Args:
            probabilities (dict, optional): {material_name: probability, ...}.
                Probabilities are auto-normalized to sum to 1. New material names
                not currently in the lattice are accepted (a color is assigned
                automatically). Existing materials omitted from the dict are
                given probability 0 and a warning is printed.
            seed (int, optional): Seed for reproducibility. Uses a local numpy
                Generator so it does not affect global RNG state.

        Returns:
            Lattice: self, for method chaining.
        """
        # Deterministic edge ordering for reproducibility
        edges = sorted(self.graph.edges(), key=lambda e: (min(e), max(e)))
        if not edges:
            return self

        rng = np.random.default_rng(seed)

        if probabilities is None:
            # Shuffle mode: permute existing assignments, preserving counts
            current = [self.graph[u][v]['material'] for u, v in edges]
            shuffled = rng.permutation(current)
            for (u, v), mat in zip(edges, shuffled):
                self.graph[u][v]['material'] = str(mat)
        else:
            # Sample mode: i.i.d. draw from the given distribution
            existing = set(data['material'] for _, _, data in self.graph.edges(data=True))
            omitted = existing - set(probabilities.keys())
            if omitted:
                print(f"Warning: existing materials omitted from probabilities "
                      f"(will be assigned probability 0): {sorted(omitted)}")

            mats = list(probabilities.keys())
            probs = np.array([probabilities[m] for m in mats], dtype=float)
            if np.any(probs < 0):
                raise ValueError("Probabilities must be non-negative.")
            total = probs.sum()
            if total <= 0:
                raise ValueError("Probabilities must sum to a positive value.")
            if not np.isclose(total, 1.0):
                print(f"Warning: probabilities sum to {total:.6f}, normalizing to 1.")
                probs = probs / total

            # Auto-assign colors for any new materials so visualize() doesn't fall back to blue
            self._ensure_colors_for_materials(mats)

            draws = rng.choice(len(mats), size=len(edges), p=probs)
            for (u, v), idx in zip(edges, draws):
                self.graph[u][v]['material'] = mats[int(idx)]

        return self

    def _ensure_colors_for_materials(self, materials):
        """
        Ensures every material name has an entry in self._color_map.
        Assigns colors from a built-in palette for any new materials.
        """
        palette = list(mcolors.TABLEAU_COLORS.values()) + [
            'gold', 'crimson', 'darkviolet', 'lime', 'deepskyblue',
            'orangered', 'mediumspringgreen', 'hotpink', 'navy', 'chocolate',
            'teal', 'coral', 'indigo', 'yellowgreen', 'tomato',
            'steelblue', 'orchid', 'sienna', 'lightseagreen', 'salmon'
        ]
        for mat in materials:
            if mat not in self._color_map:
                self._color_map[mat] = palette[len(self._color_map) % len(palette)]

    def get_structure_by_material(self):
        """
        Returns the lattice structure organized by material type.
        Compatible with original format: {'MaterialName': [[[x1,y1],[x2,y2]], ...]}
        
        Returns:
            dict: Dictionary mapping material names to lists of line segments
        """
        structure = {}
        
        for u, v, data in self.graph.edges(data=True):
            material = data['material']
            if material not in structure:
                structure[material] = []
            
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            
            line = [[u_data['x'], u_data['y']], [v_data['x'], v_data['y']]]
            structure[material].append(line)
        
        return structure
    
    def get_lattice_by_material(self, material_name=None):
        """
        Returns new Lattice instance(s) containing only specified material(s).
        
        Args:
            material_name (str or list): Specific material name, list of names, or None for all
            
        Returns:
            Lattice or dict: 
                - If material_name is a string: returns single Lattice instance
                - If material_name is a list: returns dict {material: Lattice}
                - If material_name is None: returns dict of all materials {material: Lattice}
        """
        # Determine which materials to extract
        if material_name is None:
            # Get all materials
            materials_to_extract = set(data['material'] for _, _, data in self.graph.edges(data=True))
            return_dict = True
        elif isinstance(material_name, list):
            materials_to_extract = set(material_name)
            return_dict = True
        else:
            # Single material
            materials_to_extract = {material_name}
            return_dict = False
        
        result = {}
        
        for material in materials_to_extract:
            # Create new empty lattice
            new_lattice = Lattice()
            
            # Copy relevant parameters
            new_lattice.unit_cell = self.unit_cell
            new_lattice.n_x = self.n_x
            new_lattice.n_y = self.n_y
            new_lattice.tes_x = self.tes_x
            new_lattice.tes_y = self.tes_y
            new_lattice._color_map = self._color_map.copy()
            
            # Build node mapping (original node -> new node)
            node_mapping = {}
            next_node_id = 0
            
            # Extract edges for this material
            for u, v, data in self.graph.edges(data=True):
                if data['material'] == material:
                    # Add nodes if not already added
                    if u not in node_mapping:
                        u_data = self.graph.nodes[u]
                        node_mapping[u] = next_node_id
                        new_lattice.graph.add_node(next_node_id,
                                                  x=u_data['x'],
                                                  y=u_data['y'],
                                                  pos=u_data.get('pos'))
                        next_node_id += 1
                    
                    if v not in node_mapping:
                        v_data = self.graph.nodes[v]
                        node_mapping[v] = next_node_id
                        new_lattice.graph.add_node(next_node_id,
                                                  x=v_data['x'],
                                                  y=v_data['y'],
                                                  pos=v_data.get('pos'))
                        next_node_id += 1
                    
                    # Add edge with new node IDs
                    new_u = node_mapping[u]
                    new_v = node_mapping[v]
                    new_lattice.graph.add_edge(new_u, new_v,
                                              material=material,
                                              weight=data['weight'])
            
            result[material] = new_lattice
        
        # Return format based on input
        if return_dict:
            return result
        else:
            # Single material requested
            return result.get(material_name)

    def get_lattices_by_material(self):
        """
        Convenience method that returns a dictionary of Lattice instances, one per material.
        Equivalent to get_lattice_by_material(material_name=None)
        
        Returns:
            dict: Dictionary mapping material names to Lattice instances
        """
        return self.get_lattice_by_material(material_name=None)  

    def visualize(self, save_path=None, show_nodes=False, show_labels=False, 
                  materials=None, title='Lattice', fontsize=8):
        """
        Plots the lattice with colors based on material type.
        
        Args:
            save_path (str): Path to save the figure (optional)
            show_nodes (bool): Whether to show node markers
            show_labels (bool): Whether to show node labels
            materials (list): List of materials to display (None = all)
            title (str): Plot title
        """
        fig = plt.figure(figsize=(2.36,2.36),dpi=300)
        
        # Get edges by material
        edges_by_material = {}
        for u, v, data in self.graph.edges(data=True):
            material = data['material']
            if materials is None or material in materials:
                if material not in edges_by_material:
                    edges_by_material[material] = []
                edges_by_material[material].append((u, v))
        
        # Plot edges for each material
        for material, edges in edges_by_material.items():
            color = self._color_map.get(material, 'blue')
            for u, v in edges:
                u_data = self.graph.nodes[u]
                v_data = self.graph.nodes[v]
                plt.plot([u_data['x'], v_data['x']], 
                        [u_data['y'], v_data['y']], 
                        color=color, linewidth=1.5, label=material, alpha=0.8)
        
        # Plot nodes if requested
        if show_nodes:
            for node, data in self.graph.nodes(data=True):
                plt.scatter(data['x'], data['y'], s=30, c='black', zorder=5)
                if show_labels:
                    plt.text(data['x'], data['y'], str(node), 
                            fontsize=fontsize, ha='center', va='bottom')
        
        # Add legend (remove duplicates)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.title(title)
        plt.axis('equal')
        # plt.grid(True, alpha=0.8)
        plt.tight_layout()
        
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight',dpi=300)
        plt.show()

    def export_data(self, base_path=foldername+'/data/'):
        """
        Exports lattice structure to CSV/JSON files (backward compatible format).
        Exports one file per material type.
        
        Args:
            base_path (str): Directory path for output files
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        structure = self.get_structure_by_material()
        
        for material, lines in structure.items():
            # Sanitize material name for filename
            safe_name = material.replace(' ', '_').replace('/', '_')
            filepath = os.path.join(base_path, f'structure_{safe_name}.csv')
            
            with open(filepath, 'w') as f:
                json.dump(lines, f)
            
            print(f"Exported {len(lines)} {material} lines to {filepath}")
        
    def get_stats(self):
        """
        Returns statistics about the lattice.
        
        Returns:
            dict: Dictionary with lattice statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_components': nx.number_connected_components(self.graph),
            'materials': {}
        }
        
        # Count edges per material
        for u, v, data in self.graph.edges(data=True):
            material = data['material']
            if material not in stats['materials']:
                stats['materials'][material] = 0
            stats['materials'][material] += 1
        
        return stats

    def __repr__(self):
        """String representation of the Lattice."""
        stats = self.get_stats()
        return (f"Lattice(nodes={stats['num_nodes']}, "
                f"edges={stats['num_edges']}, "
                f"materials={list(stats['materials'].keys())})")
    
    def apply_offset(self, offset_x, offset_y):
        """
        Applies a spatial offset to all nodes in the lattice.
        
        Args:
            offset_x (float): Offset in x direction
            offset_y (float): Offset in y direction
            
        Returns:
            Lattice: Returns self for method chaining
        """
        for node in self.graph.nodes():
            self.graph.nodes[node]['x'] += offset_x
            self.graph.nodes[node]['y'] += offset_y
            
            # Update position key
            new_x = self.graph.nodes[node]['x']
            new_y = self.graph.nodes[node]['y']
            self.graph.nodes[node]['pos'] = (round(new_x, 6), round(new_y, 6))
        
        return self
    
    def apply_material_offsets(self, offsets):
        """
        Applies different spatial offsets to each material and returns a merged lattice.
        
        Args:
            offsets (dict or list): 
                - If dict: {material_name: [x_offset, y_offset], ...}
                - If list: [[x1, y1], [x2, y2], ...] in order of materials
                
        Returns:
            Lattice: New merged lattice with all materials offset
            
        Example:
            # Using dict
            offsets = {'Active': [0.1, 0.5], 'Inactive': [2, -5], 'Support': [5, -0.1]}
            merged = lattice.apply_material_offsets(offsets)
            
            # Using list (order matches get_all_materials())
            offsets = [[0.1, 0.5], [2, -5], [5, -0.1]]
            merged = lattice.apply_material_offsets(offsets)
        """
        # Get individual material lattices
        material_lattices = self.get_lattices_by_material()
        
        # Convert list to dict if necessary
        if isinstance(offsets, list):
            materials = sorted(material_lattices.keys())  # Consistent ordering
            if len(offsets) != len(materials):
                raise ValueError(f"Number of offsets ({len(offsets)}) does not match "
                               f"number of materials ({len(materials)})")
            offsets = {mat: offset for mat, offset in zip(materials, offsets)}
        
        # Apply offsets to each material lattice
        offset_lattices = {}
        for material, mat_lattice in material_lattices.items():
            if material in offsets:
                offset_x, offset_y = offsets[material]
                mat_lattice.apply_offset(offset_x, offset_y)
            offset_lattices[material] = mat_lattice
        
        # Merge all offset lattices
        merged_lattice = Lattice.merge_lattices(list(offset_lattices.values()))
        
        return merged_lattice
    
    @staticmethod
    def merge_lattices(lattices):
        """
        Merges multiple Lattice instances into a single lattice.
        Handles node ID conflicts and preserves material information.
        
        Args:
            lattices (list): List of Lattice instances to merge
            
        Returns:
            Lattice: New merged lattice containing all input lattices
            
        Example:
            lattice1 = Lattice(...)  # Material 'Active'
            lattice2 = Lattice(...)  # Material 'Inactive'
            merged = Lattice.merge_lattices([lattice1, lattice2])
        """
        if not lattices:
            return Lattice()
        
        # Create new empty lattice
        merged = Lattice()
        
        # Copy metadata from first lattice
        merged.unit_cell = lattices[0].unit_cell
        merged.n_x = lattices[0].n_x
        merged.n_y = lattices[0].n_y
        merged.tes_x = lattices[0].tes_x
        merged.tes_y = lattices[0].tes_y
        
        # Merge color maps from all lattices
        for lat in lattices:
            merged._color_map.update(lat._color_map)
        
        # Track global node ID
        next_node_id = 0
        
        # Process each lattice
        for lattice in lattices:
            # Map old node IDs to new node IDs for this lattice
            node_mapping = {}
            
            # Add all nodes from this lattice
            for old_node, data in lattice.graph.nodes(data=True):
                # Check if a node at this position already exists in merged graph
                pos_key = (round(data['x'], 6), round(data['y'], 6))
                
                # Search for existing node at same position
                existing_node = None
                for node, node_data in merged.graph.nodes(data=True):
                    if node_data.get('pos') == pos_key:
                        existing_node = node
                        break
                
                if existing_node is not None:
                    # Reuse existing node
                    node_mapping[old_node] = existing_node
                else:
                    # Add new node
                    node_mapping[old_node] = next_node_id
                    merged.graph.add_node(next_node_id,
                                         x=data['x'],
                                         y=data['y'],
                                         pos=pos_key)
                    next_node_id += 1
            
            # Add all edges from this lattice with remapped node IDs
            for u, v, data in lattice.graph.edges(data=True):
                new_u = node_mapping[u]
                new_v = node_mapping[v]
                
                # Check if edge already exists
                if merged.graph.has_edge(new_u, new_v):
                    # Edge exists - check if it's the same material
                    existing_material = merged.graph[new_u][new_v]['material']
                    new_material = data['material']
                    if existing_material != new_material:
                        print(f"Warning: Edge ({new_u}, {new_v}) exists with different material. "
                              f"Keeping {existing_material}, ignoring {new_material}")
                else:
                    # Add new edge
                    merged.graph.add_edge(new_u, new_v,
                                         material=data['material'],
                                         weight=data['weight'])
        
        return merged
    
    def get_material_offsets_dict(self, offsets_list):
        """
        Helper method to convert offset list to dictionary format.
        
        Args:
            offsets_list (list): List of [x, y] offsets
            
        Returns:
            dict: Dictionary mapping material names to offsets
        """
        materials = sorted(self.get_all_materials())
        if len(offsets_list) != len(materials):
            raise ValueError(f"Number of offsets ({len(offsets_list)}) does not match "
                            f"number of materials ({len(materials)})")
        return {mat: offset for mat, offset in zip(materials, offsets_list)}
    
    def get_all_materials(self):
        """
        Returns a sorted list of all material types in the lattice.
        
        Returns:
            list: Sorted list of material names
        """
        materials = set(data['material'] for _, _, data in self.graph.edges(data=True))
        return sorted(materials)
    
    def clone(self):
        """
        Creates a deep copy of the lattice.
        
        Returns:
            Lattice: New lattice instance with copied data
        """
        new_lattice = Lattice()
        
        # Copy attributes
        new_lattice.unit_cell = self.unit_cell  # Reference copy (unit cell is immutable in usage)
        new_lattice.n_x = self.n_x
        new_lattice.n_y = self.n_y
        new_lattice.tes_x = self.tes_x
        new_lattice.tes_y = self.tes_y
        new_lattice._color_map = self._color_map.copy()
        
        # Deep copy graphs
        new_lattice.graph = self.graph.copy()
        
        return new_lattice

# Store in a single graph the lattice after applying nozzle offsets to each material, with multi-material edges allowed
class MixedLattice:
    def __init__(self, color_map=None):
        """
        Initialize a MixedLattice that supports multiple materials per edge.
        
        Args:
            color_map (dict): Optional color mapping for materials
        """
        # NetworkX graph to store mixed lattice structure
        self.graph = nx.Graph()  # Main graph with multi-material edges
        
        # Color map for materials
        self._color_map = color_map if color_map is not None else self._generate_color_palette()
            
    def _generate_color_palette(self):
        """
        Generates a diverse palette of visually distinct colors.
        Uses a combination of qualitative colormaps and custom colors.
        """
        # Start with tableau colors (10 distinct colors)
        tableau = list(mcolors.TABLEAU_COLORS.values())
        
        # Add CSS4 colors that are visually distinct
        extra_colors = [
            'gold', 'crimson', 'darkviolet', 'lime', 'deepskyblue',
            'orangered', 'mediumspringgreen', 'hotpink', 'navy', 'chocolate',
            'teal', 'coral', 'indigo', 'yellowgreen', 'tomato',
            'steelblue', 'orchid', 'sienna', 'lightseagreen', 'salmon'
        ]
        
        # Combine and return
        return tableau + extra_colors
    
    @staticmethod
    def from_lattice_with_offsets(lattice, offsets):
        """
        Creates a MixedLattice from a Lattice by splitting materials, applying offsets, and merging.
        
        Args:
            lattice (Lattice): Source lattice
            offsets (dict): {material_name: [x_offset, y_offset], ...}
            
        Returns:
            MixedLattice: New mixed lattice with multi-material edges
            
        Example:
            mixed = MixedLattice.from_lattice_with_offsets(
                lattice, 
                {'Active': [0, 0], 'Inactive': [2, 1]}
            )
        """
        # Create new MixedLattice
        mixed = MixedLattice(color_map=lattice._color_map.copy())
        
        # Get individual material lattices
        material_lattices = lattice.get_lattices_by_material()
        
        # Convert list to dict if necessary
        if isinstance(offsets, list):
            materials = sorted(material_lattices.keys())
            if len(offsets) != len(materials):
                raise ValueError(f"Number of offsets ({len(offsets)}) does not match "
                               f"number of materials ({len(materials)})")
            offsets = {mat: offset for mat, offset in zip(materials, offsets)}
        
        # Apply offsets to each material lattice
        offset_lattices = {}
        for material, mat_lattice in material_lattices.items():
            if material in offsets:
                offset_x, offset_y = offsets[material]
                mat_lattice.apply_offset(offset_x, offset_y)
            offset_lattices[material] = mat_lattice
        
        # Merge into MixedLattice with multi-material edge support
        mixed._merge_material_lattices(offset_lattices)
        
        
        return mixed
    
    def _merge_material_lattices(self, material_lattices):
        """
        Merges multiple material-specific lattices into a single graph.
        Edges at the same location from different materials are combined into multi-material edges.
        
        Args:
            material_lattices (dict): {material_name: Lattice, ...}
        """
        # Track global node ID
        next_node_id = 0
        # Map position to node ID
        pos_to_node = {}
        
        # First pass: Add all nodes
        for material, lattice in material_lattices.items():
            for node, data in lattice.graph.nodes(data=True):
                pos_key = (round(data['x'], 6), round(data['y'], 6))
                
                if pos_key not in pos_to_node:
                    # Add new node
                    pos_to_node[pos_key] = next_node_id
                    self.graph.add_node(next_node_id,
                                       x=data['x'],
                                       y=data['y'],
                                       pos=pos_key)
                    next_node_id += 1
        
        # Second pass: Add edges with material tracking
        # Track edges: {(node1, node2): [materials]}
        edge_materials = {}
        
        for material, lattice in material_lattices.items():
            for u, v, data in lattice.graph.edges(data=True):
                # Get positions of endpoints
                u_data = lattice.graph.nodes[u]
                v_data = lattice.graph.nodes[v]
                
                u_pos = (round(u_data['x'], 6), round(u_data['y'], 6))
                v_pos = (round(v_data['x'], 6), round(v_data['y'], 6))
                
                # Get node IDs in merged graph
                u_id = pos_to_node[u_pos]
                v_id = pos_to_node[v_pos]
                
                # Canonical edge representation (smaller ID first)
                edge_key = tuple(sorted([u_id, v_id]))
                
                # Add material to this edge
                if edge_key not in edge_materials:
                    edge_materials[edge_key] = {
                        'materials': [],
                        'weight': data['weight']
                    }
                
                edge_materials[edge_key]['materials'].append(material)
        
        # Add edges to graph with material lists
        for (u, v), edge_data in edge_materials.items():
            self.graph.add_edge(u, v,
                               materials=edge_data['materials'],  # LIST of materials
                               weight=edge_data['weight'])
     
    def get_all_materials(self):
        """
        Returns a sorted list of all material types in the mixed lattice.
        
        Returns:
            list: Sorted list of unique material names
        """
        materials = set()
        for _, _, data in self.graph.edges(data=True):
            materials.update(data['materials'])
        return sorted(materials)
    
    def get_stats(self):
        """
        Returns statistics about the mixed lattice.
        
        Returns:
            dict: Dictionary with lattice statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_components': nx.number_connected_components(self.graph),
            'materials': {},
            'multi_material_edges': 0,
            'single_material_edges': 0
        }
        
        # Count edges per material and multi-material edges
        for u, v, data in self.graph.edges(data=True):
            materials = data['materials']
            
            if len(materials) > 1:
                stats['multi_material_edges'] += 1
            else:
                stats['single_material_edges'] += 1
            
            for material in materials:
                if material not in stats['materials']:
                    stats['materials'][material] = 0
                stats['materials'][material] += 1
        
        return stats
    
    def visualize(self, save_path=None, show_nodes=False, show_labels=False,
                  materials=None, title='Mixed Lattice', highlight_mixed=True,fontsize = 8):
        """
        Plots the mixed lattice with colors based on material type.
        Multi-material edges can be highlighted.
        
        Args:
            save_path (str): Path to save the figure
            show_nodes (bool): Whether to show node markers
            show_labels (bool): Whether to show node labels
            materials (list): List of materials to display (None = all)
            title (str): Plot title
            highlight_mixed (bool): Highlight multi-material edges with dashed lines
        """
        fig = plt.figure(figsize=(2.36,2.36),dpi=300)
        
        # Separate single-material and multi-material edges
        single_mat_edges = []
        multi_mat_edges = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_materials = data['materials']
            
            # Filter by requested materials
            if materials is not None:
                if not any(mat in materials for mat in edge_materials):
                    continue
            
            if len(edge_materials) == 1:
                single_mat_edges.append((u, v, edge_materials[0]))
            else:
                multi_mat_edges.append((u, v, edge_materials))
        
        # Plot single-material edges
        for u, v, material in single_mat_edges:
            color = self._color_map.get(material, 'blue')
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            plt.plot([u_data['x'], v_data['x']], 
                    [u_data['y'], v_data['y']], 
                    color=color, linewidth=1.5, linestyle='-',
                    label=material, alpha=0.8)
        
        # Plot multi-material edges
        for u, v, edge_materials in multi_mat_edges:
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            
            if highlight_mixed:
                # Use dashed line and blend colors
                if len(edge_materials) == 2:
                    # Blend two colors
                    import matplotlib.colors as mcolors
                    c1 = mcolors.to_rgb(self._color_map.get(edge_materials[0], 'blue'))
                    c2 = mcolors.to_rgb(self._color_map.get(edge_materials[1], 'red'))
                    blended = tuple((c1[i] + c2[i]) / 2 for i in range(3))
                    color = blended
                else:
                    color = 'purple'  # Default for 3+ materials
                
                plt.plot([u_data['x'], v_data['x']], 
                        [u_data['y'], v_data['y']], 
                        color=color, linewidth=1.5, linestyle='--',
                        label=f"Mixed: {', '.join(edge_materials)}", alpha=0.8)
            else:
                # Draw multiple overlapping lines
                for i, material in enumerate(edge_materials):
                    color = self._color_map.get(material, 'blue')
                    offset = (i - len(edge_materials)/2) * 0.1
                    plt.plot([u_data['x'] + offset, v_data['x'] + offset], 
                            [u_data['y'] + offset, v_data['y'] + offset], 
                            color=color, linewidth=2, alpha=0.8)
        
        # Plot nodes if requested
        if show_nodes:
            for node, data in self.graph.nodes(data=True):
                plt.scatter(data['x'], data['y'], s=30, c='black', zorder=5)
                if show_labels:
                    plt.text(data['x'], data['y'], str(node), 
                            fontsize=fontsize, ha='center', va='bottom')
        
       
        # Add legend (remove duplicates)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.title(title)
        plt.axis('equal')
        # plt.grid(True, alpha=0.8)
        plt.tight_layout()
        plt.show()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight',dpi=300)
        
    def export_data(self, base_path=f'{foldername}/data/'):
        """
        Exports mixed lattice structure to JSON files.
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Export edges with multi-material info
        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            
            edges_data.append({
                'nodes': [u, v],
                'start': [u_data['x'], u_data['y']],
                'end': [v_data['x'], v_data['y']],
                'materials': data['materials'],
                'weight': data['weight']
            })
        
        with open(os.path.join(base_path, 'mixed_lattice_edges.json'), 'w') as f:
            json.dump(edges_data, f, indent=2)
        
        print(f"Exported {len(edges_data)} edges to mixed_lattice_edges.json")
    
    def __repr__(self):
        """String representation of the MixedLattice."""
        stats = self.get_stats()
        
        return (f"MixedLattice(nodes={stats['num_nodes']}, "
                f"edges={stats['num_edges']}, "
                f"materials={list(stats['materials'].keys())}")

# Solver for the Rural Postman Problem on MixedLattice
class RuralPostmanSolver:
    def __init__(self, mixed_lattice):
        self.original_lattice = mixed_lattice
        self.working_graph = nx.MultiGraph(mixed_lattice.graph)
        self.route = []
        self.directionalty = None  
        
    def solve(self, use_mst=False, start_coords=None, debug=False):
        """
        Args:
            use_mst (bool): 
                If True, connects islands via MST.
                If False, connects islands via TSP + 3-Opt.
            start_coords (list or tuple): [x, y] coordinates for the print head start position.
                                          If None, defaults to the lattice minimum boundary.
            debug (bool): If True, generate debug visualizations.
        """
        if debug: print(f"--- Starting RPP Solver (Sort={use_mst}, Open Path Optimized) ---")
        
        # 1. Process graph structure based on use_mst
        if use_mst:
            # Option B: MST-first RPP (Frederickson's algorithm)
            
            # A. MST Connection First
            if not nx.is_connected(self.working_graph):
                if debug: print("Connecting components via MST first...")
                self._connect_components_mst()
            
            # B. Identify Odd Nodes
            odd_nodes = [v for v, d in self.working_graph.degree() if d % 2 == 1]
            
            # C. Perfect Matching Second
            if odd_nodes:
                if debug: print("Matching remaining odd nodes...")
                self._apply_biased_matching(odd_nodes, penalty_weight=0.0)
        else:
            # Default Algorithm Flow (Local Matching -> Swapping -> PCA Connection)
            
            # 1. Identify Odd Nodes
            odd_nodes = [v for v, d in self.working_graph.degree() if d % 2 == 1]
            
            # 2. Local Matching
            if odd_nodes:
                self._apply_biased_matching(odd_nodes, penalty_weight=0.0)
            
            if debug:
                self.visualize_matching_phase(
                    save_path=foldername + "/figs/debug_step2_matching.svg",
                    title="Step 2 Result: Local Matching & Disconnected Islands"
                )
            
            # 3. Patching/Swapping
            if not nx.is_connected(self.working_graph):
                if debug: print("Attempting to merge via edge swapping...")
                self._merge_components_by_swapping(debug=debug,startPos=start_coords)
                
                if debug:
                    self.visualize_after_swapping(
                        save_path=foldername + "/figs/debug_step3_after_swapping.svg",
                        title="Step 3 Result: After Edge Swapping"
                    )
                
            # 4. Final Connection for remaining components
            if not nx.is_connected(self.working_graph):
                if debug: print("Connecting remaining components via pca...")
                self._connect_components_pca(start_coords=start_coords, debug=debug)
        
        if debug:
            self.visualize_graph_structure(
                save_path=foldername + "/figs/debug_connected_graph.svg",
                title="Debug: Connected Graph Structure (Before Eulerian Path)"
            )
        
        # 5. Eulerian Circuit (Closed Loop)
        if debug: print("Generating Eulerian Circuit...")
        try:
            temp_start = min(self.working_graph.nodes, key=lambda n: self.working_graph.nodes[n]['x'])
            raw_circuit = list(nx.eulerian_circuit(self.working_graph, source=temp_start))
        except nx.NetworkXError as e:
            if debug: print(f"Error: Graph not Eulerian. {e}")
            return []

        # 6. Convert to Data-Rich Route
        closed_route = self._process_route(raw_circuit)
        
        if debug:
            self.visualize_intermediate_route(
                closed_route, 
                save_path=foldername+"/figs/debug_eulerian_closed.svg", 
                title="Debug: Closed Eulerian Circuit (Before Open Optimization)"
            )
        
        # 7. Optimize for Open Path (Rotate & Cut)
        if start_coords is None:
            start_coords = [0, 0] 
            
        self.route = self._optimize_open_path(closed_route, start_coords,debug=debug)

        # 8. Calculate Statistics
        self.stats = {
            'total_steps': len(self.route),
            'num_jumps': 0,          
            'jump_distance': 0.0,    
            'print_distance': 0.0,
            'total_distance': 0.0
        }
        
        epsilon = 1e-6
        
        for step in self.route:
            dist = step['weight']
            self.stats['total_distance'] += dist
            
            if step['type'] == 'travel':
                self.stats['jump_distance'] += dist
                
                is_entry = (step.get('u') == -1)
                if is_entry:
                    if dist > epsilon:
                        self.stats['num_jumps'] += 1
                else:
                    self.stats['num_jumps'] += 1
            else:
                self.stats['print_distance'] += dist

        print("="*45)
        print(f"OPTIMIZATION RESULTS")
        print("="*45)
        print(f"  Total Steps:       {self.stats['total_steps']}")
        print(f"  Number of Jumps:   {self.stats['num_jumps']}")
        print(f"  Total Jump Dist:   {self.stats['jump_distance']:.4f} (mm)")
        print(f"  Total Print Dist:  {self.stats['print_distance']:.4f} (mm)")
        print(f"  Total Path Dist:   {self.stats['total_distance']:.4f} (mm)")
        print("="*45)

        return self.route

    def solve_random(self, seed=None, start_coords=None, debug=False):
        """
        Generates a randomized toolpath as a baseline / comparison to solve().

        Print edges from the underlying MixedLattice are placed in a uniformly
        random order. For each edge in the sequence, the orientation (u->v or
        v->u) is chosen greedily to minimize the gap to the previous endpoint.
        When a gap remains, a travel edge is inserted. An entry travel from
        start_coords to the first edge is prepended (matching solve()).

        The result is stored in self.route in the same format as solve(),
        so visualize_route() and ToolpathVisualizer keep working.

        Args:
            seed (int, optional): Seed for reproducible randomization. Uses a
                local numpy Generator so global RNG state is unaffected.
            start_coords (list, optional): [x, y] entry position. Defaults to [0, 0].
            debug (bool): If True, print diagnostic info.

        Returns:
            list: Route in the same format as solve().
        """
        if start_coords is None:
            start_coords = [0.0, 0.0]

        rng = np.random.default_rng(seed)

        # Collect print edges from the original (non-mutated) lattice graph.
        # Deterministic ordering before shuffling so seed -> identical output.
        src_edges = sorted(
            self.original_lattice.graph.edges(data=True),
            key=lambda e: (min(e[0], e[1]), max(e[0], e[1]))
        )
        print_edges = [
            {
                'u': u, 'v': v,
                'materials': list(data.get('materials', [])),
                'weight': data['weight'],
            }
            for u, v, data in src_edges
        ]

        # Empty lattice -- nothing to do.
        if not print_edges:
            self.route = []
            self.stats = {
                'total_steps': 0, 'num_jumps': 0,
                'jump_distance': 0.0, 'print_distance': 0.0, 'total_distance': 0.0,
            }
            return self.route

        # Shuffle.
        order = rng.permutation(len(print_edges))
        shuffled = [print_edges[i] for i in order]

        if debug:
            print(f"--- solve_random: shuffled {len(shuffled)} print edges (seed={seed}) ---")

        def coord_of(n):
            d = self.original_lattice.graph.nodes[n]
            return np.array([d['x'], d['y']])

        epsilon = 1e-9
        route = []

        # Entry: travel from user start_coords to the first print edge.
        # u = -1 is the sentinel used elsewhere to mark the entry jump.
        first = shuffled[0]
        p_u = coord_of(first['u'])
        p_v = coord_of(first['v'])
        start_p = np.array(start_coords, dtype=float)

        if np.linalg.norm(start_p - p_u) <= np.linalg.norm(start_p - p_v):
            cur_start_node, cur_end_node = first['u'], first['v']
            cur_start_p, cur_end_p = p_u, p_v
        else:
            cur_start_node, cur_end_node = first['v'], first['u']
            cur_start_p, cur_end_p = p_v, p_u

        entry_dist = float(np.linalg.norm(start_p - cur_start_p))
        route.append({
            'u': -1, 'v': cur_start_node,
            'coords_start': [float(start_p[0]), float(start_p[1])],
            'coords_end': [float(cur_start_p[0]), float(cur_start_p[1])],
            'type': 'travel',
            'materials': [],
            'weight': entry_dist,
        })
        route.append({
            'u': cur_start_node, 'v': cur_end_node,
            'coords_start': [float(cur_start_p[0]), float(cur_start_p[1])],
            'coords_end': [float(cur_end_p[0]), float(cur_end_p[1])],
            'type': 'print',
            'materials': list(first['materials']),
            'weight': first['weight'],
        })

        current_pos = cur_end_p
        current_node = cur_end_node

        # Remaining print edges.
        for edge in shuffled[1:]:
            u, v = edge['u'], edge['v']
            p_u = coord_of(u)
            p_v = coord_of(v)

            d_to_u = np.linalg.norm(current_pos - p_u)
            d_to_v = np.linalg.norm(current_pos - p_v)

            if d_to_u <= d_to_v:
                s_node, e_node = u, v
                s_p, e_p = p_u, p_v
                gap = float(d_to_u)
            else:
                s_node, e_node = v, u
                s_p, e_p = p_v, p_u
                gap = float(d_to_v)

            if gap > epsilon:
                route.append({
                    'u': current_node, 'v': s_node,
                    'coords_start': [float(current_pos[0]), float(current_pos[1])],
                    'coords_end': [float(s_p[0]), float(s_p[1])],
                    'type': 'travel',
                    'materials': [],
                    'weight': gap,
                })

            route.append({
                'u': s_node, 'v': e_node,
                'coords_start': [float(s_p[0]), float(s_p[1])],
                'coords_end': [float(e_p[0]), float(e_p[1])],
                'type': 'print',
                'materials': list(edge['materials']),
                'weight': edge['weight'],
            })

            current_pos = e_p
            current_node = e_node

        self.route = route

        # Stats (mirror solve() so downstream visualizers / loggers match).
        self.stats = {
            'total_steps': len(self.route),
            'num_jumps': 0,
            'jump_distance': 0.0,
            'print_distance': 0.0,
            'total_distance': 0.0,
        }
        eps_count = 1e-6
        for step in self.route:
            dist = step['weight']
            self.stats['total_distance'] += dist
            if step['type'] == 'travel':
                self.stats['jump_distance'] += dist
                is_entry = (step.get('u') == -1)
                if is_entry:
                    if dist > eps_count:
                        self.stats['num_jumps'] += 1
                else:
                    self.stats['num_jumps'] += 1
            else:
                self.stats['print_distance'] += dist

        print("=" * 45)
        print("RANDOM PATH RESULTS")
        print("=" * 45)
        print(f"  Total Steps:       {self.stats['total_steps']}")
        print(f"  Number of Jumps:   {self.stats['num_jumps']}")
        print(f"  Total Jump Dist:   {self.stats['jump_distance']:.4f} (mm)")
        print(f"  Total Print Dist:  {self.stats['print_distance']:.4f} (mm)")
        print(f"  Total Path Dist:   {self.stats['total_distance']:.4f} (mm)")
        print("=" * 45)

        return self.route

    def solve_greedy(self, start_coords=None, debug=False):
        """
        Generates a greedy nearest-edge toolpath as a baseline / comparison to solve().

        From start_coords, repeatedly picks the unvisited print edge whose closer
        endpoint is nearest to the current pen position, traverses it starting
        from that endpoint, and inserts a travel edge when a positional gap
        remains. Ties between equally-close edges are broken by the deterministic
        (min(u,v), max(u,v)) ordering of the underlying lattice edges.

        The result is stored in self.route in the same format as solve(), so
        visualize_route() and ToolpathVisualizer keep working.

        Args:
            start_coords (list, optional): [x, y] entry position. Defaults to [0, 0].
            debug (bool): If True, print diagnostic info.

        Returns:
            list: Route in the same format as solve().
        """
        if start_coords is None:
            start_coords = [0.0, 0.0]

        # Collect print edges with deterministic ordering so ties resolve consistently.
        src_edges = sorted(
            self.original_lattice.graph.edges(data=True),
            key=lambda e: (min(e[0], e[1]), max(e[0], e[1]))
        )
        print_edges = [
            {
                'u': u, 'v': v,
                'materials': list(data.get('materials', [])),
                'weight': data['weight'],
            }
            for u, v, data in src_edges
        ]

        if not print_edges:
            self.route = []
            self.stats = {
                'total_steps': 0, 'num_jumps': 0,
                'jump_distance': 0.0, 'print_distance': 0.0, 'total_distance': 0.0,
            }
            return self.route

        def coord_of(n):
            d = self.original_lattice.graph.nodes[n]
            return np.array([d['x'], d['y']])

        epsilon = 1e-9
        route = []
        remaining = list(range(len(print_edges)))

        def pick_nearest(pos):
            """Return (list_index_in_remaining, s_node, e_node, s_p, e_p, gap).

            Tie-break uses the natural order of `remaining` (which inherits the
            sorted (min,max) ordering of print_edges).
            """
            best_idx = None
            best_gap = None
            best_orient = None
            for idx, ei in enumerate(remaining):
                e = print_edges[ei]
                p_u = coord_of(e['u'])
                p_v = coord_of(e['v'])
                d_u = float(np.linalg.norm(pos - p_u))
                d_v = float(np.linalg.norm(pos - p_v))
                if d_u <= d_v:
                    gap = d_u
                    orient = (e['u'], e['v'], p_u, p_v)
                else:
                    gap = d_v
                    orient = (e['v'], e['u'], p_v, p_u)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_idx = idx
                    best_orient = orient
            return best_idx, best_orient, best_gap

        # Entry: from start_coords pick the nearest endpoint among all edges.
        start_p = np.array(start_coords, dtype=float)
        idx, (s_node, e_node, s_p, e_p), entry_dist = pick_nearest(start_p)
        first = print_edges[remaining.pop(idx)]

        route.append({
            'u': -1, 'v': s_node,
            'coords_start': [float(start_p[0]), float(start_p[1])],
            'coords_end': [float(s_p[0]), float(s_p[1])],
            'type': 'travel',
            'materials': [],
            'weight': float(entry_dist),
        })
        route.append({
            'u': s_node, 'v': e_node,
            'coords_start': [float(s_p[0]), float(s_p[1])],
            'coords_end': [float(e_p[0]), float(e_p[1])],
            'type': 'print',
            'materials': list(first['materials']),
            'weight': first['weight'],
        })

        current_pos = e_p
        current_node = e_node

        if debug:
            print(f"--- solve_greedy: {len(print_edges)} print edges, "
                  f"start={start_coords} ---")

        # Greedy traversal of the rest.
        while remaining:
            idx, (s_node, e_node, s_p, e_p), gap = pick_nearest(current_pos)
            edge = print_edges[remaining.pop(idx)]

            if gap > epsilon:
                route.append({
                    'u': current_node, 'v': s_node,
                    'coords_start': [float(current_pos[0]), float(current_pos[1])],
                    'coords_end': [float(s_p[0]), float(s_p[1])],
                    'type': 'travel',
                    'materials': [],
                    'weight': float(gap),
                })

            route.append({
                'u': s_node, 'v': e_node,
                'coords_start': [float(s_p[0]), float(s_p[1])],
                'coords_end': [float(e_p[0]), float(e_p[1])],
                'type': 'print',
                'materials': list(edge['materials']),
                'weight': edge['weight'],
            })

            current_pos = e_p
            current_node = e_node

        self.route = route

        self.stats = {
            'total_steps': len(self.route),
            'num_jumps': 0,
            'jump_distance': 0.0,
            'print_distance': 0.0,
            'total_distance': 0.0,
        }
        eps_count = 1e-6
        for step in self.route:
            dist = step['weight']
            self.stats['total_distance'] += dist
            if step['type'] == 'travel':
                self.stats['jump_distance'] += dist
                is_entry = (step.get('u') == -1)
                if is_entry:
                    if dist > eps_count:
                        self.stats['num_jumps'] += 1
                else:
                    self.stats['num_jumps'] += 1
            else:
                self.stats['print_distance'] += dist

        print("=" * 45)
        print("GREEDY PATH RESULTS")
        print("=" * 45)
        print(f"  Total Steps:       {self.stats['total_steps']}")
        print(f"  Number of Jumps:   {self.stats['num_jumps']}")
        print(f"  Total Jump Dist:   {self.stats['jump_distance']:.4f} (mm)")
        print(f"  Total Print Dist:  {self.stats['print_distance']:.4f} (mm)")
        print(f"  Total Path Dist:   {self.stats['total_distance']:.4f} (mm)")
        print("=" * 45)

        return self.route

    def solve_line(self, start_coords=None, angle_tol_deg=5.0, debug=False):
        """
        Greedy toolpath that prefers continuing in a straight line.

        After printing an edge A->B, the next pick is biased toward an unvisited
        edge B->C whose direction is (within angle_tol_deg) parallel to A->B and
        starts at the current node B (so the segment continues straight, with no
        travel). When no straight continuation exists, falls back to the same
        nearest-endpoint rule used by solve_greedy.

        The result is stored in self.route in the same format as solve(), so
        visualize_route() and ToolpathVisualizer keep working.

        Args:
            start_coords (list, optional): [x, y] entry position. Defaults to [0, 0].
            angle_tol_deg (float): Maximum angle between consecutive edge directions
                (in degrees) to count as "straight". Defaults to 5 degrees.
            debug (bool): If True, print diagnostic info.

        Returns:
            list: Route in the same format as solve().
        """
        if start_coords is None:
            start_coords = [0.0, 0.0]

        src_edges = sorted(
            self.original_lattice.graph.edges(data=True),
            key=lambda e: (min(e[0], e[1]), max(e[0], e[1]))
        )
        print_edges = [
            {
                'u': u, 'v': v,
                'materials': list(data.get('materials', [])),
                'weight': data['weight'],
            }
            for u, v, data in src_edges
        ]

        if not print_edges:
            self.route = []
            self.stats = {
                'total_steps': 0, 'num_jumps': 0,
                'jump_distance': 0.0, 'print_distance': 0.0, 'total_distance': 0.0,
            }
            return self.route

        def coord_of(n):
            d = self.original_lattice.graph.nodes[n]
            return np.array([d['x'], d['y']])

        epsilon = 1e-9
        cos_tol = float(np.cos(np.radians(angle_tol_deg)))
        route = []
        remaining = list(range(len(print_edges)))

        def pick_nearest(pos):
            """Greedy nearest-endpoint pick (matches solve_greedy)."""
            best_idx = None
            best_gap = None
            best_orient = None
            for idx, ei in enumerate(remaining):
                e = print_edges[ei]
                p_u = coord_of(e['u'])
                p_v = coord_of(e['v'])
                d_u = float(np.linalg.norm(pos - p_u))
                d_v = float(np.linalg.norm(pos - p_v))
                if d_u <= d_v:
                    gap = d_u
                    orient = (e['u'], e['v'], p_u, p_v)
                else:
                    gap = d_v
                    orient = (e['v'], e['u'], p_v, p_u)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_idx = idx
                    best_orient = orient
            return best_idx, best_orient, best_gap

        def pick_straight(current_node, prev_dir):
            """Among unvisited edges incident to current_node, pick the one
            whose outgoing direction best aligns with prev_dir (cos similarity
            >= cos_tol). Returns (idx_in_remaining, s_node, e_node, s_p, e_p)
            or None when no straight continuation exists.

            prev_dir must be a unit vector (caller normalizes).
            """
            best_idx = None
            best_cos = cos_tol  # require at least this much alignment
            best_orient = None
            for idx, ei in enumerate(remaining):
                e = print_edges[ei]
                u, v = e['u'], e['v']
                if u == current_node:
                    s_node, e_node = u, v
                elif v == current_node:
                    s_node, e_node = v, u
                else:
                    continue
                s_p = coord_of(s_node)
                e_p = coord_of(e_node)
                d = e_p - s_p
                n = float(np.linalg.norm(d))
                if n < epsilon:
                    continue
                d_hat = d / n
                cos_sim = float(np.dot(prev_dir, d_hat))
                if cos_sim > best_cos:
                    best_cos = cos_sim
                    best_idx = idx
                    best_orient = (s_node, e_node, s_p, e_p)
            if best_idx is None:
                return None
            return best_idx, best_orient

        # Entry: same rule as solve_greedy -- pick the edge with the closest
        # endpoint to start_coords. No previous direction yet.
        start_p = np.array(start_coords, dtype=float)
        idx, (s_node, e_node, s_p, e_p), entry_dist = pick_nearest(start_p)
        first = print_edges[remaining.pop(idx)]

        route.append({
            'u': -1, 'v': s_node,
            'coords_start': [float(start_p[0]), float(start_p[1])],
            'coords_end': [float(s_p[0]), float(s_p[1])],
            'type': 'travel',
            'materials': [],
            'weight': float(entry_dist),
        })
        route.append({
            'u': s_node, 'v': e_node,
            'coords_start': [float(s_p[0]), float(s_p[1])],
            'coords_end': [float(e_p[0]), float(e_p[1])],
            'type': 'print',
            'materials': list(first['materials']),
            'weight': first['weight'],
        })

        current_pos = e_p
        current_node = e_node
        prev_dir = (e_p - s_p)
        n0 = float(np.linalg.norm(prev_dir))
        prev_dir = prev_dir / n0 if n0 > epsilon else None

        straight_count = 0
        fallback_count = 0

        while remaining:
            picked = None
            if prev_dir is not None:
                picked = pick_straight(current_node, prev_dir)

            if picked is not None:
                idx, (s_node, e_node, s_p, e_p) = picked
                edge = print_edges[remaining.pop(idx)]
                # Straight continuation starts at current_node -> zero gap.
                straight_count += 1
            else:
                idx, (s_node, e_node, s_p, e_p), gap = pick_nearest(current_pos)
                edge = print_edges[remaining.pop(idx)]
                if gap > epsilon:
                    route.append({
                        'u': current_node, 'v': s_node,
                        'coords_start': [float(current_pos[0]), float(current_pos[1])],
                        'coords_end': [float(s_p[0]), float(s_p[1])],
                        'type': 'travel',
                        'materials': [],
                        'weight': float(gap),
                    })
                fallback_count += 1

            route.append({
                'u': s_node, 'v': e_node,
                'coords_start': [float(s_p[0]), float(s_p[1])],
                'coords_end': [float(e_p[0]), float(e_p[1])],
                'type': 'print',
                'materials': list(edge['materials']),
                'weight': edge['weight'],
            })

            current_pos = e_p
            current_node = e_node
            new_dir = e_p - s_p
            nN = float(np.linalg.norm(new_dir))
            prev_dir = new_dir / nN if nN > epsilon else None

        if debug:
            print(f"--- solve_line: {len(print_edges)} print edges, "
                  f"angle_tol={angle_tol_deg} deg, "
                  f"straight={straight_count}, fallback={fallback_count} ---")

        self.route = route

        self.stats = {
            'total_steps': len(self.route),
            'num_jumps': 0,
            'jump_distance': 0.0,
            'print_distance': 0.0,
            'total_distance': 0.0,
        }
        eps_count = 1e-6
        for step in self.route:
            dist = step['weight']
            self.stats['total_distance'] += dist
            if step['type'] == 'travel':
                self.stats['jump_distance'] += dist
                is_entry = (step.get('u') == -1)
                if is_entry:
                    if dist > eps_count:
                        self.stats['num_jumps'] += 1
                else:
                    self.stats['num_jumps'] += 1
            else:
                self.stats['print_distance'] += dist

        print("=" * 45)
        print("LINE-PRIORITY PATH RESULTS")
        print("=" * 45)
        print(f"  Total Steps:       {self.stats['total_steps']}")
        print(f"  Number of Jumps:   {self.stats['num_jumps']}")
        print(f"  Total Jump Dist:   {self.stats['jump_distance']:.4f} (mm)")
        print(f"  Total Print Dist:  {self.stats['print_distance']:.4f} (mm)")
        print(f"  Total Path Dist:   {self.stats['total_distance']:.4f} (mm)")
        print("=" * 45)

        return self.route

    def _connect_components_pca(self, start_coords=None, debug = False):
        """
        Connects disjoint components in a MEANDERING (Zig-Zag) linear chain.
        Forms a closed loop by connecting Last_Vout -> First_Vin.
        
        Args:
            start_coords (list): [x, y] position to bias the starting direction of the sort.
                                 If None, defaults to [0, 0].

        Sorting Logic:
        1. Group components into 'bins' based on their projection along the Principal Axis.
        2. Sort within bins along the Perpendicular Axis.
        3. Check start_coords against the bounds of the first bin (Bin 0) to decide direction.
           - If start is closer to 'bottom', Sort Bin 0 Ascending (Low->High).
           - If start is closer to 'top', Sort Bin 0 Descending (High->Low).
        4. Alternate direction for subsequent bins.
        """
        from scipy.spatial.distance import cdist

        # 1. Identify Components
        components = list(nx.connected_components(self.working_graph))
        n_comps = len(components)
        if n_comps <= 1:
            return

        if debug: print(f"  > Connecting {n_comps} components via PCA Meandering Sort...")
        if start_coords is None:
            start_coords = [0, 0]

        # 2. Analyze Components
        comp_info = []
        for idx, comp in enumerate(components):
            nodes = list(comp)
            coords = np.array([[self.working_graph.nodes[n]['x'], self.working_graph.nodes[n]['y']] for n in nodes])
            centroid = np.mean(coords, axis=0)
            
            # Find internal travel edges
            subgraph = self.working_graph.subgraph(nodes)
            travel_edges = []
            for u, v, k, d in subgraph.edges(keys=True, data=True):
                if d.get('type') == 'travel':
                    travel_edges.append((u, v, k, d['weight']))
            
            comp_info.append({
                'id': idx,
                'nodes': nodes,
                'coords': coords,
                'centroid': centroid,
                'is_natural': (len(travel_edges) == 0),
                'travel_edges': travel_edges
            })

        # 3. PCA & Meandering Sort Setup
        centroids = np.array([c['centroid'] for c in comp_info])
        
        # A. Find Principal Axis (V1)
        mean_vec = np.mean(centroids, axis=0)
        centered = centroids - mean_vec
        
        if np.allclose(centered, 0):
            principal_axis = np.array([1.0, 0.0])
        else:
            cov = np.cov(centered, rowvar=False)
            try:
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                principal_axis = eigenvecs[:, -1]
            except np.linalg.LinAlgError:
                principal_axis = np.array([1.0, 0.0])
        
        # B. Find Secondary Axis (V2) - Perpendicular to V1
        secondary_axis = np.array([-principal_axis[1], principal_axis[0]])
        
        if debug: print(f"principal axis = {principal_axis[0]},{principal_axis[1]}")

        # C. Calculate Bin Width
        from scipy.stats import gaussian_kde
        
        p1_projections = np.dot(centroids, principal_axis)
        sorted_projections = np.sort(p1_projections)
        consecutive_distances = np.diff(sorted_projections)
        
        significant_distances = consecutive_distances[consecutive_distances > 1e-3]
        
        if len(significant_distances) >= 2:
            try:
                kde = gaussian_kde(significant_distances)
                # Sample the KDE to find the mode
                x_range = np.linspace(
                    significant_distances.min(), 
                    significant_distances.max(), 
                    200
                )
                kde_values = kde(x_range)
                avg_width = x_range[np.argmax(kde_values)]
            except np.linalg.LinAlgError:
                avg_width = np.median(significant_distances)
        elif len(significant_distances) == 1:
            avg_width = significant_distances[0]
        else:
            avg_width = 1.0

        # D. Assign Bins and Secondary Scores
        p1_scores = np.dot(centroids, principal_axis)
        p2_scores = np.dot(centroids, secondary_axis)
        
        p1_min = np.min(p1_scores)
        if debug: 
            print(f"------------------{principal_axis}")
            print(f"------------------{p1_min},  {np.max(p1_scores)}, {avg_width}")
            print(f"------------------{p1_scores}")
        for i, c in enumerate(comp_info):
            c['bin_idx'] = int((p1_scores[i] - p1_min) / avg_width)
            c['p2'] = p2_scores[i]
            if debug: print(f"------------------{c['bin_idx']}")

        # --- E. Determine Start Direction ---
        # Analyze Bin 0 to see where 'start_coords' lies relative to it along V2
        bin_0_comps = [c for c in comp_info if c['bin_idx'] == 0]
        
        if bin_0_comps:
            p2_vals_0 = [c['p2'] for c in bin_0_comps]
            min_p2 = min(p2_vals_0)
            max_p2 = max(p2_vals_0)
            
            # Project start_coords onto secondary axis
            # (Note: Dot product gives scalar projection)
            start_p2 = np.dot(np.array(start_coords), secondary_axis)
            
            # Distance to boundaries
            dist_to_min = abs(start_p2 - min_p2)
            dist_to_max = abs(start_p2 - max_p2)
            
            # If closer to min, we want Ascending (Low->High).
            # If closer to max, we want Descending (High->Low).
            start_ascending = (dist_to_min < dist_to_max)
        else:
            start_ascending = True # Fallback

        # F. Execute Sort
        def meandering_key(item):
            b = item['bin_idx']
            val = item['p2']
            
            # Logic:
            # If start_ascending=True:  Bin 0 (Even) -> Ascending (+val)
            #                           Bin 1 (Odd)  -> Descending (-val)
            # If start_ascending=False: Bin 0 (Even) -> Descending (-val)
            #                           Bin 1 (Odd)  -> Ascending (+val)
            
            is_even_bin = (b % 2 == 0)
            
            # XOR Logic for direction:
            # Ascending if (Even AND StartAsc) OR (Odd AND NotStartAsc)
            # Simply: Direction multiplier is 1 if match, -1 if not match
            
            if start_ascending:
                direction = 1 if is_even_bin else -1
            else:
                direction = -1 if is_even_bin else 1
                
            return (b, direction * val)

        comp_info.sort(key=meandering_key)

        # Variables to track connectivity
        chain_start_node = None
        current_tail_node = None
        
        # 4. Connection Logic
        for i in range(n_comps):
            comp = comp_info[i]
            
            # Lookahead Target
            if i == 0:
                if n_comps > 1:
                    target_centroid = comp_info[1]['centroid']
                else:
                    target_centroid = comp['centroid']
            elif i < n_comps - 1:
                target_centroid = comp_info[i+1]['centroid']
            else:
                # Last component targets Chain Start
                if chain_start_node is not None:
                    target_node_data = self.working_graph.nodes[chain_start_node]
                    target_centroid = np.array([target_node_data['x'], target_node_data['y']])
                else:
                    target_centroid = comp['centroid']

            # --- Internal Logic: Define Vin and Vout ---
            vin = None
            vout = None
            
            if comp['is_natural']:
                dists = np.linalg.norm(comp['coords'] - target_centroid, axis=1)
                best_idx = np.argmin(dists)
                vin = comp['nodes'][best_idx]
                vout = vin
                
                if current_tail_node is not None:
                    p_tail = np.array([self.working_graph.nodes[current_tail_node]['x'], 
                                       self.working_graph.nodes[current_tail_node]['y']])
                    d_prev = np.linalg.norm(comp['coords'] - p_tail, axis=1)
                    d_next = np.linalg.norm(comp['coords'] - target_centroid, axis=1)
                    total_dist = d_prev + d_next
                    best_idx = np.argmin(total_dist)
                    vin = comp['nodes'][best_idx]
                    vout = vin

            else:
                p_prev = None
                if current_tail_node is not None:
                    p_prev = np.array([self.working_graph.nodes[current_tail_node]['x'], 
                                       self.working_graph.nodes[current_tail_node]['y']])

                best_edge = None
                best_score = float('inf')
                best_orientation = 0
                
                for u, v, k, w in comp['travel_edges']:
                    p_u = np.array([self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']])
                    p_v = np.array([self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']])
                    
                    d_v_next = np.linalg.norm(p_v - target_centroid)
                    d_u_next = np.linalg.norm(p_u - target_centroid)
                    d_prev_u = np.linalg.norm(p_prev - p_u) if p_prev is not None else 0
                    d_prev_v = np.linalg.norm(p_prev - p_v) if p_prev is not None else 0
                    
                    score_0 = d_prev_u + d_v_next
                    if score_0 < best_score:
                        best_score = score_0
                        best_edge = (u, v, k)
                        best_orientation = 0
                        
                    score_1 = d_prev_v + d_u_next
                    if score_1 < best_score:
                        best_score = score_1
                        best_edge = (u, v, k)
                        best_orientation = 1
                
                if best_edge:
                    u, v, k = best_edge
                    self.working_graph.remove_edge(u, v, key=k)
                    if best_orientation == 0:
                        vin, vout = u, v
                    else:
                        vin, vout = v, u

            # --- Connect Previous to Current ---
            if i == 0:
                chain_start_node = vin
            else:
                p_tail = np.array([self.working_graph.nodes[current_tail_node]['x'], 
                                   self.working_graph.nodes[current_tail_node]['y']])
                p_vin = np.array([self.working_graph.nodes[vin]['x'], 
                                  self.working_graph.nodes[vin]['y']])
                dist = np.linalg.norm(p_tail - p_vin)
                
                self.working_graph.add_edge(current_tail_node, vin, type='travel', materials=[], weight=dist)
                
            current_tail_node = vout

        # --- 5. Close the Loop ---
        if current_tail_node is not None and chain_start_node is not None and current_tail_node != chain_start_node:
            p_last = np.array([self.working_graph.nodes[current_tail_node]['x'], 
                               self.working_graph.nodes[current_tail_node]['y']])
            p_start = np.array([self.working_graph.nodes[chain_start_node]['x'], 
                                self.working_graph.nodes[chain_start_node]['y']])
            dist = np.linalg.norm(p_last - p_start)
            
            self.working_graph.add_edge(current_tail_node, chain_start_node, type='travel', materials=[], weight=dist)
            if debug: print(f"  > Loop Closed (Dist: {dist:.2f})")
     
    def visualize_after_swapping(self, save_path=None, title="Debug: Graph After Swapping/Patching"):
        """
        Visualizes the graph state immediately after _merge_components_by_swapping.
        """
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        
        fig, ax = plt.subplots(figsize=(5,4),dpi=300)
        
        components = list(nx.connected_components(self.working_graph))
        num_components = len(components)
        
        node_to_comp = {}
        for idx, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = idx
        
        cmap = plt.get_cmap('tab20', max(num_components, 1))
        comp_colors = [cmap(i % 20) for i in range(num_components)]
        
        # Plot Nodes
        for node, data in self.working_graph.nodes(data=True):
            comp_idx = node_to_comp[node]
            color = comp_colors[comp_idx]
            ax.scatter(data['x'], data['y'], c=[color], s=30, zorder=5, edgecolors='black', linewidths=0.5)
        
        # Categorize and Plot Edges
        print_edges = []
        matching_edges_original = []
        bridging_edges = []
        
        for u, v, key, data in self.working_graph.edges(keys=True, data=True):
            edge_type = data.get('type', 'print')
            is_matching = data.get('is_matching', False)
            weight = data.get('weight', 0)
            
            u_comp = node_to_comp[u]
            v_comp = node_to_comp[v]
            
            p1 = np.array([self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']])
            p2 = np.array([self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']])
            
            edge_info = {
                'u': u, 'v': v, 'key': key,
                'p1': p1, 'p2': p2,
                'weight': weight,
                'u_comp': u_comp, 'v_comp': v_comp,
                'type': edge_type,
                'is_matching': is_matching
            }
            
            if edge_type == 'print':
                print_edges.append(edge_info)
            elif is_matching:
                if u_comp == v_comp:
                    matching_edges_original.append(edge_info)
                else:
                    bridging_edges.append(edge_info)
            else:
                bridging_edges.append(edge_info)
        
        # Plot Print Edges
        for edge in print_edges:
            color = comp_colors[edge['u_comp']]
            ax.plot([edge['p1'][0], edge['p2'][0]], 
                    [edge['p1'][1], edge['p2'][1]],
                    color=color, linewidth=1.5, alpha=0.8, zorder=1)
        
        # Plot Original Matching Edges
        for edge in matching_edges_original:
            ax.plot([edge['p1'][0], edge['p2'][0]], 
                    [edge['p1'][1], edge['p2'][1]],
                    color='blue', linewidth=1.5, linestyle='--', alpha=0.8, zorder=2)
            
            mid = (edge['p1'] + edge['p2']) / 2
            ax.text(mid[0], mid[1], f"{edge['weight']:.2f}", 
                    fontsize=5, color='blue', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
        
        # Plot Bridging Edges
        for edge in bridging_edges:
            ax.plot([edge['p1'][0], edge['p2'][0]], 
                    [edge['p1'][1], edge['p2'][1]],
                    color='red', linewidth=1.5, linestyle='--', alpha=0.8, zorder=3)
            
            mid = (edge['p1'] + edge['p2']) / 2
            ax.text(mid[0], mid[1], f"{edge['weight']:.2f}", 
                    fontsize=5, color='red', ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', ec='none', alpha=0.8))
            
            ax.text(mid[0], mid[1] - 0.5, f"({edge['u_comp']}↔{edge['v_comp']})", 
                    fontsize=5, color='darkred', ha='center', va='top')
        
        # Plot Component Centroids
        for idx, comp in enumerate(components):
            coords = np.array([[self.working_graph.nodes[n]['x'], self.working_graph.nodes[n]['y']] 
                              for n in comp])
            centroid = np.mean(coords, axis=0)
            
            ax.scatter(centroid[0], centroid[1], 
                      c=[comp_colors[idx]], s=300, marker='s', 
                      edgecolors='black', linewidths=1.5, zorder=10, alpha=0.8)
            ax.text(centroid[0], centroid[1], str(idx), 
                    fontsize=8, fontweight='bold', ha='center', va='center', 
                    color='white', zorder=11)
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.8, label='Print Edges'),
            Line2D([0], [0], color='blue', linewidth=1.5, linestyle='--', label='Matching (intra-component)'),
            Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', label='Bridging (inter-component)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                   markersize=10, markeredgecolor='black', markeredgewidth=0.5, label='Component Centroid'),
        ]
        
        for idx in range(min(num_components, 10)):
            legend_elements.append(
                mpatches.Patch(facecolor=comp_colors[idx], edgecolor='black', 
                              label=f'Component {idx}')
            )
        
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.axis('equal')
        # ax.grid(True, alpha=0.8)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"DEBUG: Saved visualization to {save_path}")
        
        plt.show()
        plt.close(fig)
    
    def visualize_matching_phase(self, save_path=None, title="Step 2 Debug: Local Matching & Islands"):
        """
        Visualizes the graph state immediately after Local Matching.
        """
        from matplotlib.lines import Line2D
        
        components = list(nx.connected_components(self.working_graph))
        
        node_to_comp = {}
        for idx, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = idx
        
        fig, ax = plt.subplots(figsize=(2.36,2.36),dpi=300)
        cmap = plt.get_cmap('tab20', len(components))
        
        # Plot Nodes
        # for node, data in self.working_graph.nodes(data=True):
        #     ax.scatter(data['x'], data['y'], c='black', s=10, zorder=5)

        # Plot Edges
        for u, v, data in self.working_graph.edges(data=True):
            p1 = [self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']]
            p2 = [self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']]
            
            edge_type = data.get('type', 'print')
            
            if edge_type == 'travel':
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)
            else:
                comp_id = node_to_comp[u]
                color = cmap(comp_id % 20)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=color, linestyle='-', linewidth=1.5, alpha=0.8, zorder=2)

        legend_elements = [
            Line2D([0], [0], color='gray', lw=1, label=f'Islands (Total: {len(components)})'),
            Line2D([0], [0], color='red', linestyle='--', lw=1, label='Added Match Jumps')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        # ax.set_title(title)
        
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        ax.axis('equal')
        # ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight',dpi=300)
            print(f"DEBUG: Step 2 visualization saved to {save_path}")
        plt.show()
        plt.close(fig)
    
    def visualize_graph_structure(self, save_path=None, title="Connected Graph Structure"):
        """
        Visualizes the raw graph connectivity (Printing Edges vs. Travel Edges).
        """
        from matplotlib.lines import Line2D
        
        fig, ax = plt.subplots(figsize=(2.36,2.36),dpi=300)
        
        # for node, data in self.working_graph.nodes(data=True):
        #     ax.scatter(data['x'], data['y'], c='black', s=10, zorder=5)

        for u, v, data in self.working_graph.edges(data=True):
            p1 = [self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']]
            p2 = [self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']]
            
            edge_type = data.get('type', 'print')
            
            if edge_type == 'travel':
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
            else:
                materials = data.get('materials', [])
                if materials:
                    color = self.original_lattice._color_map.get(materials[0], 'blue')
                else:
                    color = 'gray'
                    
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=color, linestyle='-', linewidth=1.5, alpha=0.8, zorder=2)

        legend_elements = [
            Line2D([0], [0], color='gray', lw=1, label='Print Edges'),
            Line2D([0], [0], color='red', linestyle='--', lw=1, label='Added Travel/Jumps')
        ]
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        ax.legend(handles=legend_elements, loc='best')
        # ax.set_title(title)
        ax.axis('equal')
        # ax.grid(True, alpha=0.3)
        plt.tight_layout()
            
        if save_path:
            fig.savefig(save_path, bbox_inches='tight',dpi=300)
            print(f"DEBUG: Graph structure visualization saved to {save_path}")
        plt.show()
        plt.close(fig)
    
    def visualize_intermediate_route(self, route_data, save_path=None, title="Intermediate Route"):
        """
        Helper to visualize a specific route list before final optimization.
        """
        if not route_data:
            return

        fig, ax = plt.subplots(figsize=(2.36,2.36),dpi=300)
        
        # for node, data in self.original_lattice.graph.nodes(data=True):
        #     ax.scatter(data['x'], data['y'], c='black', s=15, zorder=3)

        import matplotlib.patches as patches
        
        def get_offset_points(p1, p2, offset_amount):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length == 0: return p1, p2
            
            perp_x = -dy / length
            perp_y = dx / length
            
            return (p1[0] + perp_x * offset_amount, p1[1] + perp_y * offset_amount), \
                   (p2[0] + perp_x * offset_amount, p2[1] + perp_y * offset_amount)

        for i, step in enumerate(route_data):
            p1 = step['coords_start']
            p2 = step['coords_end']
            
            label_x = p1[0] + (p2[0] - p1[0]) / 3.0
            label_y = p1[1] + (p2[1] - p1[1]) / 3.0

            if step['type'] == 'travel':
                arrow = patches.FancyArrowPatch(
                    posA=p1, posB=p2,
                    arrowstyle='-|>', mutation_scale=8, 
                    color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1000,
                    shrinkA=0, shrinkB=0
                )
                ax.add_patch(arrow)
                # ax.text(label_x, label_y, str(i + 1), color='red', fontsize=5, fontweight='bold',
                #         ha='center', va='center', zorder=5,
                #         bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.8))
            else:
                materials = step['materials']
                num_mats = len(materials)
                edge_len = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                offset_base = edge_len * 0.12
                
                for m_idx, mat in enumerate(materials):
                    centering = (num_mats - 1) / 2.0
                    shift = (m_idx - centering) * offset_base
                    sp, ep = get_offset_points(p1, p2, shift)
                    color = self.original_lattice._color_map.get(mat, 'blue')
                    
                    arrow = patches.FancyArrowPatch(
                        posA=sp, posB=ep,
                        arrowstyle='-|>', mutation_scale=8, 
                        color=color, linestyle='-', linewidth=1.5, alpha=0.8, zorder=2,
                        shrinkA=0, shrinkB=0
                    )
                    ax.add_patch(arrow)
                
                # ax.text(label_x, label_y, str(i + 1), color='black', fontsize=5, fontweight='bold',
                #         ha='center', va='center', zorder=5,
                #         bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.8))

        # ax.set_title(title)
        ax.axis('equal')
        # ax.grid(True, alpha=0.3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight',dpi=300)
            print(f"DEBUG: Intermediate route visualization saved to {save_path}")
        plt.show()
        plt.close(fig)
        
        ### another figure 
        fig, ax = plt.subplots(figsize=(2.36,2.36),dpi=300)

        for i, step in enumerate(route_data):
            p1 = step['coords_start']
            p2 = step['coords_end']
            
            label_x = p1[0] + (p2[0] - p1[0]) / 3.0
            label_y = p1[1] + (p2[1] - p1[1]) / 3.0

            if step['type'] == 'travel':
                arrow = patches.FancyArrowPatch(
                    posA=p1, posB=p2,
                    arrowstyle='-', mutation_scale=8, 
                    color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1000,
                    shrinkA=0, shrinkB=0
                )
                ax.add_patch(arrow)
                # ax.text(label_x, label_y, str(i + 1), color='red', fontsize=5, fontweight='bold',
                #         ha='center', va='center', zorder=5,
                #         bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.8))
            else:
                materials = step['materials']
                num_mats = len(materials)
                edge_len = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                offset_base = edge_len * 0.12
                
                for m_idx, mat in enumerate(materials):
                    centering = (num_mats - 1) / 2.0
                    shift = (m_idx - centering) * offset_base
                    sp, ep = get_offset_points(p1, p2, shift)
                    color = self.original_lattice._color_map.get(mat, 'blue')
                    
                    arrow = patches.FancyArrowPatch(
                        posA=sp, posB=ep,
                        arrowstyle='-', mutation_scale=8, 
                        color=color, linestyle='-', linewidth=1.5, alpha=0.8, zorder=2,
                        shrinkA=0, shrinkB=0
                    )
                    ax.add_patch(arrow)
                
                # ax.text(label_x, label_y, str(i + 1), color='black', fontsize=5, fontweight='bold',
                #         ha='center', va='center', zorder=5,
                #         bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.8))

        # ax.set_title(title)
        ax.axis('equal')
        # ax.grid(True, alpha=0.3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path[:-4]+'_NoArrow.svg', bbox_inches='tight',dpi=300)
            print(f"DEBUG: Intermediate route visualization saved to {save_path}")
        plt.show()
        plt.close(fig)
      
    def _optimize_open_path(self, closed_route, user_start_pos, debug=False):
        """
        Optimizes the closed loop for an open path.
        """
        if not closed_route:
            return []

        def find_best_cut(route_candidate):
            best_idx = -1
            max_net_gain = float('-inf')
            found_travel = False
            
            p_user = np.array(user_start_pos)
            n = len(route_candidate)
            
            for i, step in enumerate(route_candidate):
                if step['type'] != 'travel':
                    continue
                
                found_travel = True
                next_idx = (i + 1) % n
                next_step = route_candidate[next_idx]
                p_new_start = np.array(next_step['coords_start'])
                
                savings = step['weight']
                entry_cost = np.linalg.norm(p_user - p_new_start)
                net_gain = savings - entry_cost
                
                if net_gain > max_net_gain:
                    max_net_gain = net_gain
                    best_idx = i
            
            if not found_travel:
                for i, step in enumerate(route_candidate):
                    d = np.linalg.norm(p_user - np.array(step['coords_start']))
                    if -d > max_net_gain:
                        max_net_gain = -d
                        best_idx = i - 1
                        
            return best_idx, max_net_gain

        fwd_cut_idx, fwd_gain = find_best_cut(closed_route)
        
        reversed_route = []
        for step in reversed(closed_route):
            new_step = step.copy()
            new_step['u'], new_step['v'] = step['v'], step['u']
            new_step['coords_start'] = step['coords_end']
            new_step['coords_end'] = step['coords_start']
            reversed_route.append(new_step)
            
        rev_cut_idx, rev_gain = find_best_cut(reversed_route)
        
        if rev_gain > fwd_gain:
            if debug: print(f"Direction: REVERSED is better (Gain {rev_gain:.2f} vs {fwd_gain:.2f})")
            final_base_route = reversed_route
            cut_idx = rev_cut_idx
            best_gain = rev_gain
        else:
            if debug: print(f"Direction: FORWARD is better (Gain {fwd_gain:.2f} vs {rev_gain:.2f})")
            final_base_route = closed_route
            cut_idx = fwd_cut_idx
            best_gain = fwd_gain

        n = len(final_base_route)
        part1 = final_base_route[(cut_idx + 1):]
        part2 = final_base_route[:cut_idx]
        rotated_route = part1 + part2
        
        if rotated_route:
            first_step = rotated_route[0]
            p_first = first_step['coords_start']
            entry_dist = np.linalg.norm(np.array(user_start_pos) - np.array(p_first))
            
            entry_jump = {
                'u': -1, 
                'v': first_step['u'],
                'coords_start': user_start_pos,
                'coords_end': p_first,
                'type': 'travel',
                'materials': [],
                'weight': entry_dist
            }
            
            final_route = [entry_jump] + rotated_route
        else:
            final_route = []

        removed_edge_weight = final_base_route[cut_idx]['weight']
        if debug: 
            print(f"Optimal Cut at Step {cut_idx}")
            print(f"  - Removed Return Jump: {removed_edge_weight:.4f}")
            print(f"  - Added Entry Jump:    {entry_jump['weight']:.4f}")
            print(f"  - Net Distance Saved:  {best_gain:.4f}")

        return final_route
    
    def get_solution_stats(self):
        """Returns the dictionary of statistics calculated during solve()."""
        return getattr(self, 'stats', None)
    
    def _connect_components_mst(self):
        """
        Strict Frederickson RPP component-connection step.
        Builds the component graph Gc (one vertex per connected component,
        edge weights = nearest-node Euclidean distance), takes its MST, and
        adds each MST connector as a SINGLE travel edge to working_graph.
        Parity is intentionally left for the subsequent matching step.
        """
        current_components = list(nx.connected_components(self.working_graph))
        if len(current_components) <= 1:
            return

        comp_graph = nx.Graph()
        
        for i, comp_a in enumerate(current_components):
            for j, comp_b in enumerate(current_components):
                if i >= j: continue
                
                nodes_a = list(comp_a)
                nodes_b = list(comp_b)
                
                coords_a = [[self.working_graph.nodes[n]['x'], self.working_graph.nodes[n]['y']] for n in nodes_a]
                coords_b = [[self.working_graph.nodes[n]['x'], self.working_graph.nodes[n]['y']] for n in nodes_b]
                
                d_matrix = cdist(coords_a, coords_b)
                min_idx = np.unravel_index(np.argmin(d_matrix), d_matrix.shape)
                
                u = nodes_a[min_idx[0]]
                v = nodes_b[min_idx[1]]
                dist = d_matrix[min_idx]
                
                comp_graph.add_edge(i, j, weight=dist, connector=(u, v))

        mst = nx.minimum_spanning_tree(comp_graph)

        # Strict Frederickson RPP: add SINGLE MST edges (no doubling).
        # The subsequent odd-degree recomputation + perfect matching will
        # restore Eulerianity over the modified parity set.
        for c1, c2, data in mst.edges(data=True):
            u, v = data['connector']
            dist = data['weight']
            self.working_graph.add_edge(u, v, type='travel', materials=[], weight=dist)

    def _apply_biased_matching(self, odd_nodes, penalty_weight=0.0):
        """
        Standard minimum weight perfect matching.
        """
        matching_graph = nx.Graph()
        coords = np.array([[self.working_graph.nodes[n]['x'], self.working_graph.nodes[n]['y']] for n in odd_nodes])
        dists = cdist(coords, coords, metric='euclidean')
        
        edges_to_add = []
        for i, u in enumerate(odd_nodes):
            for j, v in enumerate(odd_nodes):
                if i >= j: continue
                weight = -(dists[i][j] + penalty_weight)
                edges_to_add.append((u, v, weight))
        
        matching_graph.add_weighted_edges_from(edges_to_add)
        matches = nx.max_weight_matching(matching_graph, maxcardinality=True)
        
        for u, v in matches:
            u_data = self.working_graph.nodes[u]
            v_data = self.working_graph.nodes[v]
            
            dist = np.linalg.norm(
                np.array([u_data['x'], u_data['y']]) - np.array([v_data['x'], v_data['y']])
            )
            
            self.working_graph.add_edge(u, v, 
                                      type='travel', 
                                      materials=[], 
                                      weight=dist,
                                      is_matching=True)

    def _merge_components_by_swapping(self, debug=False, startPos = [0.0,0.0]):
        """
        Merges components using DP-based optimal sequential connections.
        """
        if startPos is None:
            startPos = np.array([0.0, 0.0])
        else:
            startPos = np.array(startPos)
            
        import math
        
        components = list(nx.connected_components(self.working_graph))
        if len(components) <= 1:
            if debug: print("Already connected or single component.")
            return
        
        node_to_comp = {}
        for idx, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = idx
        
        comp_matching_edges = {i: [] for i in range(len(components))}
        
        for u, v, key, data in self.working_graph.edges(keys=True, data=True):
            if data.get('is_matching'):
                c_id = node_to_comp[u]
                if c_id == node_to_comp[v]:
                    p_u = np.array([self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']])
                    p_v = np.array([self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']])
                    
                    comp_matching_edges[c_id].append({
                        'u': u, 'v': v, 'key': key, 'weight': data['weight'],
                        'p_u': p_u, 'p_v': p_v
                    })
        
        swappable_comp_indices = [i for i in range(len(components)) if comp_matching_edges[i]]
        
        if len(swappable_comp_indices) <= 1:
            if debug: print(f"Only {len(swappable_comp_indices)} component(s) with swappable edges. Cannot merge via swapping.")
            return
        
        def get_component_centroid(comp_nodes):
            coords = np.array([[self.working_graph.nodes[n]['x'], self.working_graph.nodes[n]['y']] 
                              for n in comp_nodes])
            return np.mean(coords, axis=0)
        
        comp_data = []
        for idx in swappable_comp_indices:
            comp_data.append({
                'index': idx,
                'centroid': get_component_centroid(components[idx]),
                'nodes': components[idx],
                'matching_edges': comp_matching_edges[idx]
            })
        
        self._compute_directionality()
        if self.directionalty is not None:
            sweep_angle = self.directionalty + (math.pi / 2.0)
        else:
            sweep_angle = math.pi / 2.0
        
        sweep_vec = np.array([math.cos(sweep_angle), math.sin(sweep_angle)])
        
        comp_data.sort(key=lambda item: np.dot(item['centroid'], sweep_vec))
        
        
        
        for rank, item in enumerate(comp_data):
            item['rank'] = rank
        
        if debug: print(f"Merging {len(comp_data)} components. Sweep angle: {math.degrees(sweep_angle):.1f}°")
        
        if debug:
            self._debug_plot_sorted_components(comp_data, sweep_vec, components, node_to_comp)
        
        k = len(comp_data)
        
        def get_endpoint_pos(comp_rank, edge_idx, endpoint):
            edge = comp_data[comp_rank]['matching_edges'][edge_idx]
            return edge['p_u'] if endpoint == 0 else edge['p_v']
        
        def get_endpoint_node(comp_rank, edge_idx, endpoint):
            edge = comp_data[comp_rank]['matching_edges'][edge_idx]
            return edge['u'] if endpoint == 0 else edge['v']
        
        INF = float('inf')
        dp = [{} for _ in range(k)]
        
        for e_idx, edge in enumerate(comp_data[0]['matching_edges']):
            dp[0][(e_idx, 0)] = (0.0, {'closure_endpoint': 1})
            dp[0][(e_idx, 1)] = (0.0, {'closure_endpoint': 0})
        
        for i in range(1, k):
            curr_edges = comp_data[i]['matching_edges']
            
            for curr_e_idx, curr_edge in enumerate(curr_edges):
                for curr_forward in [0, 1]:
                    curr_backward = 1 - curr_forward
                    curr_backward_pos = get_endpoint_pos(i, curr_e_idx, curr_backward)
                    
                    best_cost = INF
                    best_backtrack = None
                    
                    for (prev_e_idx, prev_forward), (prev_cost, prev_bt) in dp[i - 1].items():
                        prev_forward_pos = get_endpoint_pos(i - 1, prev_e_idx, prev_forward)
                        conn_dist = np.linalg.norm(prev_forward_pos - curr_backward_pos)
                        total_cost = prev_cost + conn_dist
                        
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_backtrack = {
                                'prev_state': (prev_e_idx, prev_forward),
                                'prev_node': get_endpoint_node(i - 1, prev_e_idx, prev_forward),
                                'curr_node': get_endpoint_node(i, curr_e_idx, curr_backward),
                                'conn_dist': conn_dist
                            }
                    
                    if best_cost < INF:
                        dp[i][(curr_e_idx, curr_forward)] = (best_cost, best_backtrack)
        
        best_score = INF
        best_final_state = None
        best_closure_info = None
        
        for (last_e_idx, last_forward), (last_cost, last_bt) in dp[k - 1].items():
            last_closure_pos = get_endpoint_pos(k - 1, last_e_idx, last_forward)
            last_closure_node = get_endpoint_node(k - 1, last_e_idx, last_forward)
            
            current_state = (last_e_idx, last_forward)
            current_bt = last_bt
            
            for j in range(k - 1, 0, -1):
                if current_bt is None or 'prev_state' not in current_bt:
                    break
                current_state = current_bt['prev_state']
                if j > 1:
                    current_bt = dp[j - 1][current_state][1]
            
            first_e_idx, first_forward = current_state
            first_closure_endpoint = 1 - first_forward
            first_closure_pos = get_endpoint_pos(0, first_e_idx, first_closure_endpoint)
            first_closure_node = get_endpoint_node(0, first_e_idx, first_closure_endpoint)
            
            closure_dist = np.linalg.norm(last_closure_pos - first_closure_pos)
            
            minStartDist = np.min([np.linalg.norm(last_closure_pos - startPos),
                                  np.linalg.norm(first_closure_pos - startPos)])
            
            # can tweak 
            score = last_cost + 0.5 * minStartDist - 0.01 * closure_dist 
            if debug:
                print(f"score = {score}, best_score = {best_score}, minStartDist={minStartDist},closure_dist={closure_dist}")
            
            if score < best_score:
                best_score = score
                best_final_state = (last_e_idx, last_forward)
                best_closure_info = {
                    'first_node': first_closure_node,
                    'first_pos': first_closure_pos,
                    'last_node': last_closure_node,
                    'last_pos': last_closure_pos,
                    'dist': closure_dist
                }
        
        if best_final_state is None:
            if debug: print("ERROR: No valid path found.")
            return
        
        connections = []
        broken_edges = {}
        
        current_state = best_final_state
        current_rank = k - 1
        
        while current_rank >= 0:
            e_idx, forward = current_state
            edge = comp_data[current_rank]['matching_edges'][e_idx]
            
            broken_edges[current_rank] = {
                'edge': edge,
                'forward_endpoint': forward,
                'backward_endpoint': 1 - forward
            }
            
            if current_rank > 0:
                cost, bt = dp[current_rank][current_state]
                if bt and 'prev_state' in bt:
                    connections.append({
                        'from_rank': current_rank - 1,
                        'to_rank': current_rank,
                        'from_node': bt['prev_node'],
                        'to_node': bt['curr_node'],
                        'dist': bt['conn_dist']
                    })
                    current_state = bt['prev_state']
            
            current_rank -= 1
        
        connections.reverse()
        
        if debug:
            self._debug_plot_planned_connections(
                comp_data, components, node_to_comp,
                broken_edges, connections, best_closure_info
            )
        
        if debug: 
            print(f"\nExecuting swaps:")
            print(f"  Components to connect: {k}")
            print(f"  Sequential connections: {len(connections)}")
        
        for rank, info in broken_edges.items():
            edge = info['edge']
            try:
                self.working_graph.remove_edge(edge['u'], edge['v'], key=edge['key'])
                if debug: print(f"    Removed edge ({edge['u']}, {edge['v']}) from comp rank {rank}")
            except nx.NetworkXError as e:
                if debug: print(f"    WARNING: Could not remove ({edge['u']}, {edge['v']}): {e}")
        
        sequential_cost = 0.0
        for conn in connections:
            self.working_graph.add_edge(
                conn['from_node'], conn['to_node'],
                type='travel',
                materials=[],
                weight=conn['dist'],
                is_matching=True
            )
            sequential_cost += conn['dist']
            if debug: print(f"    Added connection ({conn['from_node']}, {conn['to_node']}), dist={conn['dist']:.4f}")
        
        self.working_graph.add_edge(
            best_closure_info['first_node'], best_closure_info['last_node'],
            type='travel',
            materials=[],
            weight=best_closure_info['dist'],
            is_matching=True
        )
        if debug: 
            print(f"    Added CLOSURE ({best_closure_info['first_node']}, {best_closure_info['last_node']}), dist={best_closure_info['dist']:.4f}")
            
            print(f"\nSwapping complete:")
            print(f"  Sequential cost: {sequential_cost:.4f}")
            print(f"  Closure cost: {best_closure_info['dist']:.4f}")
            print(f"  Total: {sequential_cost + best_closure_info['dist']:.4f}")
        
        remaining = nx.number_connected_components(self.working_graph)
        if debug: print(f"  Components remaining: {remaining}")
        
        if debug and remaining == 1:
            self.visualize_after_swapping(
                save_path=foldername + "/figs/debug_swap_final.svg",
                title="Final Result After Swapping"
            )
    
    def _debug_plot_planned_connections(self, comp_data, components, node_to_comp,
                                           broken_edges, connections, closure_info):
        """
        Debug visualization: Shows planned edge removals and additions.
        """
        from matplotlib.lines import Line2D
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(5,4),dpi=300)
        
        num_comps = len(comp_data)
        cmap = plt.get_cmap('tab20', max(num_comps, 1))
        
        for u, v, data in self.working_graph.edges(data=True):
            p1 = [self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']]
            p2 = [self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=1.5, alpha=0.8)
        
        for node, data in self.working_graph.nodes(data=True):
            ax.scatter(data['x'], data['y'], c='gray', s=10, zorder=3, alpha=0.8)
        
        for rank, info in broken_edges.items():
            edge = info['edge']
            p_u, p_v = edge['p_u'], edge['p_v']
            
            ax.plot([p_u[0], p_v[0]], [p_u[1], p_v[1]], color='red', linewidth=1.5, alpha=0.8, zorder=5)
            
            mid = (p_u + p_v) / 2
            ax.scatter(mid[0], mid[1], marker='x', c='red', s=10, linewidths=1.5, zorder=6)
            
            forward_ep = info['forward_endpoint']
            backward_ep = info['backward_endpoint']
            
            forward_pos = p_u if forward_ep == 0 else p_v
            forward_node = edge['u'] if forward_ep == 0 else edge['v']
            ax.scatter(forward_pos[0], forward_pos[1], c='green', s=10, marker='o',
                      edgecolors='black', linewidths=1.5, zorder=7)
            ax.text(forward_pos[0], forward_pos[1] + 0.3, f'→{forward_node}', fontsize=5,
                   ha='center', va='bottom', color='green', fontweight='bold')
            
            backward_pos = p_u if backward_ep == 0 else p_v
            backward_node = edge['u'] if backward_ep == 0 else edge['v']
            ax.scatter(backward_pos[0], backward_pos[1], c='blue', s=10, marker='o',
                      edgecolors='black', linewidths=1.5, zorder=7)
            ax.text(backward_pos[0], backward_pos[1] - 0.3, f'←{backward_node}', fontsize=5,
                   ha='center', va='top', color='blue', fontweight='bold')
            
            ax.text(mid[0], mid[1] + 0.8, f'Rank {rank}', fontsize=5, ha='center',
                   va='bottom', color='red', fontweight='bold')
        
        for i, conn in enumerate(connections):
            from_node = conn['from_node']
            to_node = conn['to_node']
            dist = conn['dist']
            
            p1 = np.array([self.working_graph.nodes[from_node]['x'],
                          self.working_graph.nodes[from_node]['y']])
            p2 = np.array([self.working_graph.nodes[to_node]['x'],
                          self.working_graph.nodes[to_node]['y']])
            
            arrow = patches.FancyArrowPatch(
                posA=p1, posB=p2,
                arrowstyle='-|>', mutation_scale=8,
                color='green', linewidth=1.5, alpha=0.8, zorder=8
            )
            ax.add_patch(arrow)
            
            mid = (p1 + p2) / 2
            ax.text(mid[0], mid[1], f'{i+1}\n({dist:.2f})', fontsize=3, ha='center',
                   va='center', color='darkgreen', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        p1 = closure_info['last_pos']
        p2 = closure_info['first_pos']
        
        arrow = patches.FancyArrowPatch(
            posA=p1, posB=p2,
            arrowstyle='-|>', mutation_scale=8,
            color='orange', linewidth=1.5, linestyle='--', alpha=0.8, zorder=8
        )
        ax.add_patch(arrow)
        
        mid = (p1 + p2) / 2
        ax.text(mid[0], mid[1], f'CLOSURE\n({closure_info["dist"]:.2f})', fontsize=3,
               ha='center', va='center', color='darkorange', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        for item in comp_data:
            centroid = item['centroid']
            rank = item['rank']
            color = cmap(rank % 20)
            
            ax.scatter(centroid[0], centroid[1], c=[color], s=40, marker='s',
                      edgecolors='black', linewidths=0.5, zorder=10, alpha=0.8)
            ax.text(centroid[0], centroid[1], str(rank), fontsize=5, fontweight='bold',
                   ha='center', va='center', color='white', zorder=11)
        
        total_seq = sum(c['dist'] for c in connections)
        stats_text = [
            f"Planned Connections:",
            f"  Sequential: {len(connections)}",
            f"  Sequential dist: {total_seq:.4f}",
            f"  Closure dist: {closure_info['dist']:.4f}",
            f"  Total: {total_seq + closure_info['dist']:.4f}",
        ]
        
        textstr = '\n'.join(stats_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=5,
               verticalalignment='top', bbox=props, family='monospace')
        
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=1.5, marker='x', markersize=5,
                   label='Broken edges'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=5, markeredgecolor='black', label='Forward endpoint (→next)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=5, markeredgecolor='black', label='Backward endpoint (←prev)'),
            Line2D([0], [0], color='green', linewidth=1.5, marker='>', markersize=5,
                   label='Sequential connections'),
            Line2D([0], [0], color='orange', linewidth=1.5, linestyle='--', marker='>',
                   markersize=5, label='Closure (to be cut)'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=5)
        ax.set_title('Planned Connections (DP Optimized)', fontsize=5, fontweight='bold')
        ax.axis('equal')
        # ax.grid(True, alpha=0.3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        save_path = foldername + "/figs/debug_planned_connections_v2.svg"
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"DEBUG: Saved planned connections plot to {save_path}")
        
        plt.show()
        plt.close(fig)
    
    def _compute_directionality(self):
        """Compute average directionality of all components."""
        components = list(nx.connected_components(self.working_graph))
        if len(components) <= 1:
            return

        import math

        def get_component_directionality(comp_nodes):
            subgraph = self.working_graph.subgraph(comp_nodes)
            
            total_weighted_angle = 0.0
            total_weight = 0.0
            
            for u, v, data in subgraph.edges(data=True):
                p1 = np.array([self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']])
                p2 = np.array([self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']])
                
                delta = p2 - p1
                length = data.get('weight', np.linalg.norm(delta))
                
                if length < 1e-9: continue
                    
                angle = math.atan2(delta[1], delta[0]) % math.pi
                total_weighted_angle += angle * length
                total_weight += length
            
            if total_weight == 0:
                return 0.0
            
            return total_weighted_angle / total_weight

        global_dir_sum = 0.0
        
        for comp in components:
            global_dir_sum += get_component_directionality(comp)

        self.directionalty = global_dir_sum / len(components)
    
    def _debug_plot_sorted_components(self, comp_data, sweep_vec, components, node_to_comp):
        """
        Debug visualization: Shows sorted components with centroids and rankings.
        """
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=(5,4),dpi=300)
        
        num_comps = len(comp_data)
        cmap = plt.get_cmap('tab20', max(num_comps, 1))
        
        for u, v, data in self.working_graph.edges(data=True):
            u_comp = node_to_comp.get(u, -1)
            
            p1 = [self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']]
            p2 = [self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']]
            
            rank = -1
            for item in comp_data:
                if item['index'] == u_comp:
                    rank = item['rank']
                    break
            
            if rank >= 0:
                color = cmap(rank % 20)
            else:
                color = 'lightgray'
            
            edge_type = data.get('type', 'print')
            if edge_type == 'print':
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5, alpha=0.8)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5, 
                       linestyle='--', alpha=0.8)
        
        for node, data in self.working_graph.nodes(data=True):
            comp_idx = node_to_comp.get(node, -1)
            rank = -1
            for item in comp_data:
                if item['index'] == comp_idx:
                    rank = item['rank']
                    break
            
            if rank >= 0:
                color = cmap(rank % 20)
            else:
                color = 'lightgray'
            
            ax.scatter(data['x'], data['y'], c=[color], s=10, zorder=3)
        
        for item in comp_data:
            centroid = item['centroid']
            rank = item['rank']
            color = cmap(rank % 20)
            
            ax.scatter(centroid[0], centroid[1], c=[color], s=40, marker='s',
                      edgecolors='black', linewidths=0.1, zorder=10)
            ax.text(centroid[0], centroid[1], str(rank), fontsize=5, fontweight='bold',
                   ha='center', va='center', color='white', zorder=11)
            
            ax.text(centroid[0], centroid[1] - 1.5, f"(idx:{item['index']})", fontsize=5,
                   ha='center', va='top', color='black')
        
        all_centroids = np.array([item['centroid'] for item in comp_data])
        center = np.mean(all_centroids, axis=0)
        
        extent = np.max(np.ptp(all_centroids, axis=0)) * 0.3
        arrow_start = center - sweep_vec * extent
        arrow_end = center + sweep_vec * extent
        
        ax.annotate('', xy=arrow_end, xytext=arrow_start,
                   arrowprops=dict(arrowstyle='-|>', color='red', lw=1))
        ax.text(arrow_end[0], arrow_end[1], 'Sweep\nDirection', fontsize=5, color='red',
               ha='center', va='bottom', fontweight='bold')
        
        for i in range(len(comp_data) - 1):
            c1 = comp_data[i]['centroid']
            c2 = comp_data[i + 1]['centroid']
            
            direction = c2 - c1
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            
            ax.annotate('', xy=c2 - direction * 2, xytext=c1 + direction * 2,
                       arrowprops=dict(arrowstyle='-|>', color='green', lw=1, alpha=0.8))
        
        if len(comp_data) >= 2:
            c_first = comp_data[0]['centroid']
            c_last = comp_data[-1]['centroid']
            
            ax.annotate('', xy=c_first, xytext=c_last,
                       arrowprops=dict(arrowstyle='-|>', color='orange', lw=1, 
                                      linestyle='--', alpha=0.8))
            mid_closure = (c_first + c_last) / 2
            ax.text(mid_closure[0], mid_closure[1], 'CLOSURE\n(to be cut)', fontsize=5,
                   ha='center', va='center', color='orange', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        legend_elements = [
            Line2D([0], [0], color='green', lw=1, marker='>', markersize=3, 
                   label='Sequential Connection Order'),
            Line2D([0], [0], color='orange', lw=1, linestyle='--', marker='>', 
                   markersize=3, label='Loop Closure'),
            Line2D([0], [0], color='red', lw=1, marker='>', markersize=3, 
                   label='Sweep Direction'),
        ]
        
        for rank in range(min(num_comps, 10)):
            legend_elements.append(
                mpatches.Patch(facecolor=cmap(rank % 20), edgecolor='black',
                              label=f'Rank {rank}')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=5)
        ax.set_title(f'Sorted Components by Sweep Direction\n({num_comps} components)', 
                    fontsize=5, fontweight='bold')
        ax.axis('equal')
        # ax.grid(True, alpha=0.3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        save_path = foldername + "/figs/debug_sorted_components.svg"
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"DEBUG: Saved sorted components plot to {save_path}")
        
        plt.show()
        plt.close(fig)
   
    def _process_route(self, raw_circuit):
        """Convert raw Eulerian circuit to data-rich route."""
        processed_route = []
        for u, v in raw_circuit:
            edge_data = self.working_graph.get_edge_data(u, v)
            
            key_to_use = None
            for key, attributes in edge_data.items():
                if not attributes.get('_visited', False):
                    key_to_use = key
                    break
            
            if key_to_use is None: continue
                
            attr = edge_data[key_to_use]
            self.working_graph[u][v][key_to_use]['_visited'] = True
            
            step = {
                'u': u, 'v': v,
                'coords_start': [self.working_graph.nodes[u]['x'], self.working_graph.nodes[u]['y']],
                'coords_end': [self.working_graph.nodes[v]['x'], self.working_graph.nodes[v]['y']],
                'type': attr.get('type', 'print'),
                'materials': attr.get('materials', []),
                'weight': attr['weight']
            }
            processed_route.append(step)
        return processed_route

    def visualize_route(self, save_path=None, arrow_size=8, show_numbers=True,linewidth=1.5, fontsize=8, fontsize_axis = 8):
        """
        Visualizes the final toolpath.
        """
        if not self.route:
            print("No route to visualize. Run solve() first.")
            return

        fig, ax = plt.subplots(figsize=(5,4),dpi=300)
        
        # for node, data in self.original_lattice.graph.nodes(data=True):
        #     ax.scatter(data['x'], data['y'], c='black', s=3, zorder=3)

        import matplotlib.patches as patches
        
        def get_offset_points(p1, p2, offset_amount):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length == 0: return p1, p2
            
            perp_x = -dy / length
            perp_y = dx / length
            
            return (p1[0] + perp_x * offset_amount, p1[1] + perp_y * offset_amount), \
                   (p2[0] + perp_x * offset_amount, p2[1] + perp_y * offset_amount)

        sequence_counter = 0

        for i, step in enumerate(self.route):
            if i == 0 and step['type'] == 'travel' and step['weight'] < 1e-6:
                continue
            
            sequence_counter += 1

            p1 = step['coords_start']
            p2 = step['coords_end']
            
            label_x = p1[0] + (p2[0] - p1[0]) / 3.5
            label_y = p1[1] + (p2[1] - p1[1]) / 3.5

            if step['type'] == 'travel':
                arrow = patches.FancyArrowPatch(
                    posA=p1, posB=p2,
                    arrowstyle='-|>', mutation_scale=arrow_size, 
                    color='red', linestyle='--', linewidth=linewidth, alpha=0.5, zorder=1000,
                    shrinkA=0, shrinkB=0
                )
                ax.add_patch(arrow)
                
                if show_numbers:
                    ax.text(label_x, label_y, str(sequence_counter), color='red',
                            fontsize=fontsize, fontweight='bold', ha='center', va='center', zorder=5,
                            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.8))
            else:
                materials = step['materials']
                num_mats = len(materials)
                
                edge_len = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                offset_base = edge_len * 0.12 
                
                for m_idx, mat in enumerate(materials):
                    centering = (num_mats - 1) / 2.0
                    shift = (m_idx - centering) * offset_base
                    
                    sp, ep = get_offset_points(p1, p2, shift)
                    color = self.original_lattice._color_map.get(mat, 'blue')
                    # lw = 1.5 #if num_mats == 1 else 1.5
                    
                    arrow = patches.FancyArrowPatch(
                        posA=sp, posB=ep,
                        arrowstyle='-|>', mutation_scale=arrow_size, 
                        color=color, linestyle='-', linewidth=linewidth, alpha=0.8, zorder=2,
                        shrinkA=0, shrinkB=0
                    )
                    ax.add_patch(arrow)

                if show_numbers:
                    ax.text(label_x, label_y, str(sequence_counter), color='black',
                            fontsize=fontsize, fontweight='bold', ha='center', va='center', zorder=5,
                            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.8))

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', linestyle='--', lw=linewidth, label='Jump/Travel')]
        for m in self.original_lattice.get_all_materials():
            c = self.original_lattice._color_map.get(m, 'blue')
            legend_elements.append(Line2D([0], [0], color=c, lw=1.5, label=m))
            
        ax.legend(handles=legend_elements, loc='best',fontsize=fontsize_axis)
        # ax.set_title("Optimized Toolpath")
        plt.xticks(fontsize=fontsize_axis)
        plt.yticks(fontsize=fontsize_axis)
        ax.axis('equal')
        # ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight',dpi=300)
        plt.show()

# Visualization for planned path and generate g-code
class ToolpathVisualizer:
    def __init__(self, toolpath_source, material_offsets, lift_distance=1.0, layer_thickness=0.2):
        """
        Args:
            toolpath_source (str or list): Path to 'toolpath.json' or the list returned by solve().
            material_offsets (dict): Key=Material ID, Value=[dx, dy] offset vector. 
                                     e.g. {'Mat1': [0.2, 0.0], 'Mat2': [-0.2, 0.0]}
            lift_distance (float): Height to lift z-axis during travel moves.
            layer_thickness (float): Thickness of the print layer (for future volumetric rendering).
        """
        # 1. Load Data
        if isinstance(toolpath_source, str):
            if os.path.exists(toolpath_source):
                with open(toolpath_source, 'r') as f:
                    self.toolpath = json.load(f)
            else:
                raise FileNotFoundError(f"Toolpath file not found: {toolpath_source}")
        else:
            self.toolpath = toolpath_source
            
        # 2. Store Config
        self.offsets = material_offsets
        self.lift_dist = lift_distance
        self.layer_h = layer_thickness
        
        # Default colors for materials (expand as needed)
        self.color_map = {
            'travel': 'red',
            'support': 'gray'
        }
        # Auto-generate colors for materials in offsets if not preset
        import matplotlib.cm as cm
        mats = list(self.offsets.keys())
        cmap = plt.get_cmap('tab10')
        for i, m in enumerate(mats):
            if m not in self.color_map:
                # Convert mpl color to hex or just store tuple
                self.color_map[m] = cmap(i % 10)

    def visualize_layer_3d(self, foldername, fixed_arrow_size=1.0, show_labels=True, debug=False):
        """
        Plots the toolpath in 3D for a single layer (Z=0 base) with SEQUENCE NUMBERS.
        
        Numbering Logic:
        - Travel Moves count as 3 distinct edges: Lift -> Move -> Land.
        - Print Moves count as 1 edge (labeled on the centerline).
        
        Args:
            foldername (str): Directory to save the visualization.
            fixed_arrow_size (float): The physical size of the arrow head (in plot units/mm).
            show_labels (bool): Whether to plot the step numbers.
        """
        from mpl_toolkits.mplot3d.proj3d import proj_transform
        from matplotlib.patches import FancyArrowPatch
        
        # Custom 3D Arrow class with configurable arrow style
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, arrowstyle='-|>', mutation_scale=10, **kwargs):
                super().__init__((0, 0), (0, 0), arrowstyle=arrowstyle, 
                               mutation_scale=mutation_scale, **kwargs)
                self._verts3d = xs, ys, zs
    
            def do_3d_projection(self, renderer=None):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                return np.min(zs)
        
        # Helper function to draw 3D arrow
        def draw_arrow_3d(ax, start, end, color='black', linestyle='-', 
                          arrowstyle='-|>', mutation_scale=10, alpha=0.8, linewidth=1):
            """
            Draws a 3D arrow from start to end point.
            
            Args:
                ax: 3D axes object
                start: (x, y, z) tuple for start point
                end: (x, y, z) tuple for end point
                color: Arrow color
                linestyle: Line style ('-', '--', etc.)
                arrowstyle: Arrow head style ('-|>', '->', 'fancy', etc.)
                mutation_scale: Size of arrow head
                alpha: Transparency
                linewidth: Width of arrow shaft
            """
            arrow = Arrow3D([start[0], end[0]], 
                           [start[1], end[1]], 
                           [start[2], end[2]],
                           arrowstyle=arrowstyle,
                           mutation_scale=mutation_scale,
                           color=color,
                           linestyle=linestyle,
                           linewidth=linewidth,
                           alpha=alpha)
            ax.add_artist(arrow)
            return arrow
        
        save_dir = os.path.join(foldername, "paths")
        os.makedirs(save_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(5,4), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
    
        # Helper to add label
        def add_label(x, y, z, num, color='blue'):
            if show_labels:
                ax.text(x, y, z, str(num), color=color, fontsize=4, fontweight='bold', 
                        bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="none", alpha=0.8),zorder=100)
    
        if debug:
            print(f"--- Generating 3D Toolpath Visualization in {save_dir} ---")
    
        global_step_counter = 0
    
        for step in self.toolpath:
            p_start_base = np.array(step['coords_start'])
            p_end_base = np.array(step['coords_end'])
            
            # --- CASE 1: TRAVEL MOVE (Red Lines) ---
            if step['type'] == 'travel':
                # 1. Vertical Lift Arrow (Start_Z=0 -> Start_Z=Lift)
                global_step_counter += 1
                
                draw_arrow_3d(ax, 
                             (p_start_base[0], p_start_base[1], 0),
                             (p_start_base[0], p_start_base[1], self.lift_dist),
                             color='red', linestyle='--', arrowstyle='-|>',
                             mutation_scale=fixed_arrow_size * 10, alpha=0.5)
                
                add_label(p_start_base[0], p_start_base[1], self.lift_dist / 2, 
                         global_step_counter, color='#1f77b4')
                
                # 2. Horizontal Travel Arrow (at Z=Lift)
                global_step_counter += 1
                
                draw_arrow_3d(ax,
                             (p_start_base[0], p_start_base[1], self.lift_dist),
                             (p_end_base[0], p_end_base[1], self.lift_dist),
                             color='red', linestyle='-', arrowstyle='-|>',
                             mutation_scale=fixed_arrow_size * 10, alpha=0.5)
                
                dx = p_end_base[0] - p_start_base[0]
                dy = p_end_base[1] - p_start_base[1]
                add_label(p_start_base[0] + dx / 2, p_start_base[1] + dy / 2, 
                         self.lift_dist, global_step_counter, color='#1f77b4')
                
                # 3. Vertical Land Arrow (End_Z=Lift -> End_Z=0)
                global_step_counter += 1
                
                draw_arrow_3d(ax,
                             (p_end_base[0], p_end_base[1], self.lift_dist),
                             (p_end_base[0], p_end_base[1], 0),
                             color='red', linestyle='--', arrowstyle='-|>',
                             mutation_scale=fixed_arrow_size * 10, alpha=0.5)
                
                add_label(p_end_base[0], p_end_base[1], self.lift_dist / 2, 
                         global_step_counter, color='#1f77b4')
            # --- CASE 2: PRINT MOVE (Black Line + Colored Offsets) ---
            else:
                global_step_counter += 1
                
                dx = p_end_base[0] - p_start_base[0]
                dy = p_end_base[1] - p_start_base[1]
                dist_print = np.sqrt(dx**2 + dy**2)
                
                # 1. Draw the full line first
                ax.plot([p_start_base[0], p_end_base[0]], 
                       [p_start_base[1], p_end_base[1]], 
                       [0, 0], 
                       color='black', linewidth=1, alpha=0.8)
                
                # Label at midpoint
                add_label(p_start_base[0] + dx / 2, p_start_base[1] + dy / 2, 0, 
                         global_step_counter, color='#1f77b4')
                
                # 2. Draw short arrow at the end for direction indicator
                if dist_print > 1e-6:
                    unit_dx = dx / dist_print
                    unit_dy = dy / dist_print
                    
                    # Arrow starts slightly before endpoint
                    arrow_length = min(fixed_arrow_size, dist_print * 0.3)
                    arrow_start_x = p_end_base[0] - unit_dx * arrow_length
                    arrow_start_y = p_end_base[1] - unit_dy * arrow_length
                    
                    draw_arrow_3d(ax,
                                 (arrow_start_x, arrow_start_y, 0),
                                 (p_end_base[0], p_end_base[1], 0),
                                 color='black', linestyle='-', arrowstyle='-|>',
                                 mutation_scale=fixed_arrow_size * 10, alpha=0.8, linewidth=1)

        # Setting up plot limits and labels
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.set_zlabel('Z (mm)', fontsize=8)
        ax.set_title(f'3D Toolpath Visualization (Lift = {self.lift_dist}mm)', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        
        # Enforce equal aspect ratio for XY
        all_coords = [s['coords_start'] for s in self.toolpath] + [s['coords_end'] for s in self.toolpath]
        all_coords = np.array(all_coords)
        if len(all_coords) > 0:
            max_range = np.array([all_coords[:, 0].max() - all_coords[:, 0].min(), 
                                  all_coords[:, 1].max() - all_coords[:, 1].min(), 
                                  self.lift_dist]).max() / 2.0
            mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
            mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(0, max(self.lift_dist * 1.5, 1.0))
    
        # Create Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', lw=1, label='Nozzle Path'),
            Line2D([0], [0], color='red', linestyle='-', lw=1, label='Travel Move'),
            Line2D([0], [0], color='red', linestyle='--', lw=1, label='Lift/Land')
        ]
        
        ax.legend(handles=legend_elements, fontsize=6)
        
        # Save
        filename = os.path.join(save_dir, "layer_3d_vis.svg")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if debug:
            print(f"Saved visualization to: {filename}")
        plt.show()
        plt.close(fig)
        
    def visualize_material_paths(self, foldername, fixed_arrow_size=1.0, show_labels=True, debug=False):
        """
        Visualizes the individual material deposition paths (Material-centric view).
        Adds SEQUENCE NUMBERS to the printing edges.
        
        Args:
            foldername (str): Output folder.
            fixed_arrow_size (float): Physical size of arrow heads.
            show_labels (bool): If True, adds step numbers to print edges.
        """
        from mpl_toolkits.mplot3d.proj3d import proj_transform
        from matplotlib.patches import FancyArrowPatch
        
        # Custom 3D Arrow class with configurable arrow style
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, arrowstyle='-|>', mutation_scale=10, **kwargs):
                super().__init__((0, 0), (0, 0), arrowstyle=arrowstyle, 
                               mutation_scale=mutation_scale, **kwargs)
                self._verts3d = xs, ys, zs
    
            def do_3d_projection(self, renderer=None):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                return np.min(zs)
        
        # Helper function to draw 3D arrow
        def draw_arrow_3d(ax, start, end, color='black', linestyle='-', 
                          arrowstyle='-|>', mutation_scale=15, alpha=0.8, linewidth=1.5):
            """
            Draws a 3D arrow from start to end point.
            """
            start = np.array(start)
            end = np.array(end)
            
            # Extend endpoint so tip reaches exact position
            direction = end - start
            length = np.linalg.norm(direction)
            
            if length > 1e-6:
                unit_dir = direction / length
                extension = mutation_scale * 0.03
                end_extended = end + unit_dir * extension
            else:
                end_extended = end
            
            arrow = Arrow3D([start[0], end_extended[0]], 
                           [start[1], end_extended[1]], 
                           [start[2], end_extended[2]],
                           arrowstyle=arrowstyle,
                           mutation_scale=mutation_scale,
                           color=color,
                           linestyle=linestyle,
                           linewidth=linewidth,
                           alpha=alpha)
            ax.add_artist(arrow)
            return arrow
        
        save_dir = os.path.join(foldername, "paths")
        os.makedirs(save_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(5,4), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        # --- Helpers ---
        def subtract_offset(p, offset_vec):
            return np.array([p[0] - offset_vec[0], p[1] - offset_vec[1]])
    
        def add_label(pos, text, color='black'):
            if show_labels:
                ax.text(pos[0], pos[1], pos[2], text, color=color, fontsize=5, fontweight='bold',
                        ha='center', va='center', zorder=10,
                        bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="none", alpha=0.8))
    
        # Helper to plot a full move (Lift -> Travel -> Land)
        def plot_travel_segment(p_start_xy, p_end_xy, color='red', style='--', label=None):
            mutation = fixed_arrow_size * 10
            
            # 1. Lift
            draw_arrow_3d(ax,
                         (p_start_xy[0], p_start_xy[1], 0),
                         (p_start_xy[0], p_start_xy[1], self.lift_dist),
                         color=color, linestyle='--', arrowstyle='-|>',
                         mutation_scale=mutation, alpha=0.3)
            
            # 2. Move (at Lift Height)
            draw_arrow_3d(ax,
                         (p_start_xy[0], p_start_xy[1], self.lift_dist),
                         (p_end_xy[0], p_end_xy[1], self.lift_dist),
                         color=color, linestyle=style, arrowstyle='-|>',
                         mutation_scale=mutation, alpha=0.3)
            
            # 3. Land
            draw_arrow_3d(ax,
                         (p_end_xy[0], p_end_xy[1], self.lift_dist),
                         (p_end_xy[0], p_end_xy[1], 0),
                         color=color, linestyle='--', arrowstyle='-|>',
                         mutation_scale=mutation, alpha=0.3)
    
        if debug:
            print(f"--- Generating Material Path Visualization in {save_dir} ---")
        
        print_num = 0
        # --- Main Loop ---
        for i, step in enumerate(self.toolpath):
            p_start_raw = np.array(step['coords_start'])
            p_end_raw = np.array(step['coords_end'])
            
            # Use 1-based indexing for labels
            step_num = i + 1 
            
            # CASE 1: PRINT (Material Deposition)
            if step['type'] != 'travel':
                print_num += 1
                materials = step.get('materials', [])
                for mat in materials:
                    # Deduct offset to get material position
                    offset = self.offsets.get(mat, [0.0, 0.0])
                    p1 = subtract_offset(p_start_raw, offset)
                    p2 = subtract_offset(p_end_raw, offset)
                    
                    color = self.color_map.get(mat, 'blue')
                    
                    # Plot Line
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], 
                           color=color, linewidth=1, alpha=0.8)
                    
                    # Plot Direction Arrow
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist > 1e-6:
                        # Draw arrow at the end portion of the line
                        unit_dx = dx / dist
                        unit_dy = dy / dist
                        
                        # Arrow starts slightly before endpoint
                        arrow_length = min(fixed_arrow_size, dist * 0.3)
                        arrow_start_x = p2[0] - unit_dx * arrow_length
                        arrow_start_y = p2[1] - unit_dy * arrow_length
                        
                        draw_arrow_3d(ax,
                                     (arrow_start_x, arrow_start_y, 0),
                                     (p2[0], p2[1], 0),
                                     color=color, linestyle='-', arrowstyle='-|>',
                                     mutation_scale=fixed_arrow_size * 10, alpha=0.8, linewidth=2)
                    
                    # Add Number Label at Midpoint
                    mid_pt = (p1 + p2) / 2.0
                    add_label((mid_pt[0], mid_pt[1], 0), str(print_num), color=color)
    
            # CASE 2: TRAVEL (Connection Logic)
            else:
                # 1. Find Previous Materials
                prev_mats = []
                k = i - 1
                while k >= 0:
                    if self.toolpath[k]['type'] != 'travel':
                        prev_mats = self.toolpath[k].get('materials', [])
                        break
                    k -= 1
                
                # 2. Find Next Materials
                next_mats = []
                k = i + 1
                while k < len(self.toolpath):
                    if self.toolpath[k]['type'] != 'travel':
                        next_mats = self.toolpath[k].get('materials', [])
                        break
                    k += 1
                
                if not prev_mats:
                    prev_mats = ['__default__']
                if not next_mats:
                    next_mats = ['__default__']
    
                # --- Connection Logic ---
                
                # Scenario A: Single -> Single
                if len(prev_mats) == 1 and len(next_mats) == 1:
                    off_p = self.offsets.get(prev_mats[0], [0, 0])
                    off_n = self.offsets.get(next_mats[0], [0, 0])
                    s = subtract_offset(p_start_raw, off_p)
                    e = subtract_offset(p_end_raw, off_n)
                    plot_travel_segment(s, e, color='red', style='--',)
    
                # Scenario B: Multi -> Single
                elif len(prev_mats) > 1 and len(next_mats) == 1:
                    off_n = self.offsets.get(next_mats[0], [0, 0])
                    for m in prev_mats:
                        off_p = self.offsets.get(m, [0, 0])
                        s = subtract_offset(p_start_raw, off_p)
                        e = subtract_offset(p_end_raw, off_n)
                        c = self.color_map.get(m, 'red')
                        plot_travel_segment(s, e, color=c, style='-.')
    
                # Scenario C: Single -> Multi
                elif len(prev_mats) == 1 and len(next_mats) > 1:
                    off_p = self.offsets.get(prev_mats[0], [0, 0])
                    for m in next_mats:
                        off_n = self.offsets.get(m, [0, 0])
                        s = subtract_offset(p_start_raw, off_p)
                        e = subtract_offset(p_end_raw, off_n)
                        c = self.color_map.get(m, 'red')
                        plot_travel_segment(s, e, color=c, style='-.')
                
                # Scenario D: Multi -> Multi
                else:
                    plot_travel_segment(p_start_raw, p_end_raw, color='red', style=':')
    
        # Setup Plot
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.set_zlabel('Z (mm)', fontsize=8)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.tick_params(axis='z', labelsize=6)
        ax.set_title(f'Material-Centric Path Visualization (Lift={self.lift_dist})', fontsize=8)
        
        # Scaling
        all_coords = [s['coords_start'] for s in self.toolpath] + [s['coords_end'] for s in self.toolpath]
        all_coords = np.array(all_coords)
        if len(all_coords) > 0:
            max_range = np.array([all_coords[:, 0].max() - all_coords[:, 0].min(), 
                                  all_coords[:, 1].max() - all_coords[:, 1].min(), 
                                  self.lift_dist]).max() / 2.0
            mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
            mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(0, max(self.lift_dist * 1.5, 1.0))
    
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='black', linestyle='--', label='Material Transition')]
        for mat in self.offsets.keys():
            c = self.color_map.get(mat, 'blue')
            legend_elements.append(Line2D([0], [0], color=c, lw=1, label=f'{mat}'))
        
        ax.legend(handles=legend_elements, fontsize=6)
        
        filename = os.path.join(save_dir, "material_paths_vis.svg")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if debug:
            print(f"Saved material visualization to: {filename}")
        plt.show()
        plt.close(fig)

    def visualize_frames_firstlayer(self, foldername, fixed_arrow_size=1.0, show_labels=True, debug=False):
        """
        Generates a sequence of images (frames), one per toolpath step, for animation.
        Saves files as 000.png, 001.png, ... in 'visualization/frames_firstlayer'.
        
        Logic:
        - Calculates global view limits first to keep the camera static.
        - Iteratively adds one edge at a time and saves the snapshot.
        - Uses the same Material-Centric logic (offsets) as visualize_material_paths.
        """
        save_dir = os.path.join(foldername, "frames_firstlayer")
        os.makedirs(save_dir, exist_ok=True)
        
        if debug: print(f"--- Generating Animation Frames in {save_dir} ---")

        # 1. Setup Figure & Static Limits
        fig = plt.figure(figsize=(12, 10),dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate bounds based on ALL coords to fix the camera view
        all_coords = [s['coords_start'] for s in self.toolpath] + [s['coords_end'] for s in self.toolpath]
        all_coords = np.array(all_coords)
        
        if len(all_coords) > 0:
            max_range = np.array([all_coords[:,0].max()-all_coords[:,0].min(), 
                                  all_coords[:,1].max()-all_coords[:,1].min(), 
                                  self.lift_dist]).max() / 2.0
            mid_x = (all_coords[:,0].max()+all_coords[:,0].min()) * 0.5
            mid_y = (all_coords[:,1].max()+all_coords[:,1].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(0, max(self.lift_dist * 1.5, 1.0))
            
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Toolpath Animation')

        # Create Legend (Static)
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='gray', linestyle='--', label='Material Transition')]
        for mat in self.offsets.keys():
            c = self.color_map.get(mat, 'blue')
            legend_elements.append(Line2D([0], [0], color=c, lw=1, label=f'{mat}'))
        ax.legend(handles=legend_elements, loc='upper right')

        # --- Helpers (Reused) ---
        def subtract_offset(p, offset_vec):
            return np.array([p[0] - offset_vec[0], p[1] - offset_vec[1]])

        def get_ratio(vector_len):
            if vector_len < 1e-6: return 0.0
            return min(fixed_arrow_size / vector_len, 1.0)

        def add_label(pos, text, color='black'):
            if show_labels:
                ax.text(pos[0], pos[1], pos[2], text, color=color, fontsize=8, fontweight='bold',
                        ha='center', va='center', zorder=10,
                        bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="none", alpha=0.8))

        def plot_travel_segment(p_start_xy, p_end_xy, color='red', style='--', label=None):
            dist_lift = self.lift_dist
            # Lift
            ax.quiver(p_start_xy[0], p_start_xy[1], 0, 0, 0, dist_lift, 
                      color=color, linestyle=':', arrow_length_ratio=get_ratio(dist_lift), alpha=0.8)
            # Move
            dx = p_end_xy[0] - p_start_xy[0]
            dy = p_end_xy[1] - p_start_xy[1]
            dist_move = np.sqrt(dx**2 + dy**2)
            ax.quiver(p_start_xy[0], p_start_xy[1], dist_lift, dx, dy, 0,
                      color=color, linestyle=style, arrow_length_ratio=get_ratio(dist_move), alpha=0.8)
            # Land
            ax.quiver(p_end_xy[0], p_end_xy[1], dist_lift, 0, 0, -dist_lift, 
                      color=color, linestyle=':', arrow_length_ratio=get_ratio(dist_lift), alpha=0.8)

        # --- Iterative Plotting ---
        total_steps = len(self.toolpath)
        
        for i, step in enumerate(self.toolpath):
            p_start_raw = np.array(step['coords_start'])
            p_end_raw = np.array(step['coords_end'])
            step_num = i + 1

            # ---------------------------------------------------------
            # PLOT LOGIC (Same as visualize_material_paths)
            # ---------------------------------------------------------
            
            # CASE 1: PRINT
            if step['type'] != 'travel':
                materials = step.get('materials', [])
                for mat in materials:
                    offset = self.offsets.get(mat, [0.0, 0.0])
                    p1 = subtract_offset(p_start_raw, offset)
                    p2 = subtract_offset(p_end_raw, offset)
                    color = self.color_map.get(mat, 'blue')
                    
                    # Line
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], color=color, linewidth=1, alpha=0.8)
                    # Arrow
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    ax.quiver(p1[0], p1[1], 0, dx, dy, 0, 
                              color=color, arrow_length_ratio=get_ratio(dist), normalize=False)
                    # Label
                    mid_pt = (p1 + p2) / 2.0
                    add_label((mid_pt[0], mid_pt[1], 0), str(step_num), color=color)

            # CASE 2: TRAVEL
            else:
                # Lookbehind
                prev_mats = []
                k = i - 1
                while k >= 0:
                    if self.toolpath[k]['type'] != 'travel':
                        prev_mats = self.toolpath[k].get('materials', [])
                        break
                    k -= 1
                
                # Lookahead
                next_mats = []
                k = i + 1
                while k < total_steps:
                    if self.toolpath[k]['type'] != 'travel':
                        next_mats = self.toolpath[k].get('materials', [])
                        break
                    k += 1
                
                if not prev_mats: prev_mats = ['__default__']
                if not next_mats: next_mats = ['__default__']

                # Connection Logic
                if len(prev_mats) == 1 and len(next_mats) == 1:
                    off_p = self.offsets.get(prev_mats[0], [0,0])
                    off_n = self.offsets.get(next_mats[0], [0,0])
                    s = subtract_offset(p_start_raw, off_p)
                    e = subtract_offset(p_end_raw, off_n)
                    plot_travel_segment(s, e, color='gray', style='--')

                elif len(prev_mats) > 1 and len(next_mats) == 1:
                    off_n = self.offsets.get(next_mats[0], [0,0])
                    for m in prev_mats:
                        off_p = self.offsets.get(m, [0,0])
                        s = subtract_offset(p_start_raw, off_p)
                        e = subtract_offset(p_end_raw, off_n)
                        c = self.color_map.get(m, 'gray')
                        plot_travel_segment(s, e, color=c, style='-.')

                elif len(prev_mats) == 1 and len(next_mats) > 1:
                    off_p = self.offsets.get(prev_mats[0], [0,0])
                    for m in next_mats:
                        off_n = self.offsets.get(m, [0,0])
                        s = subtract_offset(p_start_raw, off_p)
                        e = subtract_offset(p_end_raw, off_n)
                        c = self.color_map.get(m, 'gray')
                        plot_travel_segment(s, e, color=c, style='-.')
                else:
                    plot_travel_segment(p_start_raw, p_end_raw, color='black', style=':')

            # ---------------------------------------------------------
            # SAVE FRAME
            # ---------------------------------------------------------
            # Update title with progress
            ax.set_title(f'Toolpath Animation (Step {step_num}/{total_steps})')
            
            filename = os.path.join(save_dir, f"{step_num:03d}.png")
            plt.savefig(filename, dpi=300) # Lower DPI slightly for speed if needed
            
            if i % 10 == 0:
                print(f"  > Saved frame {step_num}/{total_steps}", end='\n')

        print(f"\nSaved {total_steps} frames to {save_dir}")
        plt.close(fig)
        
    def export_to_gcode(self, filename, layer_height=None, num_layers=3, printing_speed=20.0, traveling_speed=40.0, dwell_time_travel=0.2, dwell_time_after_travel=0.09, dwell_time_mat_change=0.08):
        """
        Exports the toolpath to a G-code file with Aerotech/A3200 specific syntax.
        
        Features:
        - Pre-calculates Forward and Reverse paths for efficiency.
        - Multi-layer support with alternating directions (Boustrophedon layers).
        - Smart Start: Skips initial travel moves on every layer.
        - Layer Change: Lifts C-axis by layer_height.
        - Valve Control: Smart toggling (OFF between layers).
        """
        if layer_height is None:
            layer_height = self.layer_h

        print(f"--- Exporting G-code to {filename} (Layers: {num_layers}, H: {layer_height}) ---")
        
        # 1. Map Materials to Port Indices
        material_mapping = {mat: i+1 for i, mat in enumerate(self.offsets.keys())}
        
        # 2. Open File and Write Header
        with open(filename, 'w') as f:
            # --- HEADER BLOCK ---
            f.write("var $clientSocket as handle\n")
            f.write('$clientSocket = SocketTcpClientCreate ("127.0.0.1", 65432, 1000)\n')
            f.write("if SocketTcpClientIsConnected ($clientSocket)\n")
            f.write("    // the client is still connected.\n")
            f.write("end\n\n")
            
            f.write("var $speed_travel as real\n")
            f.write(f"$speed_travel = {traveling_speed}\n\n")

            f.write("var $speed_print as real\n")
            f.write(f"$speed_print = {printing_speed}\n\n")
            
            # Dynamic Port Declarations
            f.write("// Material Port Assignments\n")
            for mat, idx in material_mapping.items():
                f.write(f"var $port_{idx} as integer\n")
                f.write(f"$port_{idx} = ... // Set port number for {mat}\n")
            f.write("\n")
            
            # Initialize Ports to 0 (OFF)
            f.write("// Initialize Valves to OFF\n")
            for idx in material_mapping.values():
                f.write(f"$OutBits[$port_{idx}] = 0\n")
            f.write("\n")
            
            # Standard Setup Commands
            f.write("G71\nG76\nG91\nG68\n")
            f.write("Enable([X, Y, C])\n")
            f.write("Home(C)\n\n")
            
            f.write('SocketWriteString($clientSocket, "\\nexecSetPressure([8, 9, 10], [14.4,14.3, 14.5]), execTogglePressure([8, 9, 10]) ")\n\n')
            
            f.write("G90 ; absolute\n")
            f.write(f"G0 X0 Y0 C{layer_height*1.2-150}\n") #using nozzle height of 1.2x layer height
            f.write("\n// --- Begin Toolpath ---\n")

            # --- TOOLPATH PREPARATION ---
            # Pre-calculate Forward and Reverse paths ONCE to save computation
            path_forward = self.toolpath
            path_reverse = []
            
            for step in reversed(self.toolpath):
                # Create a shallow copy is sufficient since we replace top-level keys
                rev_step = step.copy()
                # Swap coordinates for the reverse path
                rev_step['coords_start'] = step['coords_end']
                rev_step['coords_end'] = step['coords_start']
                path_reverse.append(rev_step)

            def set_valve(mat_name, state):
                if mat_name in material_mapping:
                    idx = material_mapping[mat_name]
                    f.write(f"$OutBits[$port_{idx}] = {state}\n")

            # Track active materials (Reset at start)
            active_materials = set()
            # Track materials that have ever been activated (for first-activation dwell)
            ever_activated = set()

            # --- LAYER LOOP ---
            for layer_idx in range(1, num_layers + 1):
                f.write(f"\n//Options: Layer {layer_idx} of {num_layers}\n")
                
                # 1. Select Pre-calculated Path
                if layer_idx % 2 == 1:
                    layer_steps = path_forward
                    direction_str = "Forward"
                else:
                    layer_steps = path_reverse
                    direction_str = "Reverse"
                
                if not layer_steps:
                    continue

                # 2. Optimization: Smart Start
                # Check first step. If Travel, skip the logic and move directly to its End.
                start_step_index = 0
                first_step = layer_steps[0]
                end_step_index = len(layer_steps)
                last_step = layer_steps[-1]
                
                if first_step['type'] == 'travel':
                    # Skip the travel move entirely, start at its destination
                    initial_pos = first_step['coords_end']
                    start_step_index = 1
                    f.write(f"// Layer Start ({direction_str}): Skipped initial travel, moving to end\n")
                else:
                    initial_pos = first_step['coords_start']
                    start_step_index = 0
                    f.write(f"// Layer Start ({direction_str}): Moving to start\n")
                    
                    
                end_step_index = len(layer_steps)
                last_step = layer_steps[-1]
                if last_step['type'] == 'travel':
                    end_step_index = end_step_index-1
                    
                    
                
                # 3. Move to Initial Position of Layer (XY Only)
                f.write(f"G0 X{initial_pos[0]:.3f} Y{initial_pos[1]:.3f} F$speed_travel\n")
                
                # 4. Execute Steps for this Layer
                effective_steps = layer_steps[start_step_index:end_step_index]
                n_steps = len(effective_steps)
                
                for i, step in enumerate(effective_steps):
                    p_end = step['coords_end']
                    
                    # --- TRAVEL MOVE ---
                    if step['type'] == 'travel':
                        f.write(f"\n// L{layer_idx} Step {i+1}: Travel\n")
                        f.write(f"Dwell({dwell_time_after_travel})\n")
                        # Lift
                        f.write(f"G91 ; relative\n")
                        f.write(f"G0 C{self.lift_dist} F$speed_travel\n")
                        f.write(f"G90 ; absolute\n")
                        # Move
                        f.write(f"G0 X{p_end[0]:.3f} Y{p_end[1]:.3f} F$speed_travel\n")
                        # Land
                        f.write(f"G91 ; relative\n")
                        f.write(f"G0 C{-self.lift_dist} F$speed_travel\n")
                        f.write(f"G90 ; absolute\n")

                    # --- PRINT MOVE ---
                    else:
                        current_mats = set(step.get('materials', []))
                        f.write(f"\n// L{layer_idx} Step {i+1}: Print {list(current_mats)}\n")
                        
                        # Valve ON
                        mats_to_turn_on = current_mats - active_materials
                        for m in mats_to_turn_on:
                            set_valve(m, 1)
                        active_materials.update(mats_to_turn_on)

                        # First-time activations override other dwell cases
                        first_time_mats = mats_to_turn_on - ever_activated
                        ever_activated.update(mats_to_turn_on)

                        if first_time_mats:
                            f.write(f"Dwell({dwell_time_travel})\n")
                        elif i == 0:
                            f.write(f"Dwell({dwell_time_after_travel})\n")
                        else:
                            prev_step = effective_steps[i-1]
                            if prev_step['type'] == 'travel':
                                f.write(f"Dwell({dwell_time_after_travel})\n")
                            elif mats_to_turn_on:
                                f.write(f"Dwell({dwell_time_mat_change})\n")
                        
                        # Move
                        f.write(f"G1 X{p_end[0]:.3f} Y{p_end[1]:.3f} F$speed_print\n")
                        
                        # Valve OFF Logic
                        should_turn_off_all = False
                        next_mats = set()
                        
                        # Check next step IN THIS LAYER
                        if i + 1 >= n_steps:
                            should_turn_off_all = True
                        else:
                            next_step = effective_steps[i+1]
                            if next_step['type'] == 'travel':
                                should_turn_off_all = True
                            else:
                                next_mats = set(next_step.get('materials', []))
                        
                        if should_turn_off_all:
                            mats_to_turn_off = list(active_materials)
                            active_materials.clear()
                        else:
                            mats_to_turn_off = active_materials - next_mats
                            active_materials = active_materials - mats_to_turn_off
                            
                        for m in mats_to_turn_off:
                            set_valve(m, 0)

                # 5. Layer Transition (Lift C)
                # If there are more layers following this one, lift C
                if layer_idx < num_layers:
                    f.write(f"\n// End of Layer {layer_idx}, Lifting C by {layer_height}\n")
                    f.write("G91 ; relative\n")
                    f.write(f"G0 C{layer_height} F$speed_travel\n")
                    f.write("G90 ; absolute\n")
            
            # --- FOOTER ---
            f.write("\n// --- End of Toolpath ---\n")
            for idx in material_mapping.values():
                f.write(f"$OutBits[$port_{idx}] = 0\n")
            f.write("\n")
            f.write('SocketWriteString($clientSocket, "execTogglePressure([8, 9, 10]) ")\n\n')
            f.write('//SocketWriteString($clientSocket,"\\nEND PROGRAM")\n')
            f.write('Home(C)\n')
            f.write(f"G90 ; absolute\n")
            f.write("G0 X0 Y0 F$speed_travel // Return Home\n")
            
        print(f"G-code generation complete: {filename}")
        
    def visualize_material_status(self, foldername, printing_speed=20.0, traveling_speed=40.0, num_layers=1, layer_height=None, dwell_time_travel=0.2, dwell_time_after_travel=0.09, dwell_time_mat_change=0.08):
        """
        Visualizes the active status (ON/OFF) of each material over time.

        Args:
            foldername (str): Output directory.
            printing_speed (float): Speed in mm/s (or whatever unit matches distance).
            traveling_speed (float): Speed in mm/s.
            num_layers (int): Number of layers to simulate for total time calculation.
            layer_height (float): Z-lift between layers.
            dwell_time_travel (float): Long initial dwell emitted exactly once per material, immediately after that material's first activation.
            dwell_time_after_travel (float): Dwell before a travel move and at the first print of a layer / after a travel step (when no first-time activation applies).
            dwell_time_mat_change (float): Dwell at the start of a mid-run print move where existing materials are re-activated.
        """
        import matplotlib.pyplot as plt
        import math
        
        if layer_height is None:
            layer_height = self.layer_h
            
        save_dir = foldername
        os.makedirs(save_dir, exist_ok=True)
        print(f"--- Generating Material Status Plot in {save_dir} ---")

        # 1. Identify all unique materials
        materials_set = set()
        for step in self.toolpath:
            for m in step.get('materials', []):
                materials_set.add(m)
        all_materials = sorted(list(materials_set))
        
        # 2. Initialize Data Structures
        # We store (time, status) pairs to plot step functions
        timelines = {m: {'times': [0.0], 'status': [0]} for m in all_materials}
        
        current_time = 0.0
        current_pos = [0.0, 0.0]
        layer_1_time = 0.0
        
        total_dwell_time = 0.0
        total_print_time = 0.0
        total_travel_time = 0.0
        
        # Pre-calculate Forward and Reverse paths
        path_forward = self.toolpath
        path_reverse = []
        for step in reversed(self.toolpath):
            rev_step = step.copy()
            rev_step['coords_start'] = step['coords_end']
            rev_step['coords_end'] = step['coords_start']
            path_reverse.append(rev_step)

        active_materials = set()
        # Track materials that have ever been activated (for first-activation dwell)
        ever_activated = set()

        # 3. Process Toolpath
        for layer_idx in range(1, num_layers + 1):
            if layer_idx % 2 == 1:
                layer_steps = path_forward
            else:
                layer_steps = path_reverse
                
            if not layer_steps:
                continue
                
            # Smart Start Optimization (Trimming)
            first_step = layer_steps[0]
            if first_step['type'] == 'travel':
                initial_pos = first_step['coords_end']
                start_step_index = 1
            else:
                initial_pos = first_step['coords_start']
                start_step_index = 0
                
            end_step_index = len(layer_steps)
            last_step = layer_steps[-1]
            if last_step['type'] == 'travel':
                end_step_index = end_step_index - 1
                
            effective_steps = layer_steps[start_step_index:end_step_index]
            n_steps = len(effective_steps)
            
            # Repositioning move to initial_pos at the start of layer
            dist_to_initial = math.sqrt((initial_pos[0] - current_pos[0])**2 + (initial_pos[1] - current_pos[1])**2)
            repo_time = dist_to_initial / traveling_speed
            current_time += repo_time
            total_travel_time += repo_time
            current_pos = initial_pos
            
            for i, step in enumerate(effective_steps):
                p_end = step['coords_end']
                
                # --- TRAVEL MOVE ---
                if step['type'] == 'travel':
                    dwell_dur = dwell_time_after_travel
                    move_dist = math.sqrt((p_end[0] - current_pos[0])**2 + (p_end[1] - current_pos[1])**2)
                    travel_dur = (move_dist + 2 * self.lift_dist) / traveling_speed

                    step_total_dur = dwell_dur + travel_dur
                    total_dwell_time += dwell_dur
                    total_travel_time += travel_dur

                    # Update timelines ONLY during Layer 1 (all materials off during travel)
                    if layer_idx == 1:
                        for m in all_materials:
                            timelines[m]['times'].append(current_time)
                            timelines[m]['status'].append(0)
                            timelines[m]['times'].append(current_time + step_total_dur)
                            timelines[m]['status'].append(0)

                    current_time += step_total_dur
                    current_pos = p_end
                    
                # --- PRINT MOVE ---
                else:
                    current_mats = set(step.get('materials', []))
                    mats_to_turn_on = current_mats - active_materials
                    active_materials.update(mats_to_turn_on)

                    # First-time activations override other dwell cases
                    first_time_mats = mats_to_turn_on - ever_activated
                    ever_activated.update(mats_to_turn_on)

                    dwell_dur = 0.0
                    if first_time_mats:
                        dwell_dur = dwell_time_travel
                    elif i == 0:
                        dwell_dur = dwell_time_after_travel
                    else:
                        prev_step = effective_steps[i-1]
                        if prev_step['type'] == 'travel':
                            dwell_dur = dwell_time_after_travel
                        elif mats_to_turn_on:
                            dwell_dur = dwell_time_mat_change
                            
                    move_dist = math.sqrt((p_end[0] - current_pos[0])**2 + (p_end[1] - current_pos[1])**2)
                    print_dur = move_dist / printing_speed
                    
                    step_total_dur = dwell_dur + print_dur
                    total_dwell_time += dwell_dur
                    total_print_time += print_dur
                    
                    # Update timelines ONLY during Layer 1
                    if layer_idx == 1:
                        for m in all_materials:
                            is_active = 1 if m in active_materials else 0
                            timelines[m]['times'].append(current_time)
                            timelines[m]['status'].append(is_active)
                            # Maintain state until the end of this step
                            timelines[m]['times'].append(current_time + step_total_dur)
                            timelines[m]['status'].append(is_active)
                            
                    current_time += step_total_dur
                    current_pos = p_end
                    
                    # Valve OFF Logic
                    should_turn_off_all = False
                    next_mats = set()
                    if i + 1 >= n_steps:
                        should_turn_off_all = True
                    else:
                        next_step = effective_steps[i+1]
                        if next_step['type'] == 'travel':
                            should_turn_off_all = True
                        else:
                            next_mats = set(next_step.get('materials', []))
                            
                    if should_turn_off_all:
                        mats_to_turn_off = list(active_materials)
                        active_materials.clear()
                    else:
                        mats_to_turn_off = active_materials - next_mats
                        active_materials = active_materials - mats_to_turn_off
                        
            # Assert materials are cleared at the end of the layer
            assert not active_materials, f"Active materials not empty at end of layer {layer_idx}!"
            
            # Record layer 1 time and close out valves in timeline
            if layer_idx == 1:
                layer_1_time = current_time
                for m in all_materials:
                    timelines[m]['times'].append(layer_1_time)
                    timelines[m]['status'].append(0)
                    
            # Layer Transition (Lift C)
            if layer_idx < num_layers:
                layer_trans_time = layer_height / traveling_speed
                current_time += layer_trans_time
                total_travel_time += layer_trans_time
                
        # Return Home (G0 X0 Y0)
        home_dist = math.sqrt(current_pos[0]**2 + current_pos[1]**2)
        home_time = home_dist / traveling_speed
        current_time += home_time
        total_travel_time += home_time
        
        print("\n=== Printing Time Calculation Breakdown ===")
        print(f"Total Dwell Time:  {total_dwell_time:.2f} s")
        print(f"Total Print Time:  {total_print_time:.2f} s")
        print(f"Total Travel Time: {total_travel_time:.2f} s")
        print(f"Overall Time:      {current_time:.2f} s")
        
        diff = abs(current_time - (total_dwell_time + total_print_time + total_travel_time))
        assert diff < 1e-5, f"Time breakdown mismatch! diff={diff}"
        print("===========================================\n")
        
        # 4. Plot (Only for Layer 1)
        if layer_1_time == 0.0:
            return
            
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        
        y_ticks = []
        y_labels = []
        
        for i, m in enumerate(all_materials):
            color = self.color_map.get(m, 'blue')
            t = timelines[m]['times']
            s = np.array(timelines[m]['status'])
            
            y_vals = i + (s * 0.8) 
            
            ax.fill_between(t, i, y_vals, color=color, alpha=0.3, step='post')
            ax.step(t, y_vals, where='post', color=color, linewidth=2, label=m)
            
            y_ticks.append(i + 0.1)
            y_labels.append(m)

        ax.set_xlabel('Time (s)')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title(f'Material Status Over Time (Layer 1 Time: {layer_1_time:.2f} s)')
        # ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(save_dir, "material_status_timeline.svg")
        plt.savefig(filename, dpi=300)
        print(f"Saved status plot to: {filename}")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":

    # Create output subfolders
    os.makedirs(foldername, exist_ok=True)
    os.makedirs(foldername+"/figs", exist_ok=True)
    os.makedirs(foldername+"/data", exist_ok=True)
    os.makedirs(foldername+"/visualization", exist_ok=True)
    
    # check if in debug mode, which gives more intermediate plots
    debug = True # True # Set to True for more verbose output and intermediate plots 
    use_mst = False # if False, using our algorithm; if True, using mst
    
    # Step 1: Create unit cell and lattice (same as before)
    cell = UnitCell()
    
# ---------sample cases----------
    # # case 0: original lattice -- 0
    # # can test mst
    # L = 12
    # sL = np.sin(np.pi/3)*L
    # cL = np.cos(np.pi/3)*L
    # nodes = [[0, 0], [L, 0], [cL, sL],  [cL + L, sL], [0, sL + sL], [L, sL + sL]]
    # edges = [
    #     #### [node_u, node_v, material_name]
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat1'],
    #     [2, 3, 'mat1'],
    #     [2, 4, 'mat1'],
    #     [4, 5, 'mat1'],
    #     [0, 2, 'mat2'],
    #     [2, 5, 'mat2'],
    # ]
    
    # n=1
    # nX, nY =  4,2 #4,2  # 5*n,3*n #
    # tesX, tesY = L, sL + sL
    # m=0.5 * 11
    # offsets = {
    #     'mat1': [0.0, 0.0],
    #     'mat2': [0.5*L,0.5*L]#[m*L, 0.0*L] #
    # }
    
    # # case 0.1: original lattice -- 0
    # L = 25/2
    # sL = np.sin(np.pi/3)*L
    # cL = np.cos(np.pi/3)*L
    # nodes = [[0, 0], [L, 0], [cL, sL],  [cL + L, sL], [0, sL + sL], [L, sL + sL]]
    # edges = [
    #     #### [node_u, node_v, material_name]
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat1'],
    #     [2, 3, 'mat1'],
    #     [2, 4, 'mat1'],
    #     [4, 5, 'mat1'],
    #     [0, 2, 'mat2'],
    #     [2, 5, 'mat2'],
    # ]
    # n = 1
    # nX, nY =  5*n,3  #5*n,3*n # 4,2 #
    # tesX, tesY = L, sL + sL
    # m=0.00
    # offsets = {
    #     'mat1': [0.0, 0.0],
    #     'mat2': [25.0 ,0.0]#[m*L, 0.0*L],[m*L, 0.0*L] #[100,0]#
    # }
    
    # # case 1: original lattice -- 1
    # L = 18
    # nodes = [[0,0], [L/2,0], [L,0], [L,L/2], [L,L], [L/2,L], [0,L], [0,L/2], [L/2,L/2]]
    # edges = [
    #     # [0, 1, 'mat1'],
    #     # [1, 2, 'mat1'],
    #     # [2, 3, 'mat1'],
    #     # [3, 4, 'mat1'],
    #     # [4, 5, 'mat1'],
    #     # [5, 6, 'mat1'],
    #     # [6, 7, 'mat1'],
    #     # [1, 8, 'mat1'],
    #     # [3, 8, 'mat1'],
    #     # [5, 8, 'mat1'],
    #     # [7, 8, 'mat1'],
    #     # [0, 7, 'mat1'],
    #     [1, 3, 'mat2'],
    #     [3, 5, 'mat2'],
    #     [5, 7, 'mat2'],
    #     [1, 7, 'mat2']
    # ]
    # n = 1
    # nX, nY = 2,2#2*n,2
    # tesX, tesY = L,L
    # m=0.25 * 10
    # offsets = {
    #     'mat1': [0.0, 0.0],
    #     'mat2': [m*L, 0.00] #[0.5*L, 0.5*L] # or [L,L], or 
    # }

    # case 2: original lattice -- 2
    L = 9
    sL = np.sin(np.pi/3)*L
    cL = np.cos(np.pi/3)*L
    nodes = [[0,0], [L,0], [2*L,0], [3*L,0],[3*L+cL,sL], [3*L,2*sL],[2*L,2*sL], [L,2*sL], [0,2*sL],[cL,sL],[cL+L,sL],[cL+2*L,sL]]
    edges = [
        [0,1,'mat1'],
        [2,3,'mat1'],
        [3,4,'mat1'],
        [3,11,'mat1'],
        [4,5,'mat1'],
        [5,6,'mat1'],
        [7,8,'mat1'],
        [8,9,'mat1'],
        [0,9,'mat1'],
        [9,10,'mat1'],
        [1,10,'mat1'],
        [7,10,'mat1'],
        [2,10,'mat1'],
        [6,10,'mat1'],
        [10,11,'mat1'],
        [11,5,'mat1'],
        [6,7,'mat2'],
        [1,2,'mat2'],
        [1,9,'mat2'],
        [7,9,'mat2'],
        [2,11,'mat2'],
        [6,11,'mat2'],
        [11,4,'mat2']
    ]
    nX, nY = 2,4 #1, 2 #
    tesX, tesY = 3*L,sL*2
    offsets = {
        'mat1': [0.0, 0.0],
        'mat2':  [L, 0] #[0, 0] #[L,L]  #
    }


    # # case 3: original lattice -- 3
    # L = 9
    # nodes = [[0, 0], [L, 0], [L,L],  [0,L]]
    # edges = [
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat1'],
    #     [2, 3, 'mat1'],
    #     [0, 3, 'mat1'],
    #     [0, 2, 'mat2']
    # ]
    
    # n = 0.5
    # nX, nY = 3,3#6*n, 6 #3*n,3
    # tesX, tesY = L,L
    # m= 0.5 * 2
    # offsets = {
    #     'mat1': [0.0, 0.0],
    #     'mat2': [m*L, 0.0] #[0,0] #
    # }

    # # case 4 for co-design
    # L = 12

    # nodes = [[0, 0], [0.5 * L, 0], [L, 0], [L, 0.5 * L], [L, L], [0.5 * L, L], [0, L], [0, 0.5 * L], [0.5*L, 0.5*L]]
   
    # edges = [
    #     [0, 1, 'mat1'], [1, 2, 'mat1'], [2, 3, 'mat1'],
    #     [3, 4, 'mat1'], [4, 5, 'mat1'], [5, 6, 'mat1'],
    #     [6, 7, 'mat1'], [7, 0, 'mat1'],
    #     [1, 8, 'mat2'], [3, 8, 'mat2'],     [5, 8, 'mat2'], [7, 8, 'mat2']
    # ]
    # n = 1
    # nX, nY = 3*n, 3
    # tesX, tesY = L, L
    # m = 0.500 * 7.000 #characteristic length is half
    # offsets = {
    #     'mat1': [0.0, 0.0],
    #     'mat2': [m*L, 0.*L] #[m*L, m*L]
    # }
    
    
    # # case 4.1 -- PRINT for co-design
    # L = 25/2.5

    # nodes = [[0, 0], [0.5 * L, 0], [L, 0], [L, 0.5 * L], [L, L], [0.5 * L, L], [0, L], [0, 0.5 * L], [0.5*L, 0.5*L]]

    # edges = [
    #     [0, 1, 'mat1'], [1, 2, 'mat1'], [2, 3, 'mat1'],
    #     [3, 4, 'mat1'], [4, 5, 'mat1'], [5, 6, 'mat1'],
    #     [6, 7, 'mat1'], [7, 0, 'mat1'],
    #     [1, 8, 'mat2'], [3, 8, 'mat2'], [5, 8, 'mat2'], [7, 8, 'mat2']
    # ]
    # n = 1
    # nX, nY = 3*n, 3
    # tesX, tesY = L, L
    # offsets = {
    #     'mat1': [0.0, 0.0],
    #     'mat2': [25.0, 0.0] #[m*L, m*L]
    # }

    # # case 5 for single material with many connected components
    # # side case
    # # can test mst
    # L = 12
    # nodes = [[0, 0], [L, 0], [L,L],  [0,L],  [2*L,-L]]
    # edges = [
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat1'],
    #     [2, 3, 'mat1'],
    #     [0, 3, 'mat1'],
    #     [1, 4, 'mat1']
    # ]
    # n=1
    # nX, nY = 4*n,4*n
    # tesX, tesY = 2.5*L, 1.5*L #L,L #
    # offsets = {
    #     'mat1': [0.0, 0.0]
    # }
    
    
    # # case 5.5 for single material with many connected components
    # # can test mst
    # L = 12
    # sL = np.sin(np.pi/12)*L
    # cL = np.cos(np.pi/12)*L
    # nodes = [[0, 0], [L, sL], [cL/2+L,L+sL],  [cL/2,L]]
    # edges = [
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat1'],
    #     [2, 3, 'mat1'],
    #     [0, 3, 'mat1'],
    # ]
    # n=1
    # nX, nY = 4*n,4*n
    # tesX, tesY =  2*L,2*L
    # offsets = {
    #     'mat1': [0.0, 0.0]
    # }
    
    # # case 5.1 PRINT for single material with many connected components
    # L = 12
    # nodes = [[0, 0], [L, 0], [L,L],  [0,L]]
    # edges = [
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat1'],
    #     [2, 3, 'mat1'],
    #     [0, 3, 'mat1'],
    # ]
    # n=1
    # nX, nY = 5*n,3*n
    # tesX, tesY =    L,L #L+5, L+5 #
    # offsets = {
    #     'mat1': [0.0, 0.0]
    # }

    # # case 6 for single material with many connected components
    # L = 6
    # sL = np.sin(np.pi/3)*L
    # cL = np.cos(np.pi/3)*L
    # nodes = [[0, 0], [cL, sL], [L, sL + sL]]
    # edges = [
    #     # [node_u, node_v, material_name]
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat1'],
    # ]
    # n=1
    # nX, nY = n*5,n*3
    # tesX, tesY = L, sL + sL
    # offsets = {
    #     'mat1': [0, 0.0]
    # }
    


    # # case 7 for many materials with many connected components
    # L = 6
    # sL = np.sin(np.pi/3)*L
    # cL = np.cos(np.pi/3)*L
    # nodes = [[0, 0], [L, 0], [cL, sL],  [cL + L, sL], [0, sL + sL], [L, sL + sL]]
    # edges = [
    #     # [node_u, node_v, material_name]
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat3'],
    #     [2, 3, 'mat1'],
    #     [2, 4, 'mat3'],
    #     [4, 5, 'mat1'],
    #     [0, 2, 'mat2'],
    #     [2, 5, 'mat2'],
    # ]
    # nX, nY = 6, 3
    # tesX, tesY = L, sL + sL
    # offsets = {
    #     'mat1': [0.0, 0.0],
    #     'mat2': [0.5*L, 0.5*L],
    #     'mat3': [0.2*L, 0.9*L],
    # }


    # # case 8 for addressing Jochen's original concern
    # L = 12
    # nodes = [ [L/3, 2*L/3], [0, L/3], [L/3, 0], [2*L/3,L/3],[L, L/3]]
    # edges = [
    #     # [node_u, node_v, material_name]
    #     [0, 1, 'mat1'],
    #     [1, 2, 'mat1'],
    #     [3, 2, 'mat1'],
    #     [0, 3, 'mat1'],
    #     [3, 4, 'mat1']
    # ]
    # nX, nY =2,1
    # tesX, tesY = L, 2*L/3
    # #x0, x1 = -0.001, nX * tesX + 0.001 -3
    # offsets = {
    #     'mat1': [0.0, 0.0]
    # }
    
# ---------sample cases: end----------
    
    # add nodes and edges to unit cell
    for x, y in nodes:
        cell.add_node(x, y)
    
    cell.add_connections(edges)
    cell.visualize(foldername+'/figs/Fig1_unitCell.svg')

    
    # Step 2: Create lattice  
    # define boundary limits  
    x0, x1 = -0.001, nX * tesX + 0.001 
    y0, y1 = -0.001, nY * tesY + 0.001

    # create lattice
    lattice = Lattice(unit_cell=cell, n_x=nX, n_y=nY, tes_x=tesX, tes_y=tesY,
                      x_min=x0, x_max=x1, y_min=y0, y_max=y1)
    if debug: print(f"Original lattice: {lattice}")
    
    # # # [optional] apply noise to node positions
    # # lattice.apply_noise(2, seed=42)

    # # [optional] randomize materials (uncomment one of the two modes below)
    # # lattice.randomize_materials(seed=42)                         # shuffle mode
    # lattice.randomize_materials({'mat1': 0.5, 'mat2': 0.5}, seed=42)   # sample mode
        
    # Visualize lattice
    lattice.visualize(foldername+'/figs/Fig2_lattice.svg', show_nodes=False)
    


    # Step 3: Create MixedLattice for nozzle offsets: this will be the toolpath for the sigle toolhead with multiple nozzles
    
    mixed_lattice = MixedLattice.from_lattice_with_offsets(lattice, offsets)
    if debug: 
        print(f"\nMixed lattice: {mixed_lattice}")
        print(f"Stats: {mixed_lattice.get_stats()}")  
    
    # Visualize
    mixed_lattice.visualize(save_path=f'{foldername}/figs/mixed_lattice.svg',
                           title='Mixed Lattice with Multi-Material Edges',
                           highlight_mixed=True)
    
    # Step 4: Solve RPP
    # Initialize Solver
    solver = RuralPostmanSolver(mixed_lattice)
    # Solve for toolpath
    toolpath = solver.solve(start_coords=[0.0, 0.0],debug = debug,use_mst=use_mst)
    # toolpath = solver.solve_random(seed=42, start_coords=[0.0, 0.0], debug=debug) # for comparision: random edge order
    # toolpath = solver.solve_greedy(start_coords=[0.0, 0.0], debug=debug) # for comparision: greedy edge order
    # toolpath = solver.solve_line(start_coords=[0.0, 0.0], angle_tol_deg=5.0, debug=debug) # for comparision: try to preserve straight line while greedy
    
    # Visualize
    solver.visualize_route(save_path=f'{foldername}/figs/RPP_solution.svg')
    
    # Export Toolpath data
    with open(f'{foldername}/data/toolpath.json', 'w') as f:
        json.dump(toolpath, f, indent=2)
    if debug: print("Toolpath exported.")
    
    
    
    # step 5: visualization
    visualizer = ToolpathVisualizer(
        toolpath_source = toolpath, 
        material_offsets = offsets, 
        lift_distance = 5.0,
        layer_thickness = 0.8
    )
    
    # # # visualization
    # visualizer.visualize_layer_3d(foldername=foldername+"/visualization", fixed_arrow_size=0.5,debug=debug)
    
    # visualizer.visualize_material_paths(foldername=foldername+"/visualization", fixed_arrow_size=0.5,show_labels=True,debug=debug)
    
    # printing settins below
    dwell_time_travel = 0.2       # long initial dwell, emitted exactly once per material right after its first activation
    dwell_time_after_travel =  0.09 # dwell before a travel block and at first print of a layer / after a travel step  # 0.05 for single materil, 0.09 for multiple materials
    dwell_time_mat_change = 0.08   # dwell at the start of a mid-run print move when materials re-activate             # 0.05 for single materil, 0.08 for multiple materials

    # Speeds used for BOTH total-time calculation and G-code export (units: mm/s, matching distance units)
    printing_speed = 20.0
    traveling_speed = 40.0

    # # visualize materials' valve on/off over time
    visualizer.visualize_material_status(foldername=foldername+"/figs", num_layers=1, printing_speed=printing_speed, traveling_speed=traveling_speed, dwell_time_travel=dwell_time_travel, dwell_time_after_travel=dwell_time_after_travel, dwell_time_mat_change=dwell_time_mat_change)

    # # # for videos
    # visualizer.visualize_frames_firstlayer(foldername=foldername+"/visualization", fixed_arrow_size=0.5,show_labels=False,debug=debug)


    # # step 6: export g-code
    visualizer.export_to_gcode(filename=foldername+"/data/GCode.gcode",layer_height=0.8,num_layers=8, printing_speed=printing_speed, traveling_speed=traveling_speed, dwell_time_travel=dwell_time_travel, dwell_time_after_travel=dwell_time_after_travel, dwell_time_mat_change=dwell_time_mat_change)
    


# end of file