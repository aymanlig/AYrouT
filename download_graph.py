#!/usr/bin/env python3
"""
AYrouT - Graph Data Acquisition Script
Downloads and preprocesses the Agadir street network from OpenStreetMap.
Run this ONCE before starting the server.
"""

import os
import osmnx as ox
import networkx as nx
import time
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GRAPH_FILEPATH = os.path.join(DATA_DIR, "agadir_graph.graphml")
METADATA_FILEPATH = os.path.join(DATA_DIR, "graph_metadata.json")


def download_agadir_graph():
    """
    Download the drivable street network for Agadir, Morocco.
    Saves to local .graphml file to avoid repeated API calls.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(GRAPH_FILEPATH):
        print(f"[INFO] Graph already exists at: {GRAPH_FILEPATH}")
        print("[INFO] Delete the file and re-run to force a fresh download.")
        return load_existing_graph()

    print("=" * 60)
    print("  AYrouT - Downloading Agadir Street Network")
    print("=" * 60)
    print("[STEP 1/4] Querying OpenStreetMap for Agadir, Morocco...")
    print("           Network type: drive")
    print("           This may take 1-3 minutes depending on connection...")
    print()

    start_time = time.time()

    try:
        # Download the drive network for Agadir with a buffer
        # We use a larger area to cover greater Agadir (including Inezgane, Ait Melloul)
        G = ox.graph_from_place(
            "Agadir, Souss-Massa, Morocco",
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )
    except Exception as e:
        print(f"[WARNING] Place query failed: {e}")
        print("[FALLBACK] Downloading by bounding box (Greater Agadir area)...")
        # Bounding box for Greater Agadir
        # North: 30.50, South: 30.35, East: -9.45, West: -9.70
        G = ox.graph_from_bbox(
            north=30.50,
            south=30.35,
            east=-9.45,
            west=-9.70,
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )

    download_time = time.time() - start_time
    print(f"[STEP 2/4] Download complete in {download_time:.1f} seconds.")

    # Project the graph to get proper distances in meters
    print("[STEP 3/4] Processing graph attributes...")

    # Add edge speeds and travel times based on OSM tags
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Graph statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Calculate bounding box
    nodes_data = ox.graph_to_gdfs(G, edges=False)
    bbox = {
        "north": float(nodes_data["y"].max()),
        "south": float(nodes_data["y"].min()),
        "east": float(nodes_data["x"].max()),
        "west": float(nodes_data["x"].min()),
        "center_lat": float(nodes_data["y"].mean()),
        "center_lon": float(nodes_data["x"].mean()),
    }

    print(f"           Nodes: {num_nodes:,}")
    print(f"           Edges: {num_edges:,}")
    print(f"           Bounding Box: {bbox}")

    # Save graph
    print(f"[STEP 4/4] Saving graph to: {GRAPH_FILEPATH}")
    ox.save_graphml(G, filepath=GRAPH_FILEPATH)

    # Save metadata
    metadata = {
        "city": "Agadir, Morocco",
        "network_type": "drive",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "bbox": bbox,
        "download_time_seconds": round(download_time, 1),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METADATA_FILEPATH, "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  SUCCESS! Graph saved in {total_time:.1f} seconds.")
    print(f"  File size: {os.path.getsize(GRAPH_FILEPATH) / (1024*1024):.1f} MB")
    print("=" * 60)

    return G


def load_existing_graph():
    """Load an already-downloaded graph."""
    print("[INFO] Loading existing graph...")
    start = time.time()
    G = ox.load_graphml(filepath=GRAPH_FILEPATH)
    print(f"[INFO] Graph loaded in {time.time() - start:.1f}s "
          f"({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")
    return G


if __name__ == "__main__":
    download_agadir_graph()