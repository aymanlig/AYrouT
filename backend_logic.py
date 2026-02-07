#!/usr/bin/env python3
"""
AYrouT - Backend Logic Engine
==============================
Custom high-performance routing engine for logistics and delivery in Agadir, Morocco.

This module implements:
  1. Graph loading and caching from OSM data
  2. Custom cost function with multi-factor edge weighting
  3. A* pathfinding with the custom cost model
  4. Turn-by-turn navigation instruction generation
  5. Audio synthesis of navigation instructions via gTTS
  6. Geocoding support for place name resolution

Mathematical Model:
  Cost(edge) = (Distance / EffectiveSpeed) * α * β + Penalty

  Where:
    - Distance      : edge length in meters (from OSM geometry)
    - EffectiveSpeed : speed assigned by road class (km/h → m/s)
    - α (alpha)      : road quality factor (surface condition multiplier)
    - β (beta)       : traffic density heuristic (road class congestion factor)
    - Penalty        : fixed time cost at signalized intersections (seconds)

Author: AYrouT Engineering Team
"""

import os
import math
import hashlib
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from gtts import gTTS

# ─────────────────────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────────────────────

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
GRAPH_FILEPATH = os.path.join(DATA_DIR, "agadir_graph.graphml")

# Speed assignments by OSM highway tag (km/h)
SPEED_MAP: Dict[str, float] = {
    "motorway": 120.0,
    "motorway_link": 80.0,
    "trunk": 80.0,
    "trunk_link": 60.0,
    "primary": 60.0,
    "primary_link": 45.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "unclassified": 30.0,
    "service": 20.0,
    "track": 15.0,
    "road": 30.0,
}
DEFAULT_SPEED: float = 30.0  # km/h fallback

# Road quality multipliers (α factor)
SURFACE_QUALITY_MAP: Dict[str, float] = {
    "paved": 1.0,
    "asphalt": 1.0,
    "concrete": 1.0,
    "concrete:plates": 1.05,
    "concrete:lanes": 1.05,
    "paving_stones": 1.1,
    "sett": 1.15,
    "cobblestone": 1.2,
    "compacted": 1.3,
    "fine_gravel": 1.3,
    "gravel": 1.4,
    "dirt": 1.5,
    "earth": 1.5,
    "mud": 2.0,
    "sand": 2.0,
    "unpaved": 1.5,
    "ground": 1.5,
}
DEFAULT_SURFACE_FACTOR: float = 1.0

# Traffic density heuristic (β factor)
TRAFFIC_DENSITY_MAP: Dict[str, float] = {
    "motorway": 1.1,
    "motorway_link": 1.1,
    "trunk": 1.25,
    "trunk_link": 1.2,
    "primary": 1.3,
    "primary_link": 1.2,
    "secondary": 1.15,
    "secondary_link": 1.1,
    "tertiary": 1.05,
    "tertiary_link": 1.0,
    "residential": 1.0,
    "living_street": 1.0,
    "unclassified": 1.0,
    "service": 1.0,
    "track": 1.0,
    "road": 1.05,
}
DEFAULT_TRAFFIC_FACTOR: float = 1.0

# Intersection penalty (seconds)
TRAFFIC_SIGNAL_PENALTY: float = 30.0  # seconds added at traffic lights
STOP_SIGN_PENALTY: float = 10.0  # seconds at stop signs
CROSSING_PENALTY: float = 5.0  # seconds at pedestrian crossings

# Navigation constants
TURN_THRESHOLD_DEGREES: float = 45.0
SHARP_TURN_THRESHOLD: float = 120.0
UTURN_THRESHOLD: float = 160.0

# Geocoding
GEOCODER_USER_AGENT = "ayrout_routing_engine_v1"
GEOCODER_TIMEOUT = 10

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AYrouT.Engine")


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class RoutePoint:
    """Represents a geographic point on the route."""
    lat: float
    lon: float
    node_id: Optional[int] = None
    name: Optional[str] = None


@dataclass
class NavigationInstruction:
    """A single turn-by-turn navigation instruction."""
    step_number: int
    instruction: str
    direction: str  # "left", "right", "straight", "sharp_left", "sharp_right", "uturn"
    distance_meters: float
    cumulative_distance: float
    bearing: float
    angle_change: float
    street_name: str
    point: RoutePoint


@dataclass
class RouteResult:
    """Complete result of a route calculation."""
    success: bool
    error_message: Optional[str] = None
    origin: Optional[RoutePoint] = None
    destination: Optional[RoutePoint] = None
    total_distance_km: float = 0.0
    total_time_minutes: float = 0.0
    total_cost: float = 0.0
    route_nodes: List[int] = field(default_factory=list)
    route_coordinates: List[Tuple[float, float]] = field(default_factory=list)
    instructions: List[NavigationInstruction] = field(default_factory=list)
    audio_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON response."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "origin": {
                "lat": self.origin.lat,
                "lon": self.origin.lon,
                "name": self.origin.name,
            } if self.origin else None,
            "destination": {
                "lat": self.destination.lat,
                "lon": self.destination.lon,
                "name": self.destination.name,
            } if self.destination else None,
            "total_distance_km": round(self.total_distance_km, 2),
            "total_time_minutes": round(self.total_time_minutes, 2),
            "total_cost": round(self.total_cost, 4),
            "route_coordinates": self.route_coordinates,
            "instructions": [
                {
                    "step": inst.step_number,
                    "instruction": inst.instruction,
                    "direction": inst.direction,
                    "distance_m": round(inst.distance_meters, 1),
                    "cumulative_distance_m": round(inst.cumulative_distance, 1),
                    "bearing": round(inst.bearing, 1),
                    "angle_change": round(inst.angle_change, 1),
                    "street_name": inst.street_name,
                    "point": {"lat": inst.point.lat, "lon": inst.point.lon},
                }
                for inst in self.instructions
            ],
            "audio_file": self.audio_file,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────
# Core Routing Engine Class
# ─────────────────────────────────────────────────────────────

class AYrouTEngine:
    """
    AYrouT Custom Routing Engine
    =============================
    High-performance routing engine built on raw OpenStreetMap data
    with custom mathematical cost functions for optimal logistics routing.

    Usage:
        engine = AYrouTEngine()
        engine.initialize()
        result = engine.calculate_route(start_lat, start_lon, end_lat, end_lon)
    """

    def __init__(self):
        self.graph: Optional[nx.MultiDiGraph] = None
        self.graph_projected: Optional[nx.MultiDiGraph] = None
        self._is_initialized: bool = False
        self._geocoder: Optional[Nominatim] = None
        self._node_penalties: Dict[int, float] = {}
        self._edge_costs: Dict[Tuple[int, int, int], float] = {}
        self._graph_center: Tuple[float, float] = (30.4278, -9.5981)  # Agadir default

        # Ensure directories exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(AUDIO_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Initialize the engine: load graph, precompute costs.
        Must be called before any routing operations.
        """
        logger.info("=" * 50)
        logger.info("  AYrouT Engine Initialization")
        logger.info("=" * 50)

        try:
            # Step 1: Load graph
            self._load_graph()

            # Step 2: Precompute node penalties
            self._precompute_node_penalties()

            # Step 3: Precompute edge costs using the custom formula
            self._precompute_edge_costs()

            # Step 4: Initialize geocoder
            self._geocoder = Nominatim(
                user_agent=GEOCODER_USER_AGENT,
                timeout=GEOCODER_TIMEOUT,
            )

            # Step 5: Calculate graph center for map display
            self._calculate_graph_center()

            self._is_initialized = True
            logger.info("Engine initialized successfully!")
            logger.info(f"  Graph center: {self._graph_center}")
            logger.info(f"  Nodes: {self.graph.number_of_nodes():,}")
            logger.info(f"  Edges: {self.graph.number_of_edges():,}")
            logger.info(f"  Precomputed costs: {len(self._edge_costs):,}")
            logger.info(f"  Node penalties: {len(self._node_penalties):,}")
            return True

        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize AYrouT engine: {e}")

    def _load_graph(self):
        """Load the OSM graph from local file or download if missing."""
        if os.path.exists(GRAPH_FILEPATH):
            logger.info(f"Loading graph from: {GRAPH_FILEPATH}")
            start = time.time()
            self.graph = ox.load_graphml(filepath=GRAPH_FILEPATH)
            elapsed = time.time() - start
            logger.info(f"Graph loaded in {elapsed:.2f}s")
        else:
            logger.info("Graph file not found. Downloading from OSM...")
            logger.info("Querying: Agadir, Souss-Massa, Morocco (drive network)")
            try:
                self.graph = ox.graph_from_place(
                    "Agadir, Souss-Massa, Morocco",
                    network_type="drive",
                    simplify=True,
                    retain_all=False,
                    truncate_by_edge=True,
                )
            except Exception:
                logger.warning("Place query failed, using bounding box fallback.")
                self.graph = ox.graph_from_bbox(
                    north=30.50, south=30.35,
                    east=-9.45, west=-9.70,
                    network_type="drive",
                    simplify=True,
                )

            # Add speed and travel time attributes
            self.graph = ox.add_edge_speeds(self.graph)
            self.graph = ox.add_edge_travel_times(self.graph)

            # Save for future use
            ox.save_graphml(self.graph, filepath=GRAPH_FILEPATH)
            logger.info(f"Graph saved to: {GRAPH_FILEPATH}")

    def _calculate_graph_center(self):
        """Calculate the geographic center of the loaded graph."""
        try:
            lats = [data["y"] for _, data in self.graph.nodes(data=True)]
            lons = [data["x"] for _, data in self.graph.nodes(data=True)]
            self._graph_center = (
                sum(lats) / len(lats),
                sum(lons) / len(lons),
            )
        except Exception:
            self._graph_center = (30.4278, -9.5981)  # Agadir fallback

    @property
    def center(self) -> Tuple[float, float]:
        """Return the geographic center of the graph."""
        return self._graph_center

    @property
    def is_ready(self) -> bool:
        """Check if the engine is initialized and ready."""
        return self._is_initialized

    # ─────────────────────────────────────────────────────────
    # Custom Cost Function — THE MATHEMATICAL MODEL
    # ─────────────────────────────────────────────────────────

    def _get_edge_speed(self, edge_data: Dict) -> float:
        """
        Determine effective speed for an edge based on OSM highway tag.

        Returns speed in m/s.
        """
        highway = edge_data.get("highway", "residential")

        # Handle list values (multiple tags)
        if isinstance(highway, list):
            highway = highway[0]

        speed_kmh = SPEED_MAP.get(str(highway), DEFAULT_SPEED)

        # Check for maxspeed tag override
        maxspeed = edge_data.get("maxspeed")
        if maxspeed:
            try:
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                if isinstance(maxspeed, str):
                    # Remove non-numeric characters
                    cleaned = "".join(c for c in maxspeed if c.isdigit() or c == ".")
                    if cleaned:
                        parsed_speed = float(cleaned)
                        if 5 <= parsed_speed <= 150:
                            speed_kmh = parsed_speed
            except (ValueError, TypeError):
                pass

        # Convert km/h to m/s
        speed_ms = speed_kmh / 3.6
        return max(speed_ms, 1.0)  # Minimum 1 m/s to avoid division by zero

    def _get_road_quality_factor(self, edge_data: Dict) -> float:
        """
        Calculate road quality factor (α) based on surface condition.

        α = 1.0 for good paved roads
        α > 1.0 for degraded or unpaved surfaces (increases cost)
        """
        surface = edge_data.get("surface", None)

        if surface is None:
            return DEFAULT_SURFACE_FACTOR

        if isinstance(surface, list):
            surface = surface[0]

        surface = str(surface).lower().strip()
        return SURFACE_QUALITY_MAP.get(surface, DEFAULT_SURFACE_FACTOR)

    def _get_traffic_density_factor(self, edge_data: Dict) -> float:
        """
        Calculate traffic density heuristic (β) based on road classification.

        Primary and trunk roads get higher β values to simulate
        typical congestion patterns in Agadir.

        β = 1.0 for low-traffic roads
        β > 1.0 for roads with expected higher traffic density
        """
        highway = edge_data.get("highway", "residential")

        if isinstance(highway, list):
            highway = highway[0]

        highway = str(highway).lower().strip()
        return TRAFFIC_DENSITY_MAP.get(highway, DEFAULT_TRAFFIC_FACTOR)

    def _get_edge_length(self, edge_data: Dict) -> float:
        """Get edge length in meters."""
        length = edge_data.get("length", 0)
        try:
            return float(length)
        except (ValueError, TypeError):
            return 100.0  # Default 100m if unknown

    def _compute_single_edge_cost(
        self, u: int, v: int, edge_data: Dict
    ) -> float:
        """
        Compute the custom cost for a single edge using the formula:

        Cost = (Distance / Speed) * α * β + NodePenalty

        Where:
          - Distance (meters) / Speed (m/s) = base travel time in seconds
          - α = road quality factor
          - β = traffic density heuristic
          - NodePenalty = fixed time at intersections/signals at destination node

        Returns:
            Cost in seconds (travel time equivalent).
        """
        distance = self._get_edge_length(edge_data)
        speed = self._get_edge_speed(edge_data)
        alpha = self._get_road_quality_factor(edge_data)
        beta = self._get_traffic_density_factor(edge_data)

        # Base travel time in seconds
        base_time = distance / speed

        # Apply multipliers
        weighted_time = base_time * alpha * beta

        # Add intersection penalty at the destination node
        node_penalty = self._node_penalties.get(v, 0.0)

        total_cost = weighted_time + node_penalty

        return max(total_cost, 0.01)  # Ensure positive cost

    def _precompute_node_penalties(self):
        """
        Precompute time penalties for intersection nodes.

        Checks OSM tags for:
          - traffic_signals → 30s penalty
          - stop signs → 10s penalty
          - crossings → 5s penalty
        """
        logger.info("Precomputing node penalties...")
        penalty_count = 0

        for node_id, data in self.graph.nodes(data=True):
            penalty = 0.0

            # Check highway tag on node
            highway_tag = data.get("highway", "")
            if isinstance(highway_tag, list):
                highway_tag = " ".join(str(t) for t in highway_tag)
            highway_tag = str(highway_tag).lower()

            if "traffic_signals" in highway_tag:
                penalty += TRAFFIC_SIGNAL_PENALTY
            elif "stop" in highway_tag:
                penalty += STOP_SIGN_PENALTY
            elif "crossing" in highway_tag:
                penalty += CROSSING_PENALTY

            # Also check degree (high-degree nodes are likely major intersections)
            # Add a small penalty proportional to the number of connections
            degree = self.graph.degree(node_id)
            if degree >= 4:
                # Minor intersection penalty for complex junctions
                penalty += (degree - 3) * 2.0  # 2 seconds per extra connection

            if penalty > 0:
                self._node_penalties[node_id] = penalty
                penalty_count += 1

        logger.info(f"  Penalized nodes: {penalty_count:,}")

    def _precompute_edge_costs(self):
        """
        Precompute the custom cost for every edge in the graph.
        Stores results in self._edge_costs and also sets a 'custom_cost'
        attribute on each edge for use with NetworkX shortest path algorithms.
        """
        logger.info("Precomputing edge costs with custom formula...")
        logger.info("  Formula: Cost = (D/S) × α × β + P")
        start = time.time()

        total_edges = 0
        total_cost = 0.0

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            cost = self._compute_single_edge_cost(u, v, data)
            self._edge_costs[(u, v, key)] = cost

            # Set the custom_cost attribute on the edge in the graph
            self.graph[u][v][key]["custom_cost"] = cost

            total_edges += 1
            total_cost += cost

        elapsed = time.time() - start
        avg_cost = total_cost / max(total_edges, 1)

        logger.info(f"  Edges processed: {total_edges:,}")
        logger.info(f"  Average cost: {avg_cost:.2f}s")
        logger.info(f"  Computation time: {elapsed:.3f}s")

    # ─────────────────────────────────────────────────────────
    # A* Pathfinding with Custom Heuristic
    # ─────────────────────────────────────────────────────────

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate the Haversine distance between two points in meters.
        Used as the A* heuristic.
        """
        R = 6371000  # Earth's radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _astar_heuristic(self, node1: int, node2: int) -> float:
        """
        A* heuristic function: estimated minimum cost from node1 to node2.

        Uses Haversine distance divided by maximum possible speed
        to get an admissible (never overestimates) heuristic.
        """
        n1_data = self.graph.nodes[node1]
        n2_data = self.graph.nodes[node2]

        distance = self._haversine_distance(
            n1_data["y"], n1_data["x"],
            n2_data["y"], n2_data["x"],
        )

        # Maximum speed in network: 120 km/h = 33.33 m/s
        max_speed = 120.0 / 3.6  # m/s

        # Minimum possible time (admissible heuristic)
        return distance / max_speed

    def find_nearest_node(self, lat: float, lon: float) -> int:
        """Find the nearest graph node to the given coordinates."""
        return ox.nearest_nodes(self.graph, X=lon, Y=lat)

    def calculate_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        generate_audio: bool = True,
    ) -> RouteResult:
        """
        Calculate the optimal route between two points.

        Args:
            start_lat: Origin latitude
            start_lon: Origin longitude
            end_lat: Destination latitude
            end_lon: Destination longitude
            generate_audio: Whether to generate TTS audio instructions

        Returns:
            RouteResult with full route details, instructions, and audio path.
        """
        if not self._is_initialized:
            return RouteResult(
                success=False,
                error_message="Engine not initialized. Call initialize() first.",
            )

        logger.info(f"Calculating route: ({start_lat:.4f}, {start_lon:.4f}) "
                     f"→ ({end_lat:.4f}, {end_lon:.4f})")
        route_start_time = time.time()

        try:
            # Step 1: Find nearest nodes
            origin_node = self.find_nearest_node(start_lat, start_lon)
            dest_node = self.find_nearest_node(end_lat, end_lon)

            origin_data = self.graph.nodes[origin_node]
            dest_data = self.graph.nodes[dest_node]

            origin = RoutePoint(
                lat=origin_data["y"],
                lon=origin_data["x"],
                node_id=origin_node,
            )
            destination = RoutePoint(
                lat=dest_data["y"],
                lon=dest_data["x"],
                node_id=dest_node,
            )

            logger.info(f"  Origin node: {origin_node} "
                         f"({origin.lat:.4f}, {origin.lon:.4f})")
            logger.info(f"  Dest node: {dest_node} "
                         f"({destination.lat:.4f}, {destination.lon:.4f})")

            if origin_node == dest_node:
                return RouteResult(
                    success=False,
                    error_message="Origin and destination resolve to the same node. "
                                  "They may be too close together.",
                    origin=origin,
                    destination=destination,
                )

            # Step 2: Run A* with custom cost
            logger.info("  Running A* pathfinding with custom cost function...")
            pathfind_start = time.time()

            try:
                route_nodes = nx.astar_path(
                    self.graph,
                    source=origin_node,
                    target=dest_node,
                    heuristic=self._astar_heuristic,
                    weight="custom_cost",
                )
            except nx.NetworkXNoPath:
                return RouteResult(
                    success=False,
                    error_message="No route found between the selected points. "
                                  "They may be on disconnected road segments.",
                    origin=origin,
                    destination=destination,
                )

            pathfind_time = time.time() - pathfind_start
            logger.info(f"  Path found in {pathfind_time:.3f}s "
                         f"({len(route_nodes)} nodes)")

            # Step 3: Extract route coordinates and compute totals
            route_coords = []
            total_distance = 0.0
            total_cost = 0.0

            for node_id in route_nodes:
                node_data = self.graph.nodes[node_id]
                route_coords.append((node_data["y"], node_data["x"]))

            # Compute total distance and cost along the path
            for i in range(len(route_nodes) - 1):
                u = route_nodes[i]
                v = route_nodes[i + 1]

                # Get the edge with minimum cost (for multigraph)
                edge_data = self._get_best_edge(u, v)
                total_distance += self._get_edge_length(edge_data)
                total_cost += edge_data.get("custom_cost", 0)

            total_distance_km = total_distance / 1000.0
            total_time_minutes = total_cost / 60.0

            # Step 4: Generate navigation instructions
            instructions = self._generate_instructions(route_nodes)

            # Step 5: Generate audio (optional)
            audio_file = None
            if generate_audio and instructions:
                audio_file = self._generate_audio(instructions)

            # Build result
            route_time = time.time() - route_start_time

            result = RouteResult(
                success=True,
                origin=origin,
                destination=destination,
                total_distance_km=total_distance_km,
                total_time_minutes=total_time_minutes,
                total_cost=total_cost,
                route_nodes=route_nodes,
                route_coordinates=route_coords,
                instructions=instructions,
                audio_file=audio_file,
                metadata={
                    "computation_time_ms": round(route_time * 1000, 1),
                    "pathfinding_time_ms": round(pathfind_time * 1000, 1),
                    "nodes_in_path": len(route_nodes),
                    "algorithm": "A* with custom cost function",
                    "cost_formula": "Cost = (D/S) × α × β + P",
                },
            )

            logger.info(f"  Route computed successfully:")
            logger.info(f"    Distance: {total_distance_km:.2f} km")
            logger.info(f"    Est. time: {total_time_minutes:.1f} min")
            logger.info(f"    Instructions: {len(instructions)} steps")
            logger.info(f"    Total computation: {route_time * 1000:.1f} ms")

            return result

        except Exception as e:
            logger.error(f"Route calculation error: {e}", exc_info=True)
            return RouteResult(
                success=False,
                error_message=f"Internal routing error: {str(e)}",
            )

    def _get_best_edge(self, u: int, v: int) -> Dict:
        """Get the edge data with the lowest custom_cost between u and v."""
        edges = self.graph[u][v]
        best_key = min(
            edges.keys(),
            key=lambda k: edges[k].get("custom_cost", float("inf")),
        )
        return edges[best_key]

    # ─────────────────────────────────────────────────────────
    # Navigation Instructions — Bearing & Turn Detection
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the compass bearing from point 1 to point 2.

        Returns bearing in degrees (0-360), where:
          0° = North, 90° = East, 180° = South, 270° = West
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lambda = math.radians(lon2 - lon1)

        x = math.sin(delta_lambda) * math.cos(phi2)
        y = (
            math.cos(phi1) * math.sin(phi2)
            - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
        )

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    @staticmethod
    def _calculate_angle_change(bearing1: float, bearing2: float) -> float:
        """
        Calculate the angle change between two consecutive bearings.

        Returns angle in degrees:
          Positive = right turn
          Negative = left turn
          Near zero = straight
        """
        diff = bearing2 - bearing1

        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return diff

    @staticmethod
    def _classify_turn(angle_change: float) -> Tuple[str, str]:
        """
        Classify a turn based on angle change.

        Returns:
            (direction_key, human_readable_direction)
        """
        abs_angle = abs(angle_change)

        if abs_angle > UTURN_THRESHOLD:
            return ("uturn", "Make a U-turn")
        elif abs_angle > SHARP_TURN_THRESHOLD:
            if angle_change > 0:
                return ("sharp_right", "Turn sharp right")
            else:
                return ("sharp_left", "Turn sharp left")
        elif abs_angle > TURN_THRESHOLD_DEGREES:
            if angle_change > 0:
                return ("right", "Turn right")
            else:
                return ("left", "Turn left")
        else:
            return ("straight", "Continue straight")

    def _get_street_name(self, u: int, v: int) -> str:
        """Extract the street name from an edge, if available."""
        edge_data = self._get_best_edge(u, v)
        name = edge_data.get("name", "")

        if isinstance(name, list):
            name = " / ".join(str(n) for n in name if n)
        elif not name:
            highway = edge_data.get("highway", "road")
            if isinstance(highway, list):
                highway = highway[0]
            name = f"Unnamed {str(highway).replace('_', ' ').title()}"

        return str(name)

    def _generate_instructions(
        self, route_nodes: List[int]
    ) -> List[NavigationInstruction]:
        """
        Generate turn-by-turn navigation instructions for a route.

        Algorithm:
          1. Calculate bearing between each consecutive pair of nodes.
          2. Calculate angle change between consecutive bearings.
          3. Classify each turn and generate human-readable instructions.
          4. Merge consecutive "straight" segments for cleaner output.
        """
        if len(route_nodes) < 2:
            return []

        instructions = []
        cumulative_distance = 0.0

        # Calculate all bearings
        bearings = []
        segment_distances = []

        for i in range(len(route_nodes) - 1):
            u = route_nodes[i]
            v = route_nodes[i + 1]

            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]

            bearing = self._calculate_bearing(
                u_data["y"], u_data["x"],
                v_data["y"], v_data["x"],
            )
            bearings.append(bearing)

            edge_data = self._get_best_edge(u, v)
            distance = self._get_edge_length(edge_data)
            segment_distances.append(distance)

        # First instruction: Start
        first_node_data = self.graph.nodes[route_nodes[0]]
        first_street = self._get_street_name(route_nodes[0], route_nodes[1])

        instructions.append(NavigationInstruction(
            step_number=1,
            instruction=f"Head {self._bearing_to_cardinal(bearings[0])} on {first_street}",
            direction="straight",
            distance_meters=segment_distances[0],
            cumulative_distance=segment_distances[0],
            bearing=bearings[0],
            angle_change=0.0,
            street_name=first_street,
            point=RoutePoint(
                lat=first_node_data["y"],
                lon=first_node_data["x"],
                node_id=route_nodes[0],
            ),
        ))
        cumulative_distance = segment_distances[0]

        # Process intermediate turns
        # We accumulate distance for "straight" segments
        accumulated_straight_distance = segment_distances[0]
        step_counter = 2

        for i in range(1, len(bearings)):
            angle_change = self._calculate_angle_change(bearings[i - 1], bearings[i])
            distance = segment_distances[i]
            cumulative_distance += distance

            direction_key, direction_text = self._classify_turn(angle_change)

            node_id = route_nodes[i]
            node_data = self.graph.nodes[node_id]

            if direction_key == "straight":
                # Accumulate distance for straight segments
                accumulated_straight_distance += distance
                continue

            # This is a turn — emit the accumulated straight + the turn
            street_name = self._get_street_name(
                route_nodes[i], route_nodes[i + 1]
            ) if i < len(route_nodes) - 1 else "destination"

            # Build instruction text
            dist_text = self._format_distance(accumulated_straight_distance)
            instruction_text = (
                f"In {dist_text}, {direction_text.lower()} onto {street_name}"
            )

            instructions.append(NavigationInstruction(
                step_number=step_counter,
                instruction=instruction_text,
                direction=direction_key,
                distance_meters=accumulated_straight_distance,
                cumulative_distance=cumulative_distance,
                bearing=bearings[i],
                angle_change=angle_change,
                street_name=street_name,
                point=RoutePoint(
                    lat=node_data["y"],
                    lon=node_data["x"],
                    node_id=node_id,
                ),
            ))

            step_counter += 1
            accumulated_straight_distance = distance

        # Final instruction: Arrive at destination
        last_node_data = self.graph.nodes[route_nodes[-1]]
        instructions.append(NavigationInstruction(
            step_number=step_counter,
            instruction=f"You have arrived at your destination "
                        f"({self._format_distance(cumulative_distance)} total)",
            direction="arrive",
            distance_meters=accumulated_straight_distance,
            cumulative_distance=cumulative_distance,
            bearing=bearings[-1] if bearings else 0,
            angle_change=0.0,
            street_name="Destination",
            point=RoutePoint(
                lat=last_node_data["y"],
                lon=last_node_data["x"],
                node_id=route_nodes[-1],
            ),
        ))

        return instructions

    @staticmethod
    def _bearing_to_cardinal(bearing: float) -> str:
        """Convert a bearing to a cardinal direction string."""
        directions = [
            "north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest",
        ]
        index = round(bearing / 45) % 8
        return directions[index]

    @staticmethod
    def _format_distance(meters: float) -> str:
        """Format distance in human-readable form."""
        if meters >= 1000:
            return f"{meters / 1000:.1f} km"
        return f"{int(meters)} meters"

    # ─────────────────────────────────────────────────────────
    # Audio Generation (gTTS)
    # ─────────────────────────────────────────────────────────

    def _generate_audio(
        self, instructions: List[NavigationInstruction]
    ) -> Optional[str]:
        """
        Generate an MP3 audio file with spoken navigation instructions.

        Returns the relative path to the audio file, or None on failure.
        """
        if not instructions:
            return None

        try:
            # Build the full text for TTS
            text_parts = []
            text_parts.append("AYrouT Navigation. Starting route guidance.")

            for inst in instructions:
                text_parts.append(f"Step {inst.step_number}. {inst.instruction}.")

            text_parts.append("Route guidance complete. Thank you for using AYrouT.")

            full_text = " ".join(text_parts)

            # Generate a unique filename based on content hash
            text_hash = hashlib.md5(full_text.encode()).hexdigest()[:12]
            filename = f"nav_{text_hash}.mp3"
            filepath = os.path.join(AUDIO_DIR, filename)

            # Check if already generated (cache)
            if os.path.exists(filepath):
                logger.info(f"  Audio cached: {filename}")
                return f"/static/audio/{filename}"

            # Generate audio
            logger.info("  Generating TTS audio...")
            tts = gTTS(text=full_text, lang="en", slow=False)
            tts.save(filepath)

            file_size = os.path.getsize(filepath) / 1024
            logger.info(f"  Audio saved: {filename} ({file_size:.1f} KB)")

            return f"/static/audio/{filename}"

        except Exception as e:
            logger.warning(f"  Audio generation failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # Geocoding
    # ─────────────────────────────────────────────────────────

    def geocode(self, query: str) -> Optional[RoutePoint]:
        """
        Geocode a place name to coordinates.
        Biased toward Agadir, Morocco.
        """
        if not self._geocoder:
            return None

        try:
            # Add Agadir bias to the query
            search_query = query
            if "agadir" not in query.lower() and "morocco" not in query.lower():
                search_query = f"{query}, Agadir, Morocco"

            logger.info(f"Geocoding: '{search_query}'")

            location = self._geocoder.geocode(
                search_query,
                exactly_one=True,
                language="en",
                viewbox=[(30.50, -9.70), (30.35, -9.45)],
                bounded=False,
            )

            if location:
                point = RoutePoint(
                    lat=location.latitude,
                    lon=location.longitude,
                    name=location.address,
                )
                logger.info(f"  Found: {point.lat:.4f}, {point.lon:.4f} - {point.name}")
                return point

            logger.warning(f"  Geocoding returned no results for: {query}")
            return None

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"  Geocoding service error: {e}")
            return None
        except Exception as e:
            logger.error(f"  Geocoding error: {e}")
            return None

    def reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
        """Reverse geocode coordinates to a place name."""
        if not self._geocoder:
            return None

        try:
            location = self._geocoder.reverse(
                f"{lat}, {lon}",
                exactly_one=True,
                language="en",
            )
            return location.address if location else None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────
    # Utility / Debug Methods
    # ─────────────────────────────────────────────────────────

    def get_edge_cost_breakdown(self, u: int, v: int) -> Dict:
        """
        Get a detailed breakdown of the cost calculation for a specific edge.
        Useful for debugging and understanding routing decisions.
        """
        edge_data = self._get_best_edge(u, v)
        distance = self._get_edge_length(edge_data)
        speed_ms = self._get_edge_speed(edge_data)
        alpha = self._get_road_quality_factor(edge_data)
        beta = self._get_traffic_density_factor(edge_data)
        node_penalty = self._node_penalties.get(v, 0.0)
        base_time = distance / speed_ms
        total_cost = base_time * alpha * beta + node_penalty

        return {
            "edge": f"{u} → {v}",
            "distance_m": round(distance, 1),
            "speed_kmh": round(speed_ms * 3.6, 1),
            "speed_ms": round(speed_ms, 2),
            "base_time_s": round(base_time, 2),
            "alpha_road_quality": alpha,
            "beta_traffic_density": beta,
            "node_penalty_s": node_penalty,
            "total_cost_s": round(total_cost, 2),
            "highway": edge_data.get("highway", "unknown"),
            "surface": edge_data.get("surface", "unknown"),
            "street_name": edge_data.get("name", "unnamed"),
        }

    def get_graph_stats(self) -> Dict:
        """Return statistics about the loaded graph."""
        if not self.graph:
            return {"error": "Graph not loaded"}

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "center": self._graph_center,
            "penalized_nodes": len(self._node_penalties),
            "precomputed_costs": len(self._edge_costs),
            "is_initialized": self._is_initialized,
        }


# ─────────────────────────────────────────────────────────────
# Module-level engine instance (Singleton Pattern)
# ─────────────────────────────────────────────────────────────

_engine_instance: Optional[AYrouTEngine] = None


def get_engine() -> AYrouTEngine:
    """
    Get or create the singleton engine instance.
    Thread-safe for Flask usage.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AYrouTEngine()
        _engine_instance.initialize()
    return _engine_instance


# ─────────────────────────────────────────────────────────────
# CLI Testing
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AYrouT Engine — Direct Test")
    print("=" * 60)

    engine = AYrouTEngine()
    engine.initialize()

    stats = engine.get_graph_stats()
    print(f"\nGraph Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test route: Agadir Beach → Souk El Had
    print("\n--- Test Route ---")
    result = engine.calculate_route(
        start_lat=30.4278,
        start_lon=-9.5981,
        end_lat=30.4220,
        end_lon=-9.5700,
    )

    if result.success:
        print(f"  Distance: {result.total_distance_km:.2f} km")
        print(f"  Time: {result.total_time_minutes:.1f} min")
        print(f"  Steps: {len(result.instructions)}")
        for inst in result.instructions:
            print(f"    [{inst.step_number}] {inst.instruction}")
        if result.audio_file:
            print(f"  Audio: {result.audio_file}")
    else:
        print(f"  FAILED: {result.error_message}")