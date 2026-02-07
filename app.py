#!/usr/bin/env python3
"""
AYrouT - Flask Web Server
==========================
Serves the routing interface and handles API requests.
"""

import os
import json
import time
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory

from backend_logic import AYrouTEngine, get_engine

# ─────────────────────────────────────────────────────────────
# Flask App Configuration
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "ayrout-secret-key-2024"
app.config["JSON_SORT_KEYS"] = False

logger = logging.getLogger("AYrouT.Server")

# Initialize engine at startup
engine: AYrouTEngine = None


def init_engine():
    """Initialize the routing engine."""
    global engine
    try:
        engine = get_engine()
        logger.info("Routing engine ready.")
    except Exception as e:
        logger.error(f"CRITICAL: Engine init failed: {e}")
        engine = None


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main application page."""
    center = engine.center if engine else (30.4278, -9.5981)
    return render_template(
        "index.html",
        map_center_lat=center[0],
        map_center_lon=center[1],
    )


@app.route("/calculate_route", methods=["POST"])
def calculate_route():
    """
    API endpoint: Calculate route between two points.

    Expects JSON:
    {
        "start_lat": float,
        "start_lon": float,
        "end_lat": float,
        "end_lon": float,
        "generate_audio": bool (optional, default true)
    }
    """
    if not engine or not engine.is_ready:
        return jsonify({
            "success": False,
            "error_message": "Routing engine is not ready. Please wait and retry.",
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error_message": "Invalid request. JSON body required.",
            }), 400

        # Extract coordinates
        start_lat = float(data.get("start_lat", 0))
        start_lon = float(data.get("start_lon", 0))
        end_lat = float(data.get("end_lat", 0))
        end_lon = float(data.get("end_lon", 0))
        generate_audio = bool(data.get("generate_audio", True))

        # Validate coordinates
        if not all([start_lat, start_lon, end_lat, end_lon]):
            return jsonify({
                "success": False,
                "error_message": "Missing coordinates. Provide start_lat, start_lon, end_lat, end_lon.",
            }), 400

        # Validate coordinate ranges (rough check for Agadir area)
        for lat in [start_lat, end_lat]:
            if not (29.0 <= lat <= 31.5):
                return jsonify({
                    "success": False,
                    "error_message": f"Latitude {lat} is outside the supported area (Agadir region).",
                }), 400

        for lon in [start_lon, end_lon]:
            if not (-10.5 <= lon <= -8.5):
                return jsonify({
                    "success": False,
                    "error_message": f"Longitude {lon} is outside the supported area (Agadir region).",
                }), 400

        # Calculate route
        result = engine.calculate_route(
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            generate_audio=generate_audio,
        )

        return jsonify(result.to_dict())

    except (ValueError, TypeError) as e:
        return jsonify({
            "success": False,
            "error_message": f"Invalid parameter format: {str(e)}",
        }), 400
    except Exception as e:
        logger.error(f"Route calculation error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error_message": f"Server error: {str(e)}",
        }), 500


@app.route("/geocode", methods=["POST"])
def geocode():
    """
    API endpoint: Geocode a place name to coordinates.

    Expects JSON:
    {
        "query": "place name string"
    }
    """
    if not engine or not engine.is_ready:
        return jsonify({"success": False, "error_message": "Engine not ready."}), 503

    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({
                "success": False,
                "error_message": "Empty search query.",
            }), 400

        point = engine.geocode(query)

        if point:
            return jsonify({
                "success": True,
                "lat": point.lat,
                "lon": point.lon,
                "name": point.name,
            })
        else:
            return jsonify({
                "success": False,
                "error_message": f"Location '{query}' not found.",
            })

    except Exception as e:
        return jsonify({
            "success": False,
            "error_message": str(e),
        }), 500


@app.route("/reverse_geocode", methods=["POST"])
def reverse_geocode():
    """
    API endpoint: Reverse geocode coordinates to a place name.

    Expects JSON:
    {
        "lat": float,
        "lon": float
    }
    """
    if not engine or not engine.is_ready:
        return jsonify({"success": False, "error_message": "Engine not ready."}), 503

    try:
        data = request.get_json()
        lat = float(data.get("lat", 0))
        lon = float(data.get("lon", 0))

        address = engine.reverse_geocode(lat, lon)

        return jsonify({
            "success": bool(address),
            "address": address or "Unknown location",
            "lat": lat,
            "lon": lon,
        })

    except Exception as e:
        return jsonify({"success": False, "error_message": str(e)}), 500


@app.route("/engine_status")
def engine_status():
    """API endpoint: Check engine health and statistics."""
    if engine and engine.is_ready:
        stats = engine.get_graph_stats()
        return jsonify({
            "status": "ready",
            "stats": stats,
        })
    else:
        return jsonify({
            "status": "not_ready",
            "message": "Engine is initializing or failed to start.",
        }), 503


@app.route("/static/audio/<path:filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    audio_dir = os.path.join(app.root_path, "static", "audio")
    return send_from_directory(audio_dir, filename)


# ─────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AYrouT — Custom Routing Engine for Agadir")
    print("=" * 60)
    print()

    init_engine()

    if engine and engine.is_ready:
        print()
        print("  Server starting at: http://localhost:5000")
        print("  Press Ctrl+C to stop.")
        print()
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
        )
    else:
        print("FATAL: Engine failed to initialize.")
        print("Run 'python download_graph.py' first to download the map data.")