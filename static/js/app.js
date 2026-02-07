/*
 * AYrouT External JavaScript Module
 * Core logic is inlined in index.html for simplicity.
 * This file is reserved for future modular expansion:
 *   - Client auto-detect module
 *   - Route history / favorites
 *   - Fleet management overlay
 *   - Real-time traffic data integration
 */

// Future: Client Auto-Detect Module
class AYrouTClientDetector {
    constructor() {
        this.clientId = null;
        this.lastKnownPosition = null;
        this.watchId = null;
    }

    startTracking(onUpdate) {
        if (!navigator.geolocation) {
            console.warn('Geolocation not supported');
            return false;
        }

        this.watchId = navigator.geolocation.watchPosition(
            (pos) => {
                this.lastKnownPosition = {
                    lat: pos.coords.latitude,
                    lon: pos.coords.longitude,
                    accuracy: pos.coords.accuracy,
                    timestamp: pos.timestamp,
                };
                if (onUpdate) onUpdate(this.lastKnownPosition);
            },
            (err) => console.error('Tracking error:', err),
            {
                enableHighAccuracy: true,
                maximumAge: 5000,
                timeout: 10000,
            }
        );

        return true;
    }

    stopTracking() {
        if (this.watchId !== null) {
            navigator.geolocation.clearWatch(this.watchId);
            this.watchId = null;
        }
    }
}

// Export for future use
window.AYrouTClientDetector = AYrouTClientDetector;