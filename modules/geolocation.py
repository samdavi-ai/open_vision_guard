import datetime

class GeoLocationManager:
    """
    Manages geolocation data for the system.
    Modes supported:
      - 'static': Uses hardcoded latitude and longitude for testing.
      - 'api': (Placeholder) Fetches IP-based location via an external API.
      - 'gps': (Placeholder) Reads from a connected hardware GPS sensor.
    """
    def __init__(self, mode="static", default_lat=13.0827, default_lon=80.2707):
        self.mode = mode
        self.default_lat = default_lat
        self.default_lon = default_lon

    def get_current_location(self):
        """Returns the current latitude and longitude based on the selected mode."""
        if self.mode == "static":
            return {
                "latitude": self.default_lat,
                "longitude": self.default_lon
            }
        elif self.mode == "api":
            # Extend to call an external service e.g., ip-api.com
            return {
                "latitude": self.default_lat,
                "longitude": self.default_lon
            }
        elif self.mode == "gps":
            # Extend to read serial stream 
            return {
                "latitude": self.default_lat,
                "longitude": self.default_lon
            }
        return {"latitude": 0.0, "longitude": 0.0}

# Global singleton instance
geolocation_engine = GeoLocationManager()
