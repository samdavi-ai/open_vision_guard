import datetime
import urllib.request
import json

class GeoLocationManager:
    """
    Manages geolocation data for the system.
    Modes supported:
      - 'auto': Fetches IP-based location automatically via ip-api.com.
      - 'static': Uses hardcoded latitude and longitude for testing.
      - 'gps': (Placeholder) Reads from a connected hardware GPS sensor.
    """
    def __init__(self, mode="auto", default_lat=13.0827, default_lon=80.2707):
        self.mode = mode
        self.lat = default_lat
        self.lon = default_lon
        
        if self.mode == "auto":
            try:
                print("[GeoLocation] Fetching actual location...")
                req = urllib.request.Request(
                    "http://ip-api.com/json", 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    if data.get('status') == 'success':
                        self.lat = data['lat']
                        self.lon = data['lon']
                        print(f"[GeoLocation] Acquired actual location: {self.lat}, {self.lon} ({data.get('city')}, {data.get('country')})")
            except Exception as e:
                print(f"[GeoLocation] Failed to fetch IP location: {e}. Falling back to default.")

    def get_current_location(self):
        """Returns the current latitude and longitude."""
        return {
            "latitude": self.lat,
            "longitude": self.lon
        }

# Global singleton instance
geolocation_engine = GeoLocationManager()