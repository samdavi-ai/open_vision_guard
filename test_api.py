import requests
import time

API_URL = "http://localhost:8080"

def test_stream():
    print("Testing /stream/start endpoint...")
    payload = {
        "source": "d:/firsttask_pon/OpenVisionGuard/test4video.mp4",
        "camera_id": "TEST_CAM_01"
    }
    response = requests.post(f"{API_URL}/stream/start", json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    
    if response.status_code == 200:
        print("Stream started successfully. Waiting 5 seconds to let it process some frames...")
        time.sleep(5)
        
        print("\nChecking /stream/list...")
        list_resp = requests.get(f"{API_URL}/stream/list")
        if list_resp.status_code == 200:
            print("Active Streams:", list_resp.json())
            
        print("\nChecking /alerts...")
        alerts_resp = requests.get(f"{API_URL}/alerts")
        if alerts_resp.status_code == 200:
            alerts = alerts_resp.json()
            print(f"Generated {len(alerts)} alerts so far.")
            for i, alert in enumerate(alerts[:3]):
                print(f"  Alert {i+1}: {alert['type']} - {alert['message']}")

        print("\nChecking /identities...")
        ident_resp = requests.get(f"{API_URL}/identities")
        if ident_resp.status_code == 200:
            identities = ident_resp.json()
            print(f"Tracked {len(identities)} identities so far.")
            
        print("\nStopping stream...")
        stop_resp = requests.post(f"{API_URL}/stream/stop/TEST_CAM_01")
        print("Stop Response:", stop_resp.json())

if __name__ == "__main__":
    test_stream()
