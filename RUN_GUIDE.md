# How to Run Open Vision Guard (Upgraded)

Follow these steps to start the surveillance system and test the new baggage-swap detection features.

### 1. Start the Database (Docker)
The system uses a PostgreSQL database to store identities and alerts.

1. Open Docker Desktop on your Mac.
2. In your terminal, run:
   ```bash
   cd "/Volumes/Oswald Stack/alwar kuruchi/open_vision_guard"
   docker-compose up -d postgres
   ```

### 2. Start the Backend (AI Engine)
The backend handles the video processing, Re-ID, and alert logic.

1. In the same directory, activate the Python virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Start the FastAPI server:
   ```bash
   python main.py
   ```
   *The server will start at `http://127.0.0.1:8080`. You should see logs indicating that `osnet_ain_x1_0` has been loaded successfully.*

---

### 3. Start the Frontend (Dashboard)
The frontend provides the visual monitoring grid and the alerts feed.

1. Open a **new** terminal tab or window.
2. Navigate to the frontend directory:
   ```bash
   cd "/Volumes/Oswald Stack/alwar kuruchi/open_vision_guard/openvision-ui"
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   *Note: If you haven't installed dependencies recently, run `npm install` first.*
4. Open your browser to the URL shown in the terminal (usually `http://localhost:5173`).

---

### 4. Testing the Baggage-Swap Scenario
To verify the new features with your video clip:

1. In the Dashboard, look for the **Source Controls** at the bottom left.
2. Select the **Upload Video** tab.
3. Click to select your video file (the one where the man enters with a backpack and leaves without it).
4. Once uploaded, click **Start Camera**.
5. **Watch the Alerts Feed (Right Side):**
   *   When the man first enters, he will be assigned a `global_id` (e.g., `Person_OSN_001`).
   *   When he exits the frame without his backpack, a **high-priority** alert will appear: `🚨 ITEM LEFT BEHIND`.
   *   The card will show exactly what was left (e.g., `backpack`) and the time of entry/exit.
   *   When he re-enters and takes a laptop, the system will match his identity (even without the bag) and fire a `🚨 ITEM TAKEN` or `🔴 BAGGAGE SWAP` alert upon his next exit.

---

### 5. Troubleshooting
*   **Permissions**: If you get a "camera access" or "file access" error, ensure your terminal has "Full Disk Access" in macOS System Settings.
*   **Performance**: If the video is laggy, the system is likely utilizing the **MPS (Metal Performance Shaders)** on your M4 chip correctly, but you can lower the resolution in `config.py` if needed.
*   **Re-ID Matching**: If the person is not matched correctly after taking off the bag, try lowering `similarity_threshold` in `config.py` to `0.68`.
