import { useState, useEffect } from 'react';
import CameraGrid from './components/CameraGrid.jsx';
import AlertSidebar from './components/AlertSidebar.jsx';
import { Shield, Settings, activity } from 'lucide-react';

function App() {
  const [cameras, setCameras] = useState([]);
  const [alerts, setAlerts] = useState([]);
  
  // Backend config
  const API_URL = "http://localhost:8000/api";

  const fetchCameras = async () => {
    try {
      const res = await fetch(`${API_URL}/cameras`);
      const data = await res.json();
      setCameras(data);
    } catch (e) {
      console.error("Failed to fetch cameras", e);
    }
  };

  useEffect(() => {
    fetchCameras();
    // In a real scenario we'd poll or use a global SSE/WS for config changes
  }, []);

  const addTestCamera = async () => {
    const newCam = {
      id: `cam_${Date.now()}`,
      url: "d:\\firsttask_pon\\OpenVisionGuard\\test_video.mp4", // Test video instead of webcam
      name: `Webcam ${cameras.length + 1}`,
      modules: ["object_detector", "face_recognition", "pose", "motion"]
    };
    await fetch(`${API_URL}/cameras`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newCam)
    });
    fetchCameras();
  };

  return (
    <div className="flex h-screen w-full bg-background text-white overflow-hidden font-sans">
      
      {/* Sidebar / Nav */}
      <div className="w-16 flex flex-col items-center py-6 border-r border-border bg-panel z-10">
        <div className="text-primary mb-8">
          <Shield size={32} />
        </div>
        <div className="flex flex-col gap-6 mt-4">
          <button className="p-3 bg-primary/20 text-primary rounded-xl hover:bg-primary/30 transition shadow-[0_0_15px_rgba(59,130,246,0.5)]">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><line x1="3" x2="21" y1="9" y2="9"/><line x1="9" x2="9" y1="21" y2="9"/></svg>
          </button>
          <button className="p-3 text-gray-400 hover:text-white transition">
            <Settings size={24} />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full bg-[#0a0a0a] relative overflow-hidden">
        {/* Header */}
        <div className="h-16 flex items-center justify-between px-6 border-b border-border/50 bg-black/40 backdrop-blur-md absolute top-0 w-full z-10">
          <div>
            <h1 className="text-xl font-bold tracking-wider">OPEN<span className="text-primary">VISION</span>GUARD</h1>
            <p className="text-xs text-gray-400 font-mono tracking-widest">AI SURVEILLANCE SYSTEM</p>
          </div>
          <div className="flex items-center gap-4">
            <button 
              onClick={addTestCamera}
              className="px-4 py-2 bg-primary/10 border border-primary/30 text-primary rounded text-sm font-medium hover:bg-primary/20 transition flex items-center gap-2">
              <span>+ Add Camera Stream</span>
            </button>
            <div className="flex items-center gap-2">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
              </span>
              <span className="text-xs text-red-500 font-bold uppercase tracking-widest">Live</span>
            </div>
          </div>
        </div>

        {/* Video Grid Area */}
        <div className="flex-1 p-6 pt-24 overflow-y-auto">
          {cameras.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500">
              <Shield size={64} className="opacity-20 mb-4" />
              <p>No active camera streams.</p>
              <p className="text-sm">Click 'Add Camera Stream' to begin monitoring.</p>
            </div>
          ) : (
            <CameraGrid cameras={cameras} onAlert={(alert) => setAlerts(prev => [alert, ...prev].slice(0, 50))} />
          )}
        </div>
      </div>

      {/* Alerts Sidebar */}
      <AlertSidebar alerts={alerts} />
      
    </div>
  )
}

export default App
