import React, { useEffect, useState, useRef } from 'react';
import { Camera, AlertTriangle, Crosshair, Activity, User, Eye } from 'lucide-react';

const StreamPlayer = ({ camera, onAlert }) => {
  const [frameData, setFrameData] = useState(null);
  const [status, setStatus] = useState("Connecting...");
  const [recentAlert, setRecentAlert] = useState(null);
  const ws = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    ws.current = new WebSocket(`ws://localhost:8000/ws/${camera.id}`);

    ws.current.onopen = () => setStatus("Live");
    ws.current.onclose = () => setStatus("Offline");
    ws.current.onerror = () => setStatus("Error");

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.frame) {
        setFrameData(`data:image/jpeg;base64,${data.frame}`);
      }
      
      if (data.alerts && data.alerts.length > 0) {
        const primaryAlert = data.alerts[0];
        setRecentAlert(primaryAlert);
        onAlert({ time: new Date().toLocaleTimeString(), text: primaryAlert, camName: camera.name });
        
        // Clear alert overlay after 2s
        setTimeout(() => setRecentAlert(null), 2000);
      }
    };

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [camera.id]);

  return (
    <div className="relative border border-border bg-panel rounded-xl overflow-hidden group shadow-lg flex flex-col h-full transform transition-all duration-300 hover:border-primary/50">
      
      {/* Stream Viewer */}
      <div className="flex-1 bg-black relative flex items-center justify-center overflow-hidden">
        {frameData ? (
          <img src={frameData} alt={`Stream ${camera.name}`} className="w-full h-full object-contain" />
        ) : (
          <div className="text-gray-600 flex flex-col items-center">
             <Camera size={48} className="mb-2 opacity-50 animate-pulse" />
             <p className="font-mono text-xs uppercase tracking-widest">{status}</p>
          </div>
        )}
        
        {/* Overlays */}
        <div className="absolute top-4 left-4 flex gap-2">
            <div className={`px-2 py-1 rounded text-xs font-bold uppercase tracking-wider backdrop-blur-md 
                ${status === 'Live' ? 'bg-green-500/20 text-green-400 border border-green-500/50' : 'bg-red-500/20 text-red-500 border border-red-500/50'}`}>
                {status}
            </div>
            <div className="px-2 py-1 bg-black/50 backdrop-blur-md text-white/80 border border-white/10 rounded text-xs font-mono">
                {camera.name}
            </div>
        </div>

        {/* Dynamic Alert Overlay (Tesla Vision style) */}
        {recentAlert && (
          <div className="absolute inset-0 border-4 border-red-500 animate-pulse pointer-events-none z-20 shadow-[inset_0_0_50px_rgba(239,68,68,0.5)]">
             <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-red-600 text-white px-4 py-2 rounded-full font-bold flex items-center gap-2 shadow-2xl">
                <AlertTriangle size={18} />
                <span className="uppercase tracking-widest text-sm">{recentAlert}</span>
             </div>
          </div>
        )}
        
        {/* Crosshair decoration */}
        <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-30 transition-opacity flex items-center justify-center">
           <Crosshair size={120} className="text-primary animate-spin-slow" strokeWidth={0.5} />
        </div>
      </div>

      {/* Module Footer Info */}
      <div className="h-10 border-t border-border bg-black/40 px-4 flex items-center gap-4 py-1 text-xs text-gray-400 font-mono">
         <span className="flex items-center gap-1.5" title="Object Detection">
             <Eye size={12} className={camera.modules.includes('object_detector') ? 'text-primary' : 'text-gray-600'}/> OBJ
         </span>
         <span className="flex items-center gap-1.5" title="Face Recognition">
             <User size={12} className={camera.modules.includes('face_recognition') ? 'text-blue-400' : 'text-gray-600'}/> FACE
         </span>
         <span className="flex items-center gap-1.5" title="Pose/Activity Detection">
             <Activity size={12} className={camera.modules.includes('pose') ? 'text-purple-400' : 'text-gray-600'}/> POSE
         </span>
         <span className="flex items-center gap-1.5" title="Motion/Anomaly">
             <AlertTriangle size={12} className={camera.modules.includes('motion') ? 'text-yellow-400' : 'text-gray-600'}/> MOT
         </span>
      </div>
    </div>
  );
};

export default StreamPlayer;
