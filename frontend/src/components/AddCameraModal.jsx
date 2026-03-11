import React, { useState, useRef } from 'react';
import { Upload, X, Shield, PlusCircle } from 'lucide-react';

const AddCameraModal = ({ isOpen, onClose, onAdd }) => {
  const [mode, setMode] = useState('url'); // 'url', 'webcam', 'upload'
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  
  // Modules state
  const [modules, setModules] = useState({
    object_detector: true,
    face_recognition: true,
    pose: true,
    motion: true,
    weapon: false
  });

  const fileInputRef = useRef(null);

  if (!isOpen) return null;

  const toggleModule = (mod) => {
    setModules(prev => ({ ...prev, [mod]: !prev[mod] }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const activeModules = Object.keys(modules).filter(k => modules[k]);

    if (mode === 'upload' && file) {
      // Create FormData
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', name || file.name);
      formData.append('modules', activeModules.join(','));
      
      await onAdd(formData, true);
    } else {
      // JSON Payload
      const streamUrl = mode === 'webcam' ? '0' : url;
      const payload = {
        id: `cam_${Date.now()}`,
        url: streamUrl,
        name: name || (mode === 'webcam' ? 'Local Webcam' : 'IP Camera'),
        modules: activeModules
      };
      
      await onAdd(payload, false);
    }
    
    // Reset and close
    setName('');
    setUrl('');
    setFile(null);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 min-w-[320px]">
      <div className="bg-panel border border-border w-full max-w-md rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border bg-black/40">
          <h2 className="font-bold tracking-wider flex items-center gap-2 text-white">
            <PlusCircle className="text-primary" size={20} />
            ADD SOURCE
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition">
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <form onSubmit={handleSubmit} className="p-6 flex flex-col gap-6">
          
          {/* Mode Selector */}
          <div className="flex bg-black/50 p-1 rounded-lg border border-border">
            <button type="button" onClick={() => setMode('url')} className={`flex-1 py-1.5 text-xs font-bold rounded-md transition ${mode === 'url' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white'}`}>RTSP / URL</button>
            <button type="button" onClick={() => setMode('webcam')} className={`flex-1 py-1.5 text-xs font-bold rounded-md transition ${mode === 'webcam' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white'}`}>WEBCAM</button>
            <button type="button" onClick={() => setMode('upload')} className={`flex-1 py-1.5 text-xs font-bold rounded-md transition ${mode === 'upload' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white'}`}>UPLOAD FILE</button>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-xs text-gray-400 font-mono mb-1 block">DISPLAY NAME</label>
              <input 
                type="text" 
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder={mode === 'upload' ? "Video 1" : "Front Door"}
                className="w-full bg-black/40 border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary transition"
              />
            </div>

            {mode === 'url' && (
              <div>
                <label className="text-xs text-gray-400 font-mono mb-1 block">STREAM URL</label>
                <input 
                  type="text" 
                  value={url}
                  onChange={e => setUrl(e.target.value)}
                  placeholder="rtsp://192.168.1.100:554/stream"
                  className="w-full bg-black/40 border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary transition"
                  required
                />
              </div>
            )}
            
            {mode === 'upload' && (
              <div>
                <label className="text-xs text-gray-400 font-mono mb-1 block">VIDEO FILE (.mp4, .avi)</label>
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full h-24 border-2 border-dashed border-border hover:border-primary/50 bg-black/20 rounded-xl flex flex-col items-center justify-center cursor-pointer transition text-gray-400 hover:text-blue-400"
                >
                  <Upload size={24} className="mb-2" />
                  <span className="text-xs font-mono">{file ? file.name : "Click to select video"}</span>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="video/*"
                    onChange={e => setFile(e.target.files[0])}
                  />
                </div>
              </div>
            )}
          </div>

          {/* AI Modules Configuration */}
          <div>
             <label className="text-xs text-primary font-mono mb-3 block">ACTIVE INTELLIGENCE MODULES</label>
             <div className="grid grid-cols-2 gap-3">
                {Object.keys(modules).map(mod => (
                  <label key={mod} className="flex items-center gap-2 cursor-pointer group">
                     <div className={`w-5 h-5 rounded border flex items-center justify-center transition ${modules[mod] ? 'bg-primary border-primary text-white' : 'bg-black/40 border-border text-transparent group-hover:border-gray-500'}`}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" className="w-3 h-3"><path d="M20 6L9 17l-5-5"></path></svg>
                     </div>
                     <span className="text-xs text-gray-300 font-mono capitalize">{mod.replace('_', ' ')}</span>
                  </label>
                ))}
             </div>
          </div>

          {/* Submit */}
          <button 
            type="submit" 
            disabled={mode === 'upload' && !file}
            className="w-full py-3 bg-primary text-white rounded-lg font-bold tracking-widest hover:bg-blue-600 transition shadow-[0_0_20px_rgba(59,130,246,0.3)] disabled:opacity-50 disabled:shadow-none mt-2"
          >
            INITIALIZE STREAM
          </button>
        </form>
      </div>
    </div>
  );
};

export default AddCameraModal;
