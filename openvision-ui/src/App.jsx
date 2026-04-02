import React, { useState, useEffect, useRef } from 'react';
import { Activity, ShieldAlert, Cpu, Upload, Video, Camera } from 'lucide-react';
import VideoStream from './components/VideoStream';
import AlertsFeed from './components/AlertsFeed';
import PersonView from './components/PersonView';
import './index.css';

const API_BASE = `${window.location.protocol}//${window.location.hostname}:8080`;
const WS_BASE  = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8080`;

/* ── Live Clock Hook ── */
function useLiveClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

export default function App() {
  const [activeStreams, setActiveStreams] = useState([]);
  const [sourceData,  setSourceData]   = useState('');
  const [sourceMode,  setSourceMode]   = useState('url');
  const [alerts,      setAlerts]       = useState([]);
  const [activePerson, setActivePerson] = useState(null); // {globalId, cameraId}

  const now          = useLiveClock();
  const alertsWsRef  = useRef(null);
  const fileInputRef = useRef(null);

  /* ── Sync active streams with backend ── */
  const fetchStreams = async () => {
    try {
      const r = await fetch(`${API_BASE}/stream/list`);
      if (r.ok) {
        const list = await r.json();
        setActiveStreams(list.filter(s => s.status === 'running' || s.status === 'starting'));
      }
    } catch (e) {}
  };

  useEffect(() => {
    fetchStreams();
    const id = setInterval(fetchStreams, 3000);
    return () => clearInterval(id);
  }, []);

  /* ── Alerts WebSocket ── */
  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/alerts`);
    alertsWsRef.current = ws;
    ws.onmessage = (e) => {
      try { setAlerts(p => [JSON.parse(e.data), ...p].slice(0, 60)); } catch (_) {}
    };
    return () => ws.close();
  }, []);

  const isStreaming = activeStreams.length > 0;

  /* ── Stream controls ── */
  const startStream = async (src) => {
    const source = src ?? sourceData;
    if (!source) return;
    try {
      const r = await fetch(`${API_BASE}/stream/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source }), // Let backend assign camera_id
      });
      if (r.ok) {
        const d = await r.json();
        setActiveStreams(prev => [...prev.filter(s => s.camera_id !== d.camera_id), { camera_id: d.camera_id, source, status: 'starting' }]);
        setSourceData(''); // clear input
      }
    } catch (e) { console.error(e); }
  };

  const stopStream = async (cameraId) => {
    try {
      await fetch(`${API_BASE}/stream/stop/${cameraId}`, { method: 'POST' });
      setActiveStreams(prev => prev.filter(s => s.camera_id !== cameraId));
      if (activePerson?.cameraId === cameraId) setActivePerson(null);
    } catch (e) {}
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData(); fd.append('file', file);
    try {
      const r = await fetch(`${API_BASE}/stream/upload`, { method: 'POST', body: fd });
      const d = await r.json();
      if (d.path) { 
        setSourceData(d.path); 
        setSourceMode('url'); 
        alert(`Successfully uploaded ${file.name}. Click 'Start' to begin monitoring.`);
      } else {
        alert("Upload failed: " + (d.error || "Unknown error"));
      }
    } catch (err) { 
      alert("Upload failed. Check console for details.");
    }
  };

  /* ── Person click handler ── */
  const handlePersonClick = (globalId, cameraId) => {
    setActivePerson({ globalId, cameraId });
  };

  /* ────────────────────────────────────────────────
     If a person is selected → show full-screen view
  ──────────────────────────────────────────────── */
  if (activePerson) {
    return (
      <PersonView
        globalId={activePerson.globalId}
        cameraId={activePerson.cameraId}
        apiBase={API_BASE}
        onBack={() => setActivePerson(null)}
      />
    );
  }

  /* ────────────────────────────────────────────────
     Main Dashboard
  ──────────────────────────────────────────────── */
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Header */}
      <header style={S.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={S.logo}><Cpu size={18} color="white" /></div>
          <span style={{ fontSize: '1.05rem', fontWeight: 700, letterSpacing: '-0.01em' }}>OpenVisionGuard</span>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          {/* Live Clock */}
          <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)', fontFamily: 'monospace', letterSpacing: '0.04em' }}>
            🕐 {now.toLocaleTimeString()}
          </span>
          {isStreaming && (
            <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
              <span style={{ color: 'var(--low)', fontWeight: 600 }}>{activeStreams.length} Cameras</span>
              &nbsp;·&nbsp;{alerts.length} alerts
            </span>
          )}
          <div style={S.statusBadge}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: isStreaming ? 'var(--low)' : '#475569', alignSelf: 'center', animation: isStreaming ? 'pulse 2s infinite' : 'none' }} />
            {isStreaming ? 'Monitoring Active' : 'System Ready'}
          </div>
        </div>
      </header>

      {/* Grid: video | alerts */}
      <div style={S.mainGrid}>

        {/* LEFT: Video feeds */}
        <div style={{ ...S.panel, display: 'flex', flexDirection: 'column' }}>
          <div style={S.panelHeader}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: isStreaming ? 'var(--low)' : '#475569', animation: isStreaming ? 'pulse 2s infinite' : 'none' }} />
              Command Center — {isStreaming ? `${activeStreams.length} Live Feed(s)` : 'No Feeds Active'}
            </span>
            {isStreaming && <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Click a person to open full profile</span>}
          </div>

          {/* Video canvas (Grid) */}
          <div style={{ flex: 1, background: '#080c14', position: 'relative', minHeight: 0, overflow: 'hidden' }}>
            {isStreaming ? (
              <VideoGrid streams={activeStreams} stopStream={stopStream} onPersonClick={handlePersonClick} />
            ) : (
              <div style={S.emptyState}>
                <div style={S.emptyIcon}><Activity size={26} style={{ opacity: 0.4 }} /></div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>No Active Streams</div>
                <div style={{ fontSize: '0.78rem' }}>Choose a source below to start AI monitoring</div>
              </div>
            )}
          </div>

          {/* Source controls */}
          {activeStreams.length < 4 && (
            <div style={{ padding: '10px 12px', borderTop: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)', flexShrink: 0 }}>
              {/* Tabs */}
              <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
                {[
                  { mode: 'url',    icon: <Video size={12} />,  label: 'Path / URL' },
                  { mode: 'upload', icon: <Upload size={12} />, label: 'Upload Video' },
                  { mode: 'webcam', icon: <Camera size={12} />, label: 'Webcam' },
                ].map(t => (
                  <button key={t.mode} onClick={() => setSourceMode(t.mode)} style={{
                    ...S.tabBtn,
                    background: sourceMode === t.mode ? 'var(--accent)' : 'rgba(255,255,255,0.05)',
                    color: sourceMode === t.mode ? '#fff' : 'var(--text-dim)',
                    border: `1px solid ${sourceMode === t.mode ? 'var(--accent)' : 'var(--border-color)'}`,
                  }}>
                    {t.icon} {t.label}
                  </button>
                ))}
                {isStreaming && <div style={{ marginLeft: 'auto', fontSize: '0.7rem', color: 'var(--text-dim)', alignSelf: 'center' }}>Max 4 concurrent cameras</div>}
              </div>

              {sourceMode === 'url' && (
                <div style={{ display: 'flex', gap: 6 }}>
                  <input value={sourceData} onChange={e => setSourceData(e.target.value)}
                    placeholder="Video path or RTSP URL (e.g. data/uploads/video.mp4)"
                    style={S.input} onKeyDown={e => e.key === 'Enter' && startStream()} />
                  <button onClick={() => startStream()} disabled={!sourceData} style={S.primaryBtn}>Start Camera</button>
                </div>
              )}
              {sourceMode === 'upload' && (
                <div onClick={() => fileInputRef.current?.click()} style={S.dropZone}>
                  <Upload size={18} style={{ opacity: 0.4 }} />
                  <span style={{ fontSize: '0.82rem' }}>Click to select video file</span>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>MP4, AVI, MOV, MKV</span>
                  <input ref={fileInputRef} type="file" accept="video/*" onChange={handleUpload} style={{ display: 'none' }} />
                </div>
              )}
              {sourceMode === 'webcam' && (
                <button onClick={() => startStream('0')} style={{ ...S.primaryBtn, width: '100%', justifyContent: 'center', padding: '0.6rem' }}>
                  <Camera size={15} /> Connect Webcam
                </button>
              )}
            </div>
          )}
        </div>

        {/* RIGHT: Alerts */}
        <div style={{ ...S.panel, display: 'flex', flexDirection: 'column' }}>
          <div style={S.panelHeader}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ShieldAlert size={14} /> System Alerts
            </span>
            {alerts.length > 0 && (
              <span style={{ fontSize: '0.7rem', padding: '0.15rem 0.45rem', background: 'rgba(239,68,68,0.15)', color: 'var(--critical)', borderRadius: 999, fontWeight: 600 }}>
                {alerts.length}
              </span>
            )}
          </div>
          <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <AlertsFeed alerts={alerts} setAlerts={setAlerts} apiBase={API_BASE} />
          </div>
          {isStreaming && (
            <div style={{ padding: '8px 12px', borderTop: '1px solid var(--border-color)', fontSize: '0.7rem', color: 'var(--text-dim)', textAlign: 'center' }}>
              👆 Click any person in the video to open their Intelligence Profile
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── Multi-Camera Helper Components ─── */
function VideoGrid({ streams, stopStream, onPersonClick }) {
  const count = streams.length;
  const cols = count === 1 ? '1fr' : '1fr 1fr';
  const rows = count <= 2 ? '1fr' : '1fr 1fr';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: cols, gridTemplateRows: rows, gap: 2, width: '100%', height: '100%' }}>
      {streams.map(stream => (
        <VideoSlot key={stream.camera_id} stream={stream} stopStream={stopStream} onPersonClick={onPersonClick} />
      ))}
    </div>
  );
}

function VideoSlot({ stream, stopStream, onPersonClick }) {
  const [fps, setFps] = useState(0);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden', background: '#000' }}>
      {/* Overlay header */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10,
        padding: '6px 10px', background: 'linear-gradient(to bottom, rgba(0,0,0,0.85), transparent)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center', pointerEvents: 'none'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.68rem', fontWeight: 600, color: '#fff', fontFamily: 'monospace' }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--low)', animation: 'pulse 2s infinite' }} />
          {stream.camera_id}
          <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}>|</span>
          <span style={{ color: 'var(--accent)' }}>{fps} FPS</span>
        </div>
        <button
          onClick={() => stopStream(stream.camera_id)}
          style={{ ...S.stopBtn, padding: '2px 8px', fontSize: '0.62rem', pointerEvents: 'auto' }}
        >
          Close X
        </button>
      </div>

      <VideoStream
        wsUrl={`${WS_BASE}/ws/stream/${stream.camera_id}`}
        setFps={setFps}
        onPersonClick={(gid) => onPersonClick(gid, stream.camera_id)}
      />
    </div>
  );
}

/* ─── Styles ─── */
const S = {
  header: {
    padding: '0.6rem 1rem', borderBottom: '1px solid var(--border-color)',
    background: 'rgba(8,15,30,0.95)',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    flexShrink: 0,
  },
  logo: {
    width: 30, height: 30,
    background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
    borderRadius: 7, display: 'flex', alignItems: 'center', justifyContent: 'center',
    boxShadow: '0 4px 12px rgba(59,130,246,0.3)',
  },
  statusBadge: {
    display: 'flex', gap: 6, fontSize: '0.78rem', color: 'var(--text-dim)',
    background: 'rgba(255,255,255,0.04)', padding: '0.35rem 0.8rem',
    borderRadius: 999, border: '1px solid var(--border-color)',
  },
  mainGrid: {
    flex: 1, display: 'grid', gridTemplateColumns: '1fr 340px',
    gap: 10, padding: 10, minHeight: 0, overflow: 'hidden',
  },
  panel: {
    background: 'rgba(10,20,40,0.8)', border: '1px solid var(--border-color)',
    borderRadius: 10, backdropFilter: 'blur(12px)', minHeight: 0, overflow: 'hidden',
  },
  panelHeader: {
    padding: '0.55rem 0.9rem', borderBottom: '1px solid var(--border-color)',
    fontWeight: 600, fontSize: '0.82rem',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    flexShrink: 0,
  },
  emptyState: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    height: '100%', color: 'var(--text-dim)', gap: 8,
  },
  emptyIcon: {
    width: 56, height: 56, borderRadius: '50%',
    background: 'rgba(255,255,255,0.04)', display: 'flex', alignItems: 'center', justifyContent: 'center',
    border: '1px solid var(--border-color)',
  },
  input: {
    flex: 1, background: 'rgba(255,255,255,0.05)',
    border: '1px solid var(--border-color)', color: 'var(--text-main)',
    padding: '0.45rem 0.75rem', borderRadius: 6, fontSize: '0.82rem',
  },
  primaryBtn: {
    background: 'var(--accent)', color: 'white', border: 'none',
    padding: '0.45rem 0.9rem', borderRadius: 6, cursor: 'pointer',
    fontWeight: 600, display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.82rem',
    whiteSpace: 'nowrap', fontFamily: 'inherit',
  },
  stopBtn: {
    background: 'rgba(239,68,68,0.15)', color: 'var(--critical)',
    border: '1px solid rgba(239,68,68,0.3)',
    padding: '0.35rem 0.8rem', borderRadius: 6, cursor: 'pointer',
    fontWeight: 500, fontSize: '0.8rem', fontFamily: 'inherit',
  },
  tabBtn: {
    padding: '0.3rem 0.6rem', borderRadius: 5, cursor: 'pointer',
    fontSize: '0.74rem', fontWeight: 500,
    display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'inherit',
  },
  dropZone: {
    border: '1px dashed var(--border-color)', borderRadius: 7, padding: '0.75rem',
    textAlign: 'center', cursor: 'pointer',
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
    color: 'var(--text-dim)',
  },
};
