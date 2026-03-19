import React, { useState, useEffect, useRef } from 'react';
import { Activity, ShieldAlert, Cpu, Upload, Video, Camera } from 'lucide-react';
import VideoStream from './components/VideoStream';
import AlertsFeed from './components/AlertsFeed';
import PersonView from './components/PersonView';
import './index.css';

const API_BASE = "http://localhost:8080";
const WS_BASE  = "ws://localhost:8080";

export default function App() {
  const [isStreaming, setIsStreaming]   = useState(false);
  const [sourceData,  setSourceData]   = useState('');
  const [sourceMode,  setSourceMode]   = useState('url');
  const [alerts,      setAlerts]       = useState([]);
  const [fps,         setFps]          = useState(0);
  const [activePerson, setActivePerson] = useState(null); // {globalId, cameraId}

  const alertsWsRef = useRef(null);
  const fileInputRef = useRef(null);

  /* ── Alerts WebSocket ── */
  useEffect(() => {
    if (!isStreaming) return;
    const ws = new WebSocket(`${WS_BASE}/ws/alerts`);
    alertsWsRef.current = ws;
    ws.onmessage = (e) => {
      try { setAlerts(p => [JSON.parse(e.data), ...p].slice(0, 60)); } catch (_) {}
    };
    return () => ws.close();
  }, [isStreaming]);

  /* ── Stream controls ── */
  const startStream = async (src) => {
    const source = src ?? sourceData;
    if (!source) return;
    try {
      const r = await fetch(`${API_BASE}/stream/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source, camera_id: 'CAM_01' }),
      });
      if (r.ok) setIsStreaming(true);
    } catch (e) { console.error(e); }
  };

  const stopStream = async () => {
    await fetch(`${API_BASE}/stream/stop/CAM_01`, { method: 'POST' });
    setIsStreaming(false); setAlerts([]); setActivePerson(null);
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData(); fd.append('file', file);
    try {
      const r = await fetch(`${API_BASE}/stream/upload`, { method: 'POST', body: fd });
      const d = await r.json();
      if (d.path) { setSourceData(d.path); setSourceMode('url'); }
    } catch (err) { console.error(err); }
  };

  /* ── Person click handler ── */
  const handlePersonClick = (globalId) => {
    setActivePerson({ globalId, cameraId: 'CAM_01' });
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
          {isStreaming && (
            <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
              <span style={{ color: 'var(--low)', fontWeight: 600 }}>{fps} FPS</span>
              &nbsp;·&nbsp;{alerts.length} alerts
            </span>
          )}
          <div style={S.statusBadge}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: isStreaming ? 'var(--low)' : '#475569', alignSelf: 'center', animation: isStreaming ? 'pulse 2s infinite' : 'none' }} />
            {isStreaming ? 'Processing Active' : 'System Ready'}
          </div>
        </div>
      </header>

      {/* Grid: video | alerts */}
      <div style={S.mainGrid}>

        {/* LEFT: Video */}
        <div style={{ ...S.panel, display: 'flex', flexDirection: 'column' }}>
          <div style={S.panelHeader}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: isStreaming ? 'var(--low)' : '#475569', animation: isStreaming ? 'pulse 2s infinite' : 'none' }} />
              CAM_01 — Live Feed
            </span>
            {isStreaming && <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Click a person to open full profile</span>}
          </div>

          {/* Video canvas */}
          <div style={{ flex: 1, background: '#000', position: 'relative', minHeight: 0, overflow: 'hidden' }}>
            {isStreaming ? (
              <VideoStream wsUrl={`${WS_BASE}/ws/stream/CAM_01`} setFps={setFps} onPersonClick={handlePersonClick} />
            ) : (
              <div style={S.emptyState}>
                <div style={S.emptyIcon}><Activity size={26} style={{ opacity: 0.4 }} /></div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>No Active Stream</div>
                <div style={{ fontSize: '0.78rem' }}>Choose a source below to start AI monitoring</div>
              </div>
            )}
          </div>

          {/* Source controls */}
          <div style={{ padding: '10px 12px', borderTop: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)', flexShrink: 0 }}>
            {!isStreaming ? (
              <>
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
                </div>

                {sourceMode === 'url' && (
                  <div style={{ display: 'flex', gap: 6 }}>
                    <input value={sourceData} onChange={e => setSourceData(e.target.value)}
                      placeholder="Video path or RTSP URL (e.g. data/uploads/video.mp4)"
                      style={S.input} onKeyDown={e => e.key === 'Enter' && startStream()} />
                    <button onClick={() => startStream()} disabled={!sourceData} style={S.primaryBtn}>Start</button>
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
              </>
            ) : (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <div style={{ flex: 1, fontSize: '0.78rem', color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  📹 {sourceData || 'Webcam'}
                </div>
                <button onClick={stopStream} style={S.stopBtn}>Stop Stream</button>
              </div>
            )}
          </div>
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
    flex: 1, display: 'grid', gridTemplateColumns: '1fr 320px',
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
