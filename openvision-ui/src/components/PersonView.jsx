import React, { useEffect, useState, useRef } from 'react';
import {
  ArrowLeft, Fingerprint, Activity, Package, Clock,
  AlertTriangle, Eye, RefreshCw, Wifi
} from 'lucide-react';

const RISK_COLORS = { low: '#22c55e', medium: '#eab308', high: '#f97316', critical: '#ef4444' };
const WS_BASE = "ws://localhost:8000";

export default function PersonView({ globalId, cameraId = 'CAM_01', apiBase, onBack }) {
  const [profile, setProfile]   = useState(null);
  const [cropSrc, setCropSrc]   = useState(null);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading]   = useState(true);

  const cropWsRef   = useRef(null);
  const profileRef  = useRef(null);

  /* ── Live Crop WebSocket — streams person crop at ~20fps ── */
  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/person_crop/${cameraId}/${globalId}`);
    cropWsRef.current = ws;

    ws.onopen  = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.found && d.crop) {
        setCropSrc(`data:image/jpeg;base64,${d.crop}`);
      }
    };
    return () => ws.close();
  }, [globalId, cameraId]);

  /* ── Profile polling (metadata only, 2s refresh — no flicker) ── */
  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const r = await fetch(`${apiBase}/identities/${globalId}`);
        if (r.ok) { setProfile(await r.json()); setLoading(false); }
      } catch (_) {}
    };
    fetchProfile();
    profileRef.current = setInterval(fetchProfile, 2000);
    return () => clearInterval(profileRef.current);
  }, [globalId, apiBase]);

  const m = profile?.metadata ?? {};
  const history = profile?.history ?? [];
  const risk = m.risk_level || 'low';
  const rc = RISK_COLORS[risk] || RISK_COLORS.low;

  return (
    <div style={S.root}>
      {/* Background glow */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
        background: `radial-gradient(ellipse at 15% 50%, ${rc}14 0%, transparent 45%), radial-gradient(ellipse at 85% 25%, rgba(59,130,246,0.07) 0%, transparent 45%)` }} />

      {/* Top Bar */}
      <header style={S.topBar}>
        <button onClick={onBack} style={S.backBtn}>
          <ArrowLeft size={15} /> Back
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Eye size={17} style={{ color: 'var(--accent)' }} />
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Intelligence Profile</span>
          <span style={S.idBadge}>{globalId}</span>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', color: connected ? '#22c55e' : '#ef4444' }}>
            <Wifi size={12} /> {connected ? 'Live' : 'Reconnecting...'}
          </div>
          <div style={{ ...S.riskBadge, background: `${rc}18`, border: `1px solid ${rc}40`, color: rc }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: rc, boxShadow: `0 0 8px ${rc}` }} />
            {risk} risk
          </div>
        </div>
      </header>

      {/* Content Grid */}
      <div style={S.grid}>
        {/* LEFT: Live Feed + Objects */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          
          {/* Live Crop — WebSocket, no polling flicker */}
          <div style={S.card}>
            <div style={S.cardHdr}><Eye size={12} /> Live Tracking Feed</div>
            <div style={{ background: '#000', minHeight: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              {cropSrc ? (
                <img
                  src={cropSrc}
                  alt="Live person crop"
                  style={{ width: '100%', objectFit: 'contain', maxHeight: 400, display: 'block' }}
                />
              ) : (
                <div style={{ color: 'var(--text-dim)', fontSize: '0.82rem', textAlign: 'center', padding: 20 }}>
                  <RefreshCw size={20} style={{ opacity: 0.4, marginBottom: 8 }} />
                  <div>Waiting for detection...</div>
                </div>
              )}
              {connected && (
                <div style={{ position: 'absolute', top: 8, right: 8, width: 8, height: 8, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 6px #22c55e', animation: 'pulse 2s infinite' }} />
              )}
            </div>
          </div>

          {/* Carried Objects */}
          <div style={S.card}>
            <div style={S.cardHdr}><Package size={12} /> Carried Objects</div>
            <div style={{ padding: 12 }}>
              {m.carried_objects?.length > 0 ? (
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {m.carried_objects.map((o, i) => <span key={i} style={S.tag}>🎒 {o}</span>)}
                </div>
              ) : (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>No objects detected</span>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: Identity + Behavior + Timeline */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0 }}>

          {/* Identity */}
          <div style={S.card}>
            <div style={S.cardHdr}><Fingerprint size={12} /> Identity</div>
            <div style={{ padding: 12 }}>
              {loading ? <Skeleton rows={4} /> : (
                <div style={S.infoGrid}>
                  <InfoCell label="Name" value={m.face_name || 'Unknown'} />
                  <InfoCell label="Global ID" value={globalId} mono />
                  <InfoCell label="Risk Level" value={risk} color={rc} />
                  <InfoCell label="Appearances" value={history.length || 0} />
                  <InfoCell label="Last Camera" value={m.last_seen_camera || '—'} />
                  <InfoCell label="Last Seen" value={m.last_seen_time ? new Date(m.last_seen_time).toLocaleTimeString() : '—'} />
                </div>
              )}
            </div>
          </div>

          {/* Behavior */}
          <div style={S.card}>
            <div style={S.cardHdr}><Activity size={12} /> Behavior Analysis</div>
            <div style={{ padding: 12 }}>
              {loading ? <Skeleton rows={2} /> : (
                <div style={S.infoGrid}>
                  <InfoCell label="Activity" value={m.activity || 'unknown'} />
                  <InfoCell label="Clothing" value={m.clothing_color || 'N/A'} />
                </div>
              )}
            </div>
          </div>

          {/* Timeline */}
          <div style={{ ...S.card, flex: 1, minHeight: 0 }}>
            <div style={S.cardHdr}><Clock size={12} /> Event Timeline</div>
            <div style={{ padding: 12, overflowY: 'auto', flex: 1 }}>
              {history.length > 0 ? (
                <div style={{ position: 'relative', paddingLeft: 14 }}>
                  <div style={{ position: 'absolute', left: 5, top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.07)' }} />
                  {history.slice(0, 30).map((ev, i) => (
                    <div key={i} style={{ position: 'relative', marginBottom: 10 }}>
                      <div style={{ position: 'absolute', left: -13, top: 5, width: 7, height: 7, borderRadius: '50%', background: 'var(--accent)', border: '1.5px solid var(--bg-color)' }} />
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>{new Date(ev.timestamp).toLocaleString()}</div>
                      <div style={{ fontSize: '0.8rem', marginTop: 1 }}>{ev.activity || 'Detected'} — {ev.camera_id}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>No events recorded yet</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Sub-components ─── */
function InfoCell({ label, value, color, mono }) {
  return (
    <div style={{ background: 'rgba(0,0,0,0.35)', padding: '0.5rem 0.65rem', borderRadius: 7, border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ fontSize: '0.58rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>{label}</div>
      <div style={{ fontWeight: 600, fontSize: '0.8rem', color: color || 'var(--text-main)', fontFamily: mono ? 'monospace' : 'inherit', textTransform: 'capitalize', wordBreak: 'break-all' }}>
        {String(value)}
      </div>
    </div>
  );
}

function Skeleton({ rows = 2 }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
      {Array.from({ length: rows * 2 }).map((_, i) => (
        <div key={i} style={{ height: 50, borderRadius: 7, background: 'rgba(255,255,255,0.04)' }} />
      ))}
    </div>
  );
}

/* ─── Styles ─── */
const S = {
  root: {
    position: 'fixed', inset: 0, background: 'var(--bg-color)', zIndex: 100,
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
  },
  topBar: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '0.55rem 1rem', borderBottom: '1px solid var(--border-color)',
    background: 'rgba(8,15,30,0.98)', flexShrink: 0, zIndex: 1,
  },
  backBtn: {
    display: 'flex', alignItems: 'center', gap: 5,
    background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)',
    color: 'var(--text-main)', padding: '0.3rem 0.75rem', borderRadius: 7,
    cursor: 'pointer', fontSize: '0.8rem', fontFamily: 'inherit',
  },
  idBadge: {
    fontSize: '0.74rem', color: 'var(--text-dim)', fontFamily: 'monospace',
    background: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: 4,
  },
  riskBadge: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '0.28rem 0.7rem', borderRadius: 999,
    fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase',
  },
  grid: {
    flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr',
    gap: 10, padding: 10, overflow: 'hidden', minHeight: 0, zIndex: 1,
  },
  card: {
    background: 'rgba(10,20,40,0.85)', border: '1px solid var(--border-color)',
    borderRadius: 10, display: 'flex', flexDirection: 'column', overflow: 'hidden',
  },
  cardHdr: {
    padding: '0.5rem 0.85rem', borderBottom: '1px solid var(--border-color)',
    fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em',
    color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0,
  },
  infoGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 },
  tag: {
    background: 'rgba(59,130,246,0.15)', color: '#60a5fa',
    padding: '0.22rem 0.55rem', borderRadius: 999, fontSize: '0.74rem', fontWeight: 500,
  },
};
