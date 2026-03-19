import React, { useEffect, useState, useRef } from 'react';
import {
  ArrowLeft, Fingerprint, Activity, Package, Clock,
  AlertTriangle, Eye, RefreshCw, Wifi, Shield, Move,
  LogIn, LogOut, Crosshair, Zap, ChevronRight
} from 'lucide-react';

const RISK_COLORS = { low: '#22c55e', medium: '#eab308', high: '#f97316', critical: '#ef4444' };
const RISK_GLOW   = { low: 'rgba(34,197,94,.12)', medium: 'rgba(234,179,8,.12)', high: 'rgba(249,115,22,.12)', critical: 'rgba(239,68,68,.12)' };
const DIR_ARROWS  = { left: '←', right: '→', towards: '↓', away: '↑', stationary: '•' };
const WS_BASE = "ws://localhost:8080";
const API_BASE = "http://localhost:8080";

export default function PersonView({ globalId, cameraId = 'CAM_01', apiBase, onBack }) {
  const [profile, setProfile]       = useState(null);
  const [cropSrc, setCropSrc]       = useState(null);
  const [connected, setConnected]   = useState(false);
  const [loading, setLoading]       = useState(true);
  const [personAlerts, setPersonAlerts] = useState([]);

  const cropWsRef  = useRef(null);
  const profileRef = useRef(null);

  /* ── Live Crop WebSocket — streams person crop at ~20fps ── */
  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/person_crop/${cameraId}/${globalId}`);
    cropWsRef.current = ws;
    ws.onopen  = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.found && d.crop) setCropSrc(`data:image/jpeg;base64,${d.crop}`);
    };
    return () => ws.close();
  }, [globalId, cameraId]);

  /* ── Profile polling (2s refresh) ── */
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

  /* ── Person-specific alerts (5s refresh) ── */
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const r = await fetch(`${apiBase}/identities/${globalId}/alerts`);
        if (r.ok) { const d = await r.json(); setPersonAlerts(d.alerts || []); }
      } catch (_) {}
    };
    fetchAlerts();
    const id = setInterval(fetchAlerts, 5000);
    return () => clearInterval(id);
  }, [globalId, apiBase]);

  const m = profile?.metadata ?? {};
  const history = profile?.history ?? [];
  const risk = m.risk_level || 'low';
  const rc = RISK_COLORS[risk] || RISK_COLORS.low;
  const rg = RISK_GLOW[risk] || RISK_GLOW.low;

  return (
    <div style={S.root}>
      {/* Background */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
        background: `radial-gradient(ellipse at 10% 40%, ${rc}10 0%, transparent 50%), radial-gradient(ellipse at 90% 20%, rgba(59,130,246,0.06) 0%, transparent 50%)` }} />

      {/* Top Bar */}
      <header style={S.topBar}>
        <button onClick={onBack} style={S.backBtn}><ArrowLeft size={14} /> Back to Feed</button>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Crosshair size={16} style={{ color: 'var(--accent)' }} />
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Intelligence Profile</span>
          <span style={S.idBadge}>{globalId}</span>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', color: connected ? '#22c55e' : '#ef4444' }}>
            <Wifi size={12} /> {connected ? 'Live Tracking' : 'Reconnecting...'}
          </div>
          <div style={{ ...S.riskBadge, background: rg, border: `1px solid ${rc}40`, color: rc }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: rc, boxShadow: `0 0 8px ${rc}`, animation: 'pulse 2s infinite' }} />
            {risk.toUpperCase()} RISK
          </div>
        </div>
      </header>

      {/* 3-Column Grid */}
      <div style={S.grid}>

        {/* ═══ LEFT COLUMN: Live Feed + Movement ═══ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

          {/* Live Crop */}
          <Card icon={<Eye size={12} />} title="LIVE TRACKING FEED" accent={connected ? '#22c55e' : '#475569'}>
            <div style={{ background: '#000', minHeight: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              {cropSrc ? (
                <img src={cropSrc} alt="Live" style={{ width: '100%', objectFit: 'contain', maxHeight: 320, display: 'block' }} />
              ) : (
                <div style={{ color: 'var(--text-dim)', fontSize: '0.82rem', textAlign: 'center', padding: 20 }}>
                  <RefreshCw size={20} style={{ opacity: 0.4, marginBottom: 8, animation: 'spin 2s linear infinite' }} />
                  <div>Acquiring target...</div>
                </div>
              )}
              {connected && <div style={{ position: 'absolute', top: 8, right: 8, width: 8, height: 8, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 6px #22c55e', animation: 'pulse 2s infinite' }} />}
            </div>
          </Card>

          {/* Movement & Speed */}
          <Card icon={<Move size={12} />} title="MOVEMENT ANALYSIS">
            <div style={{ padding: 10 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                <InfoCell label="Direction" value={
                  <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: '1.2rem' }}>{DIR_ARROWS[m.movement_direction] || '•'}</span>
                    {m.movement_direction || 'stationary'}
                  </span>
                } />
                <InfoCell label="Speed" value={`${m.speed || 0} px/s`} />
                <InfoCell label="Pose" value={m.pose_detail || m.activity || 'unknown'} />
                <InfoCell label="Activity" value={m.activity || 'unknown'} />
              </div>
            </div>
          </Card>
        </div>

        {/* ═══ CENTER COLUMN: Identity + Threat + Objects ═══ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

          {/* Identity */}
          <Card icon={<Fingerprint size={12} />} title="IDENTITY">
            <div style={{ padding: 10 }}>
              {loading ? <Skeleton rows={3} /> : (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                  <InfoCell label="Name" value={m.face_name || 'Unknown'} wide />
                  <InfoCell label="Global ID" value={globalId} mono />
                  <InfoCell label="Camera" value={m.last_seen_camera || '—'} />
                  <InfoCell label="Last Seen" value={m.last_seen_time ? new Date(m.last_seen_time).toLocaleTimeString() : '—'} />
                  <InfoCell label="Appearances" value={m.total_appearances || history.length || 0} />
                  <InfoCell label="Clothing" value={m.clothing_color || 'N/A'} />
                </div>
              )}
            </div>
          </Card>

          {/* Threat Assessment */}
          <Card icon={<Shield size={12} />} title="THREAT ASSESSMENT">
            <div style={{ padding: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <div style={{
                  width: 40, height: 40, borderRadius: 8,
                  background: rg, border: `1.5px solid ${rc}50`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: rc, fontWeight: 800, fontSize: '0.8rem', textTransform: 'uppercase'
                }}>{risk}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 600, textTransform: 'uppercase', color: rc }}>{risk} Risk Level</div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: 2 }}>
                    {risk === 'critical' && '⚠ Armed / dangerous individual'}
                    {risk === 'high' && '⚠ Suspicious behavior detected'}
                    {risk === 'medium' && 'Potential loitering or unusual activity'}
                    {risk === 'low' && 'No threats detected — normal behavior'}
                  </div>
                </div>
              </div>
              <ThreatBar label="Weapon" level={risk === 'critical' ? 95 : 0} color="#ef4444" />
              <ThreatBar label="Loitering" level={m.activity === 'loitering' ? 70 : (personAlerts.some(a => a.type === 'loitering') ? 50 : 5)} color="#eab308" />
              <ThreatBar label="Suspicious" level={risk === 'high' ? 60 : risk === 'medium' ? 30 : 5} color="#f97316" />
            </div>
          </Card>

          {/* Carried Objects */}
          <Card icon={<Package size={12} />} title="CARRIED OBJECTS">
            <div style={{ padding: 10 }}>
              {m.carried_objects?.length > 0 ? (
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {m.carried_objects.map((o, i) => <span key={i} style={S.tag}>🎒 {o}</span>)}
                </div>
              ) : (
                <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>No objects detected</span>
              )}
              {/* Object acquisition log */}
              {m.object_log?.length > 0 && (
                <div style={{ marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 8 }}>
                  <div style={{ fontSize: '0.62rem', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: 4 }}>Acquisition Log</div>
                  {m.object_log.map((entry, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', padding: '3px 0', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                      <span><span style={{ color: '#60a5fa' }}>{entry.action}</span> {entry.object}</span>
                      <span style={{ color: 'var(--text-dim)' }}>{entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : ''}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* ═══ RIGHT COLUMN: Entry/Exit + Alerts + Timeline ═══ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0 }}>

          {/* Entry / Exit Log */}
          <Card icon={<LogIn size={12} />} title="ENTRY / EXIT LOG">
            <div style={{ padding: 10 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                <InfoCell label="Entry Time" value={m.entry_time ? new Date(m.entry_time).toLocaleTimeString() : '—'} icon={<LogIn size={10} color="#22c55e" />} />
                <InfoCell label="Last Seen" value={m.exit_time ? new Date(m.exit_time).toLocaleTimeString() : '—'} icon={<LogOut size={10} color="#f97316" />} />
                <InfoCell label="Duration" value={
                  m.entry_time && m.exit_time
                    ? formatDuration(new Date(m.exit_time) - new Date(m.entry_time))
                    : '—'
                } />
                <InfoCell label="Camera" value={m.last_seen_camera || '—'} />
              </div>
            </div>
          </Card>

          {/* Person-Specific Alerts */}
          <Card icon={<AlertTriangle size={12} />} title={`ALERTS (${personAlerts.length})`} accent={personAlerts.length > 0 ? '#ef4444' : undefined}>
            <div style={{ padding: 10, maxHeight: 160, overflowY: 'auto' }}>
              {personAlerts.length > 0 ? personAlerts.slice(0, 20).map((a, i) => (
                <div key={i} style={{
                  padding: '6px 8px', marginBottom: 4, borderRadius: 6,
                  background: 'rgba(0,0,0,0.3)', borderLeft: `3px solid ${RISK_COLORS[a.severity] || '#eab308'}`,
                  fontSize: '0.72rem',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                    <span style={{ color: RISK_COLORS[a.severity] || '#eab308', fontWeight: 600, textTransform: 'uppercase', fontSize: '0.62rem' }}>{a.type}</span>
                    <span style={{ color: 'var(--text-dim)', fontSize: '0.6rem' }}>{a.timestamp ? new Date(a.timestamp).toLocaleTimeString() : ''}</span>
                  </div>
                  <div style={{ color: 'var(--text-main)' }}>{a.message}</div>
                </div>
              )) : (
                <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>No alerts for this person</span>
              )}
            </div>
          </Card>

          {/* Event Timeline */}
          <Card icon={<Clock size={12} />} title="EVENT TIMELINE" style={{ flex: 1, minHeight: 0 }}>
            <div style={{ padding: 10, overflowY: 'auto', flex: 1 }}>
              {history.length > 0 ? (
                <div style={{ position: 'relative', paddingLeft: 14 }}>
                  <div style={{ position: 'absolute', left: 5, top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.06)' }} />
                  {history.slice(0, 40).map((ev, i) => (
                    <div key={i} style={{ position: 'relative', marginBottom: 8 }}>
                      <div style={{ position: 'absolute', left: -13, top: 5, width: 7, height: 7, borderRadius: '50%', background: 'var(--accent)', border: '1.5px solid var(--bg-color)' }} />
                      <div style={{ fontSize: '0.62rem', color: 'var(--text-dim)' }}>{ev.timestamp ? new Date(ev.timestamp).toLocaleString() : ''}</div>
                      <div style={{ fontSize: '0.74rem', marginTop: 1 }}>
                        <span style={{ color: '#60a5fa' }}>{ev.activity || 'Detected'}</span>
                        {' '}— {ev.camera_id}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>No events recorded yet</span>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

/* ─── Sub-components ─── */

function Card({ icon, title, accent, style, children }) {
  return (
    <div style={{ ...S.card, ...style }}>
      <div style={{ ...S.cardHdr, ...(accent ? { borderLeftColor: accent } : {}) }}>
        {icon} {title}
      </div>
      {children}
    </div>
  );
}

function InfoCell({ label, value, color, mono, icon, wide }) {
  return (
    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.45rem 0.6rem', borderRadius: 6, border: '1px solid rgba(255,255,255,0.04)', ...(wide && { gridColumn: 'span 2' }) }}>
      <div style={{ fontSize: '0.55rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2, display: 'flex', alignItems: 'center', gap: 3 }}>
        {icon}{label}
      </div>
      <div style={{ fontWeight: 600, fontSize: '0.76rem', color: color || 'var(--text-main)', fontFamily: mono ? 'monospace' : 'inherit', textTransform: 'capitalize', wordBreak: 'break-all' }}>
        {typeof value === 'string' ? value : value}
      </div>
    </div>
  );
}

function ThreatBar({ label, level, color }) {
  return (
    <div style={{ marginBottom: 5 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--text-dim)', marginBottom: 2 }}>
        <span>{label}</span><span>{level}%</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.06)' }}>
        <div style={{ height: '100%', borderRadius: 2, width: `${level}%`, background: color, transition: 'width 0.5s ease', boxShadow: level > 30 ? `0 0 6px ${color}` : 'none' }} />
      </div>
    </div>
  );
}

function Skeleton({ rows = 2 }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
      {Array.from({ length: rows * 2 }).map((_, i) => (
        <div key={i} style={{ height: 44, borderRadius: 6, background: 'rgba(255,255,255,0.04)', animation: 'pulse 1.5s infinite' }} />
      ))}
    </div>
  );
}

function formatDuration(ms) {
  if (!ms || ms < 0) return '—';
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const min = Math.floor(s / 60);
  const sec = s % 60;
  if (min < 60) return `${min}m ${sec}s`;
  const hr = Math.floor(min / 60);
  return `${hr}h ${min % 60}m`;
}

/* ─── Styles ─── */
const S = {
  root: {
    position: 'fixed', inset: 0, background: 'var(--bg-color)', zIndex: 100,
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
  },
  topBar: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '0.5rem 1rem', borderBottom: '1px solid var(--border-color)',
    background: 'rgba(8,15,30,0.98)', flexShrink: 0, zIndex: 1,
  },
  backBtn: {
    display: 'flex', alignItems: 'center', gap: 5,
    background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)',
    color: 'var(--text-main)', padding: '0.28rem 0.7rem', borderRadius: 7,
    cursor: 'pointer', fontSize: '0.78rem', fontFamily: 'inherit',
    transition: 'all 0.15s ease',
  },
  idBadge: {
    fontSize: '0.7rem', color: 'var(--text-dim)', fontFamily: 'monospace',
    background: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: 4,
  },
  riskBadge: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '0.25rem 0.65rem', borderRadius: 999,
    fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.04em',
  },
  grid: {
    flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
    gap: 8, padding: 8, overflow: 'hidden', minHeight: 0, zIndex: 1,
  },
  card: {
    background: 'rgba(10,20,40,0.85)', border: '1px solid var(--border-color)',
    borderRadius: 8, display: 'flex', flexDirection: 'column', overflow: 'hidden',
  },
  cardHdr: {
    padding: '0.4rem 0.75rem', borderBottom: '1px solid var(--border-color)',
    fontWeight: 600, fontSize: '0.64rem', textTransform: 'uppercase', letterSpacing: '0.06em',
    color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0,
    borderLeft: '3px solid var(--accent)',
  },
  tag: {
    background: 'rgba(59,130,246,0.15)', color: '#60a5fa',
    padding: '0.2rem 0.5rem', borderRadius: 999, fontSize: '0.7rem', fontWeight: 500,
  },
};
