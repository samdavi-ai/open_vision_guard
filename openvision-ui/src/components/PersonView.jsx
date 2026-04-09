import React, { useEffect, useState, useRef } from 'react';
import {
  ArrowLeft, Fingerprint, Activity, Package, Clock,
  AlertTriangle, Eye, RefreshCw, Wifi, Shield, Move,
  LogIn, LogOut, Crosshair, Zap, ChevronRight, MapPin, BarChart3
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer
} from 'recharts';

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
  const [activeTab, setActiveTab]   = useState('profile');
  const [movementLogs, setMovementLogs] = useState([]);

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
  
  const [faceLogs, setFaceLogs] = useState([]);
  const [presenceLogs, setPresenceLogs] = useState([]);

  /* ── Face & Presence Logs polling (5s refresh) ── */
  useEffect(() => {
    const fetchLogs = async () => {
      if (activeTab !== 'logs') return;
      try {
        const [r1, r2] = await Promise.all([
          fetch(`${apiBase}/face-logs/${globalId}`),
          fetch(`${apiBase}/analytics/presence/${globalId}`)
        ]);
        if (r1.ok) setFaceLogs(await r1.json());
        if (r2.ok) setPresenceLogs(await r2.json());
      } catch (_) {}
    };
    fetchLogs();
    const id = setInterval(fetchLogs, 5000);
    return () => clearInterval(id);
  }, [globalId, apiBase, activeTab]);

  /* ── Movement Logs polling (3s refresh) ── */
  useEffect(() => {
    const fetchMovement = async () => {
      if (activeTab !== 'analytics') return;
      try {
        const r = await fetch(`${apiBase}/analytics/movement/${globalId}`);
        if (r.ok) {
          const d = await r.json();
          // Map timestamps for charts
          const mapped = (d.movement_logs || []).map(log => ({
            ...log,
            timeLabel: new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            posX: Math.round(log.position_x || 0),
            posY: Math.round(log.position_y || 0)
          }));
          setMovementLogs(mapped);
        }
      } catch (_) {}
    };
    fetchMovement();
    const id = setInterval(fetchMovement, 3000);
    return () => clearInterval(id);
  }, [globalId, apiBase, activeTab]);

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
          {/* Tab Switcher */}
          <div style={{ display: 'flex', background: 'rgba(0,0,0,0.4)', borderRadius: 8, padding: 3, marginRight: 15, border: '1px solid rgba(255,255,255,0.05)' }}>
            <button 
              onClick={() => setActiveTab('profile')}
              style={{ ...S.tabBtn, background: activeTab === 'profile' ? 'rgba(59,130,246,0.2)' : 'transparent', color: activeTab === 'profile' ? '#fff' : 'var(--text-dim)' }}
            >
              <Fingerprint size={12} /> Profile
            </button>
            <button 
              onClick={() => setActiveTab('analytics')}
              style={{ ...S.tabBtn, background: activeTab === 'analytics' ? 'rgba(59,130,246,0.2)' : 'transparent', color: activeTab === 'analytics' ? '#fff' : 'var(--text-dim)' }}
            >
              <BarChart3 size={12} /> Analytics
            </button>
            <button 
              onClick={() => setActiveTab('logs')}
              style={{ ...S.tabBtn, background: activeTab === 'logs' ? 'rgba(59,130,246,0.2)' : 'transparent', color: activeTab === 'logs' ? '#fff' : 'var(--text-dim)' }}
            >
              <Shield size={12} /> Logs
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', color: connected ? '#22c55e' : '#ef4444' }}>
            <Wifi size={12} /> {connected ? 'Live Tracking' : 'Reconnecting...'}
          </div>
          <div style={{ ...S.riskBadge, background: rg, border: `1px solid ${rc}40`, color: rc }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: rc, boxShadow: `0 0 8px ${rc}`, animation: 'pulse 2s infinite' }} />
            {risk.toUpperCase()} RISK
          </div>
        </div>
      </header>

      {activeTab === 'profile' ? (
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

          {/* Behaviour Analysis */}
          <Card icon={<Activity size={12} />} title="BEHAVIOUR ANALYSIS">
            <div style={{ padding: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-main)', textTransform: 'capitalize' }}>
                  {m.behaviour_label ? m.behaviour_label.replace('_', ' ') : 'Normal walking'}
                </span>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: m.behaviour_score > 50 ? '#ef4444' : '#22c55e' }}>
                  {Math.round(m.behaviour_score || 0)} / 100
                </span>
              </div>
              <ThreatBar label="Abnormality Level" level={m.behaviour_score || 0} color={m.behaviour_score > 50 ? '#ef4444' : '#eab308'} />
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
                  <InfoCell label="Last Seen" value={m.last_seen_time ? new Date(m.last_seen_time).toLocaleString() : '—'} />
                  <InfoCell label="Location" value={m.latitude && m.longitude ? `${m.latitude.toFixed(6)}, ${m.longitude.toFixed(6)}` : '—'} />
                  <InfoCell label="Appearances" value={m.total_appearances || history.length || 0} />
                  <InfoCell label="Clothing" value={m.clothing_color || 'N/A'} wide />
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
                }}>{Math.round(m.risk_score || 0)}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 600, textTransform: 'uppercase', color: rc }}>{risk} Risk Level</div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: 2 }}>
                    Composite threat score based on all signals
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {m.risk_factors?.length > 0 ? (
                  m.risk_factors.map(f => <span key={f} style={{...S.tag, background: 'rgba(239,68,68,0.1)', color: '#ef4444'}}>⚠ {f.replace(/_/g, ' ')}</span>)
                ) : (
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>No active risk factors</span>
                )}
              </div>
            </div>
          </Card>

          {/* Camera Avoidance */}
          <Card icon={<Eye size={12} />} title="CAMERA AVOIDANCE" accent="#f59e0b">
            <div style={{ padding: 10 }}>
              <ThreatBar label="Avoidance Score" level={m.avoidance_score || 0} color="#f59e0b" />
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
                {m.avoidance_flags?.map(f => <span key={f} style={{...S.tag, background: 'rgba(245,158,11,0.1)', color: '#f59e0b'}}>{f.replace(/_/g, ' ')}</span>)}
              </div>
            </div>
          </Card>

          {/* Luggage Ownership */}
          <Card icon={<Package size={12} />} title="LUGGAGE TRACKING">
            <div style={{ padding: 10 }}>
              {m.luggage_status && Object.keys(m.luggage_status).length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {Object.entries(m.luggage_status).map(([lid, l]) => (
                    <div key={lid} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 8px', background: 'rgba(255,255,255,0.03)', borderRadius: 5, border: '1px solid rgba(255,255,255,0.05)' }}>
                      <span style={{ fontSize: '0.78rem', textTransform: 'capitalize' }}>🎒 {l.type}</span>
                      <span style={{ fontSize: '0.7rem', color: l.status === 'abandoned' ? '#ef4444' : l.status === 'carried' ? '#22c55e' : '#f97316', fontWeight: 600 }}>{l.status}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>No luggage tracked</span>
              )}
            </div>
          </Card>
        </div>

        {/* ═══ RIGHT COLUMN: Entry/Exit + Alerts + Timeline ═══ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0 }}>

          {/* Presence & Frequency */}
          <Card icon={<LogIn size={12} />} title="PRESENCE &amp; FREQUENCY">
            <div style={{ padding: 10 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                <InfoCell label="Dwell Time" value={formatDuration((m.dwell_time_seconds || 0) * 1000)} icon={<Clock size={10} />} />
                <InfoCell label="Visits" value={m.visit_count || 1} />
                <InfoCell label="Frequency" value={m.frequency_label || 'new'} color="#60a5fa" wide />
                <InfoCell label="Entry Time" value={m.entry_time ? new Date(m.entry_time).toLocaleTimeString() : '—'} />
                <InfoCell label="Last Seen" value={m.exit_time ? new Date(m.exit_time).toLocaleTimeString() : '—'} />
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

          {/* Camera Map Timeline */}
          <Card icon={<MapPin size={12} />} title="CAMERA MAP &amp; TIMELINE" style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <CameraMapTimeline history={history} />
          </Card>
        </div>
      </div>
      ) : activeTab === 'analytics' ? (
        /* ═══ ANALYTICS TAB ═══ */
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 15, overflow: 'hidden' }}>
          <Card icon={<BarChart3 size={14} />} title="MOVEMENT & KINETICS ANALYTICS (REAL-TIME)" style={{ flex: 1 }}>
            <div style={{ flex: 1, padding: 20, minHeight: 0 }}>
              {movementLogs.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={movementLogs} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="timeLabel" stroke="var(--text-dim)" fontSize={11} tickMargin={10} minTickGap={30} />
                    
                    {/* Primary Y-Axis (Location/Position X) */}
                    <YAxis yAxisId="left" stroke="#3b82f6" fontSize={11} label={{ value: 'X Position (Geo Context)', angle: -90, position: 'insideLeft', fill: 'var(--text-dim)', fontSize: 11 }} />
                    
                    {/* Secondary Y-Axis (Speed) */}
                    <YAxis yAxisId="right" orientation="right" stroke="#eab308" fontSize={11} label={{ value: 'Speed (px/s)', angle: 90, position: 'insideRight', fill: 'var(--text-dim)', fontSize: 11 }} />
                    
                    <RechartsTooltip 
                      contentStyle={{ backgroundColor: 'rgba(8,15,30,0.95)', border: '1px solid var(--border-color)', borderRadius: 8, fontSize: '0.75rem' }}
                      itemStyle={{ fontWeight: 600 }}
                      labelStyle={{ color: 'var(--text-dim)', marginBottom: 5 }}
                    />
                    <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: 10 }} />
                    
                    {/* Lines */}
                    <Line yAxisId="left" type="monotone" dataKey="posX" name="Floor Geo Location X" stroke="#3b82f6" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
                    <Line yAxisId="left" type="monotone" dataKey="posY" name="Floor Geo Location Y" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                    <Line yAxisId="right" type="monotone" dataKey="speed" name="Kinetic Speed" stroke="#eab308" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-dim)' }}>
                  <RefreshCw size={24} style={{ animation: 'spin 2s linear infinite', marginBottom: 10, opacity: 0.5 }} />
                  Gathering movement telemetry...
                </div>
              )}
            </div>
          </Card>
        </div>
      ) : (
        /* ═══ LOGS TAB ═══ */
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 10, padding: 10, overflow: 'hidden' }}>
          
          {/* Face Recognition Log */}
          <Card icon={<Eye size={14} />} title="FACE RECOGNITION LOG" style={{ overflow: 'hidden' }}>
            <div style={{ flex: 1, overflowY: 'auto', padding: 10 }}>
              {faceLogs.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 }}>
                  {faceLogs.map((log, i) => (
                    <div key={i} style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 8, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.05)' }}>
                       <div style={{ height: 100, background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                         <img 
                           src={`${API_BASE}/static/${log.crop_path}`} 
                           alt="Face Crop" 
                           style={{ height: '100%', width: '100%', objectFit: 'cover' }}
                           onError={(e) => { e.target.src = 'https://via.placeholder.com/100?text=No+Image'; }}
                         />
                       </div>
                       <div style={{ padding: 8 }}>
                         <div style={{ fontSize: '0.7rem', fontWeight: 700, color: log.match_status === 'known' ? '#22c55e' : 'var(--text-dim)' }}>
                           {log.face_name || 'Unknown'}
                         </div>
                         <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)', marginTop: 2 }}>
                           {new Date(log.timestamp).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                         </div>
                       </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-dim)' }}>No face logs recorded</div>
              )}
            </div>
          </Card>

          {/* Presence Timeline Log */}
          <Card icon={<Clock size={14} />} title="PRESENCE TIMELINE" style={{ overflow: 'hidden' }}>
             <div style={{ flex: 1, overflowY: 'auto', padding: 10 }}>
               {presenceLogs.length > 0 ? (
                 <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {presenceLogs.map((log, i) => (
                      <div key={i} style={{ padding: 8, borderRadius: 6, background: log.event_type === 'entry' ? 'rgba(34,197,94,0.05)' : 'rgba(239,68,68,0.05)', border: `1px solid ${log.event_type === 'entry' ? '#22c55e30' : '#ef444430'}` }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                          <span style={{ fontSize: '0.65rem', fontWeight: 800, textTransform: 'uppercase', color: log.event_type === 'entry' ? '#22c55e' : '#ef4444' }}>
                            {log.event_type === 'entry' ? <LogIn size={10} style={{marginRight:4}} /> : <LogOut size={10} style={{marginRight:4}} />}
                            {log.event_type}
                          </span>
                          <span style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>{new Date(log.timestamp).toLocaleString()}</span>
                        </div>
                        {log.session_duration > 0 && (
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-main)' }}>
                             Dwell Time: {formatDuration(log.session_duration * 1000)}
                          </div>
                        )}
                        <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)', marginTop: 2 }}>{log.camera_id}</div>
                      </div>
                    ))}
                 </div>
               ) : (
                 <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-dim)' }}>No presence history</div>
               )}
             </div>
          </Card>
        </div>
      )}
    </div>
  );
}

/* ─── CameraMapTimeline ─── */

function CameraMapTimeline({ history }) {
  if (!history || history.length === 0) {
    return (
      <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.78rem' }}>
        No movement data recorded yet
      </div>
    );
  }

  // Build ordered list of unique cameras (in first-seen order)
  const seenCams = [];
  const camSet = new Set();
  history.forEach(ev => {
    if (ev.camera_id && !camSet.has(ev.camera_id)) {
      camSet.add(ev.camera_id);
      seenCams.push(ev.camera_id);
    }
  });

  const lastCam = history[history.length - 1]?.camera_id;

  // ── Camera node row ──
  const nodeW = 72, nodeH = 38, gap = 36;
  const svgW = seenCams.length > 1 ? (seenCams.length - 1) * (nodeW + gap) + nodeW : nodeW;
  const svgH = nodeH + 4;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>

      {/* ── Camera Node Map ── */}
      <div style={{ padding: '10px 10px 6px', flexShrink: 0 }}>
        <div style={{ fontSize: '0.58rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          Movement Path
        </div>
        <div style={{ overflowX: 'auto', paddingBottom: 4 }}>
          <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', gap: 0, minWidth: svgW }}>
            {seenCams.map((cam, i) => {
              const isLast = cam === lastCam;
              const isFirst = i === 0;
              return (
                <React.Fragment key={cam}>
                  {/* Connector line */}
                  {i > 0 && (
                    <div style={{
                      width: gap, height: 2,
                      background: 'linear-gradient(to right, rgba(59,130,246,0.4), rgba(59,130,246,0.2))',
                      flexShrink: 0, position: 'relative',
                    }}>
                      {/* Arrow head */}
                      <div style={{
                        position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)',
                        width: 0, height: 0,
                        borderTop: '4px solid transparent', borderBottom: '4px solid transparent',
                        borderLeft: '6px solid rgba(59,130,246,0.5)',
                      }} />
                    </div>
                  )}
                  {/* Camera Node */}
                  <div style={{
                    width: nodeW, flexShrink: 0,
                    background: isLast
                      ? 'linear-gradient(135deg, rgba(59,130,246,0.25), rgba(139,92,246,0.2))'
                      : 'rgba(255,255,255,0.04)',
                    border: `1.5px solid ${isLast ? 'rgba(59,130,246,0.6)' : 'rgba(255,255,255,0.08)'}`,
                    borderRadius: 7,
                    padding: '5px 6px',
                    textAlign: 'center',
                    boxShadow: isLast ? '0 0 12px rgba(59,130,246,0.25)' : 'none',
                    transition: 'all 0.3s ease',
                    position: 'relative',
                  }}>
                    {isFirst && (
                      <div style={{ position: 'absolute', top: -8, left: '50%', transform: 'translateX(-50%)', fontSize: '0.48rem', color: '#22c55e', fontWeight: 700, letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>ENTRY</div>
                    )}
                    {isLast && (
                      <div style={{ position: 'absolute', top: -8, left: '50%', transform: 'translateX(-50%)', fontSize: '0.48rem', color: '#60a5fa', fontWeight: 700, letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>LAST SEEN</div>
                    )}
                    <div style={{ fontSize: '0.56rem', color: isLast ? '#93c5fd' : 'var(--text-dim)', marginBottom: 2 }}>📹</div>
                    <div style={{ fontSize: '0.68rem', fontWeight: 700, color: isLast ? '#fff' : 'var(--text-main)', fontFamily: 'monospace', letterSpacing: '0.02em' }}>{cam}</div>
                    {isLast && (
                      <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 6px #22c55e', animation: 'pulse 2s infinite', margin: '3px auto 0' }} />
                    )}
                  </div>
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: 'rgba(255,255,255,0.05)', flexShrink: 0 }} />

      {/* ── Scrollable Event Log ── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 10px', minHeight: 0 }}>
        <div style={{ fontSize: '0.58rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          Trace Log — {history.length} events
        </div>
        <div style={{ position: 'relative', paddingLeft: 16 }}>
          {/* Vertical rail */}
          <div style={{ position: 'absolute', left: 5, top: 4, bottom: 4, width: 1, background: 'rgba(255,255,255,0.06)' }} />

          {history.slice().reverse().slice(0, 60).map((ev, i) => {
            const isActive = ev.camera_id === lastCam && i === 0;
            const dotColor = isActive ? '#22c55e' : 'var(--accent)';
            return (
              <div key={i} style={{ position: 'relative', marginBottom: 7 }}>
                {/* Dot */}
                <div style={{
                  position: 'absolute', left: -13, top: 5,
                  width: isActive ? 8 : 6, height: isActive ? 8 : 6,
                  borderRadius: '50%',
                  background: dotColor,
                  border: `1.5px solid var(--bg-color)`,
                  boxShadow: isActive ? `0 0 6px ${dotColor}` : 'none',
                  animation: isActive ? 'pulse 2s infinite' : 'none',
                  transition: 'all 0.3s ease',
                }} />

                {/* Content */}
                <div style={{ background: isActive ? 'rgba(34,197,94,0.06)' : 'rgba(0,0,0,0.2)', borderRadius: 5, padding: '4px 7px', border: `1px solid ${isActive ? 'rgba(34,197,94,0.2)' : 'rgba(255,255,255,0.04)'}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 1 }}>
                    <span style={{ fontSize: '0.68rem', fontWeight: 600, fontFamily: 'monospace', color: isActive ? '#93c5fd' : '#60a5fa' }}>
                      {ev.camera_id}
                    </span>
                    <span style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>
                      {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-main)', textTransform: 'capitalize' }}>
                    {ev.activity || 'detected'}
                    {ev.location ? <span style={{ color: 'var(--text-dim)' }}> · {ev.location}</span> : null}
                  </div>
                </div>
              </div>
            );
          })}
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
  tabBtn: {
    padding: '4px 12px', display: 'flex', alignItems: 'center', gap: 6,
    borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: '0.7rem',
    fontWeight: 600, transition: 'all 0.2s ease', fontFamily: 'inherit'
  }
};
