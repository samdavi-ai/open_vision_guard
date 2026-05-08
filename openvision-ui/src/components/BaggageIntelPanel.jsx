import React, { useState, useEffect } from 'react';
import { Package, ShieldAlert, ArrowRightLeft, PackageCheck, Clock, Wifi } from 'lucide-react';

const STATE_STYLE = {
  carried:  { color: '#22c55e',  label: 'CARRIED',   bg: 'rgba(34,197,94,0.12)'  },
  put_down: { color: '#f97316',  label: 'ON FLOOR',  bg: 'rgba(249,115,22,0.12)' },
  unowned:  { color: '#94a3b8',  label: 'UNOWNED',   bg: 'rgba(148,163,184,0.1)' },
};

const CLASS_ICON = {
  backpack: '🎒', suitcase: '🧳', handbag: '👜',
  'laptop': '💻', 'cell phone': '📱',
};

function fmtSec(s) {
  if (!s || s < 1) return '0s';
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s/60)}m ${Math.round(s%60)}s`;
}

/* ── Single Bag Card ── */
function BagCard({ bag }) {
  const st = STATE_STYLE[bag.state] || STATE_STYLE.unowned;
  const icon = CLASS_ICON[bag.class_name] || '📦';
  const isAlert = bag.state === 'put_down' && bag.unattended_s > 30;
  const isCritical = bag.unattended_s > 60;

  return (
    <div style={{
      background: isCritical ? 'rgba(239,68,68,0.08)' : isAlert ? 'rgba(249,115,22,0.07)' : 'rgba(15,23,42,0.6)',
      border: `1px solid ${isCritical ? '#ef4444' : isAlert ? '#f97316' : 'rgba(255,255,255,0.08)'}`,
      borderLeft: `4px solid ${isCritical ? '#ef4444' : st.color}`,
      borderRadius: 10, padding: '0.7rem 0.85rem',
      animation: isCritical ? 'pulse 1.5s infinite' : 'none',
    }}>
      {/* Row 1: icon + ID + state badge */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: '1.1rem' }}>{icon}</span>
          <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#e2e8f0', fontFamily: 'monospace' }}>
            {bag.bag_id || bag.global_track_id || 'BAG-???'}
          </span>
        </div>
        <span style={{
          fontSize: '0.65rem', fontWeight: 700, padding: '0.18rem 0.5rem',
          borderRadius: 4, background: st.bg, color: st.color, textTransform: 'uppercase',
        }}>{st.label}</span>
      </div>

      {/* Row 2: owner / unattended */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--text-dim)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span>Owner: <span style={{ color: '#93c5fd', fontWeight: 600 }}>{bag.owner_id || '—'}</span></span>
          {bag.first_owner && bag.owner_id !== bag.first_owner && (
            <span style={{ color: '#f97316' }}>
              <ArrowRightLeft size={10} style={{ display: 'inline', marginRight: 3 }}/>
              Originally: {bag.first_owner}
            </span>
          )}
        </div>
        <div style={{ textAlign: 'right' }}>
          {bag.put_down_count > 0 && (
            <div>Put down: <b style={{ color: '#e2e8f0' }}>{bag.put_down_count}×</b></div>
          )}
          {bag.state === 'put_down' && bag.unattended_s > 0 && (
            <div style={{ color: isCritical ? '#ef4444' : '#f97316', fontWeight: 700 }}>
              <Clock size={10} style={{ display: 'inline', marginRight: 2 }}/>
              {fmtSec(bag.unattended_s)} unattended
            </div>
          )}
          {bag.owner_count > 1 && (
            <div style={{ color: '#a78bfa' }}>{bag.owner_count} handlers</div>
          )}
        </div>
      </div>

      {/* Camera + class */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 5, fontSize: '0.62rem', color: 'rgba(148,163,184,0.6)' }}>
        <span>{bag.class_name}</span>
        <span>{bag.camera_id}</span>
      </div>
    </div>
  );
}

/* ── Main BaggageIntelPanel ── */
export default function BaggageIntelPanel({ apiBase, onClose }) {
  const [bags, setBags] = useState([]);
  const [fusion, setFusion] = useState({});
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');   // all | carried | put_down | alert

  useEffect(() => {
    const fetchBags = async () => {
      try {
        const r = await fetch(`${apiBase}/stream/luggage`);
        if (r.ok) {
          const d = await r.json();
          setBags(d.bags || []);
          setFusion(d.fusion || {});
          setError(null);
        }
      } catch (e) { setError('Cannot reach backend'); }
    };
    fetchBags();
    const id = setInterval(fetchBags, 2000);
    return () => clearInterval(id);
  }, [apiBase]);

  const filtered = bags.filter(b => {
    if (filter === 'carried')  return b.state === 'carried';
    if (filter === 'put_down') return b.state === 'put_down';
    if (filter === 'alert')    return b.unattended_s > 30 || b.owner_count > 1;
    return true;
  });

  const criticalCount = bags.filter(b => b.unattended_s > 60).length;
  const warnCount     = bags.filter(b => b.unattended_s > 30 && b.unattended_s <= 60).length;

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 900,
      background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'flex-start', justifyContent: 'flex-end',
      padding: '60px 10px 10px',
    }} onClick={onClose}>
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: 380, maxWidth: '96vw', height: 'calc(100vh - 80px)',
          background: 'linear-gradient(145deg, #0a1628, #080f1e)',
          border: '1px solid rgba(255,255,255,0.09)',
          borderTop: `3px solid ${criticalCount > 0 ? '#ef4444' : '#3b82f6'}`,
          borderRadius: 14, display: 'flex', flexDirection: 'column',
          boxShadow: '0 24px 60px rgba(0,0,0,0.8)',
          animation: 'slideIn 0.2s ease',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '0.9rem 1rem 0.7rem',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
              <div style={{
                width: 28, height: 28, borderRadius: 7,
                background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>🧳</div>
              <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Luggage Intelligence</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {criticalCount > 0 && (
                <span style={{ fontSize: '0.65rem', padding: '0.15rem 0.45rem', background: 'rgba(239,68,68,0.2)', color: '#ef4444', borderRadius: 999, fontWeight: 700 }}>
                  🚨 {criticalCount} CRITICAL
                </span>
              )}
              {warnCount > 0 && (
                <span style={{ fontSize: '0.65rem', padding: '0.15rem 0.45rem', background: 'rgba(249,115,22,0.2)', color: '#f97316', borderRadius: 999, fontWeight: 700 }}>
                  ⚠ {warnCount} WARN
                </span>
              )}
              <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)', alignSelf: 'center' }}>
                {bags.length} bags tracked
              </span>
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
            color: 'var(--text-dim)', borderRadius: 6, width: 28, height: 28,
            cursor: 'pointer', fontSize: '0.9rem',
          }}>✕</button>
        </div>

        {/* Fusion Stats bar */}
        <div style={{
          padding: '0.5rem 1rem', background: 'rgba(59,130,246,0.06)',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          display: 'flex', gap: 16, fontSize: '0.68rem', color: 'var(--text-dim)',
        }}>
          <span><Wifi size={10} style={{ display: 'inline', marginRight: 3 }}/>Re-ID gallery: <b style={{ color: '#93c5fd' }}>{fusion.gallery_size ?? '—'}</b></span>
          <span>Kalman tracks: <b style={{ color: '#93c5fd' }}>{fusion.kalman_tracks ?? '—'}</b></span>
        </div>

        {/* Filter tabs */}
        <div style={{ display: 'flex', gap: 4, padding: '0.5rem 0.75rem', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          {[
            { key: 'all',      label: `All (${bags.length})` },
            { key: 'carried',  label: `Carried (${bags.filter(b=>b.state==='carried').length})` },
            { key: 'put_down', label: `On Floor (${bags.filter(b=>b.state==='put_down').length})` },
            { key: 'alert',    label: `Alerts (${bags.filter(b=>b.unattended_s>30||b.owner_count>1).length})` },
          ].map(t => (
            <button key={t.key} onClick={() => setFilter(t.key)} style={{
              padding: '0.22rem 0.55rem', borderRadius: 5, cursor: 'pointer',
              fontSize: '0.68rem', fontWeight: 600, fontFamily: 'inherit',
              background: filter === t.key ? 'rgba(59,130,246,0.2)' : 'rgba(255,255,255,0.05)',
              color: filter === t.key ? '#93c5fd' : 'var(--text-dim)',
              border: `1px solid ${filter === t.key ? 'rgba(59,130,246,0.4)' : 'rgba(255,255,255,0.08)'}`,
            }}>{t.label}</button>
          ))}
        </div>

        {/* Bag list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '0.6rem 0.75rem', display: 'flex', flexDirection: 'column', gap: 8 }}>
          {error ? (
            <div style={{ textAlign: 'center', color: '#ef4444', marginTop: 40, fontSize: '0.8rem' }}>{error}</div>
          ) : filtered.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-dim)', marginTop: 60 }}>
              <Package size={28} style={{ opacity: 0.2, marginBottom: 8 }}/>
              <div style={{ fontSize: '0.82rem' }}>No bags in this filter</div>
              <div style={{ fontSize: '0.68rem', marginTop: 4 }}>Start a stream to begin tracking</div>
            </div>
          ) : (
            filtered
              .sort((a, b) => (b.unattended_s || 0) - (a.unattended_s || 0))
              .map(bag => <BagCard key={bag.bag_id || bag.global_track_id} bag={bag} />)
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '0.5rem 1rem', borderTop: '1px solid rgba(255,255,255,0.06)',
          fontSize: '0.62rem', color: 'var(--text-dim)', textAlign: 'center',
        }}>
          Auto-refreshes every 2s · SensorFusion + AirportLuggageTracker
        </div>
      </div>
    </div>
  );
}
