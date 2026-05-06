import React, { useState } from 'react';
import { AlertTriangle, Clock, Package, PackageCheck, PackageMinus, ArrowRightLeft, X, Camera, ShieldAlert, ZoomIn } from 'lucide-react';

/* ── Severity helpers ── */
const SEV_COLOR = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#22c55e' };
const SEV_BG    = { critical: 'rgba(239,68,68,0.10)', high: 'rgba(249,115,22,0.10)', medium: 'rgba(234,179,8,0.10)', low: 'rgba(34,197,94,0.10)' };

/* ── Alert type display config ── */
const ALERT_CONFIG = {
  baggage_swap:       { icon: <ArrowRightLeft size={14}/>, label: 'BAGGAGE SWAP',      emoji: '🔴' },
  baggage_taken:      { icon: <PackageCheck size={14}/>,   label: 'ITEM TAKEN',        emoji: '🚨' },
  baggage_left_behind:{ icon: <PackageMinus size={14}/>,   label: 'ITEM LEFT BEHIND',  emoji: '🚨' },
  luggage_abandoned:  { icon: <Package size={14}/>,        label: 'LUGGAGE ABANDONED', emoji: '⚠️' },
  loitering:          { icon: <ShieldAlert size={14}/>,    label: 'LOITERING',         emoji: '⚠️' },
  sudden_movement:    { icon: <AlertTriangle size={14}/>,  label: 'SUDDEN MOVEMENT',   emoji: '⚠️' },
  luggage_theft:      { icon: <Package size={14}/>,        label: 'LUGGAGE THEFT',     emoji: '🚨' },
};
const DEFAULT_CFG   = { icon: <AlertTriangle size={14}/>, label: 'ALERT', emoji: '⚠️' };

function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }
  catch { return iso || '—'; }
}

/* ════════════════════════════════════════════════════════════════
   EVIDENCE MODAL — shown when user clicks an alert
════════════════════════════════════════════════════════════════ */
function EvidenceModal({ alert, apiBase, onClose }) {
  const [imgErr, setImgErr] = useState(false);
  const [imgZoom, setImgZoom] = useState(false);

  const sevColor = SEV_COLOR[alert.severity] || '#94a3b8';
  const cfg      = ALERT_CONFIG[alert.type] || DEFAULT_CFG;
  const removed  = alert.items_removed || [];
  const added    = alert.items_added   || [];

  // Thumbnail URL: backend serves data/ as /static/data
  const thumbUrl = alert.thumbnail_path
    ? `${apiBase}/static/data/thumbnails/${alert.alert_id}.jpg`
    : null;

  return (
    /* Backdrop */
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.82)',
        backdropFilter: 'blur(6px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        animation: 'fadeIn 0.15s ease',
      }}
    >
      {/* Panel — stop propagation so clicking inside doesn't close */}
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: 'linear-gradient(145deg, #0d1a2e, #0a1220)',
          border: `1px solid ${sevColor}40`,
          borderTop: `3px solid ${sevColor}`,
          borderRadius: 16,
          width: 620,
          maxWidth: '96vw',
          maxHeight: '90vh',
          overflowY: 'auto',
          boxShadow: `0 24px 80px rgba(0,0,0,0.8), 0 0 30px ${sevColor}20`,
          animation: 'slideUp 0.2s ease',
        }}
      >
        {/* ── Header ── */}
        <div style={{
          padding: '1rem 1.2rem 0.8rem',
          borderBottom: `1px solid rgba(255,255,255,0.08)`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
        }}>
          <div>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5,
            }}>
              <span style={{
                display: 'flex', alignItems: 'center', gap: 5,
                fontSize: '0.72rem', fontWeight: 800,
                color: sevColor, textTransform: 'uppercase', letterSpacing: '0.06em',
                background: `${sevColor}15`, padding: '0.22rem 0.6rem', borderRadius: 5,
              }}>
                {cfg.icon} {cfg.label}
              </span>
              <span style={{
                fontSize: '0.68rem', padding: '0.2rem 0.5rem',
                background: 'rgba(255,255,255,0.06)', borderRadius: 4,
                color: 'var(--text-dim)', fontFamily: 'monospace',
              }}>
                {alert.camera_id}
              </span>
            </div>
            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>
              {alert.global_id}
              {alert.session_index && (
                <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontWeight: 400, marginLeft: 8 }}>
                  · visit #{alert.session_index}
                </span>
              )}
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
            color: 'var(--text-dim)', borderRadius: 8, width: 32, height: 32,
            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}>
            <X size={16} />
          </button>
        </div>

        {/* ── Image Proof ── */}
        <div style={{ padding: '1rem 1.2rem 0' }}>
          {thumbUrl && !imgErr ? (
            <div style={{ position: 'relative' }}>
              <img
                src={thumbUrl}
                alt="Evidence snapshot"
                onError={() => setImgErr(true)}
                onClick={() => setImgZoom(true)}
                style={{
                  width: '100%',
                  maxHeight: imgZoom ? '70vh' : 240,
                  objectFit: 'cover',
                  borderRadius: 10,
                  border: `1px solid ${sevColor}30`,
                  cursor: 'zoom-in',
                  display: 'block',
                  transition: 'max-height 0.3s ease',
                }}
              />
              {/* Snapshot label overlay */}
              <div style={{
                position: 'absolute', top: 10, left: 10,
                background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)',
                border: `1px solid ${sevColor}40`,
                borderRadius: 6, padding: '3px 8px',
                fontSize: '0.65rem', fontWeight: 700, color: sevColor,
                textTransform: 'uppercase', letterSpacing: '0.05em',
                display: 'flex', alignItems: 'center', gap: 4,
              }}>
                <Camera size={10}/> Evidence Snapshot
              </div>
              <div style={{
                position: 'absolute', top: 10, right: 10,
                background: 'rgba(0,0,0,0.6)',
                borderRadius: 5, padding: '3px 6px',
                fontSize: '0.6rem', color: 'var(--text-dim)',
                display: 'flex', alignItems: 'center', gap: 3,
              }}>
                <ZoomIn size={10}/> {imgZoom ? 'click again to shrink' : 'click to expand'}
              </div>
            </div>
          ) : (
            <div style={{
              background: 'rgba(255,255,255,0.03)',
              border: `1px dashed rgba(255,255,255,0.12)`,
              borderRadius: 10, padding: '1.5rem',
              textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.8rem',
            }}>
              <Camera size={22} style={{ opacity: 0.3, marginBottom: 6 }}/>
              <div>No snapshot available for this event</div>
              <div style={{ fontSize: '0.68rem', marginTop: 4 }}>Snapshots are saved when the alert first fires</div>
            </div>
          )}
        </div>

        {/* ── What Happened ── */}
        <div style={{ padding: '0.9rem 1.2rem 0' }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 8 }}>
            What Happened
          </div>
          <div style={{
            background: `${sevColor}08`,
            border: `1px solid ${sevColor}20`,
            borderRadius: 10, padding: '0.75rem 1rem',
            fontSize: '0.82rem', lineHeight: 1.6, color: 'var(--text-main)',
          }}>
            {alert.message}
          </div>
        </div>

        {/* ── Item change diff (baggage alerts) ── */}
        {(removed.length > 0 || added.length > 0) && (
          <div style={{ padding: '0.9rem 1.2rem 0' }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 8 }}>
              Item Change
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: removed.length > 0 && added.length > 0 ? '1fr 1fr' : '1fr', gap: 8 }}>
              {removed.length > 0 && (
                <div style={{
                  background: 'rgba(239,68,68,0.10)',
                  border: '1px solid rgba(239,68,68,0.25)',
                  borderRadius: 8, padding: '0.65rem 0.9rem',
                }}>
                  <div style={{ fontSize: '0.66rem', color: '#ef4444', fontWeight: 700, marginBottom: 5, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <PackageMinus size={11}/> LEFT BEHIND / DROPPED
                  </div>
                  {removed.map(item => (
                    <div key={item} style={{ fontSize: '0.8rem', color: '#fca5a5', fontWeight: 600, textTransform: 'capitalize' }}>
                      • {item}
                    </div>
                  ))}
                </div>
              )}
              {added.length > 0 && (
                <div style={{
                  background: 'rgba(249,115,22,0.10)',
                  border: '1px solid rgba(249,115,22,0.25)',
                  borderRadius: 8, padding: '0.65rem 0.9rem',
                }}>
                  <div style={{ fontSize: '0.66rem', color: '#fb923c', fontWeight: 700, marginBottom: 5, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <PackageCheck size={11}/> TAKEN / ACQUIRED
                  </div>
                  {added.map(item => (
                    <div key={item} style={{ fontSize: '0.8rem', color: '#fdba74', fontWeight: 600, textTransform: 'capitalize' }}>
                      • {item}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Timeline ── */}
        <div style={{ padding: '0.9rem 1.2rem' }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 8 }}>
            Timeline
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {alert.entry_time && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#22c55e', flexShrink: 0 }}/>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Entry</span>
                <span style={{ fontSize: '0.78rem', fontFamily: 'monospace', color: 'var(--text-main)', fontWeight: 600 }}>{fmtTime(alert.entry_time)}</span>
              </div>
            )}
            {alert.entry_time && alert.exit_time && (
              <div style={{ width: 1, height: 14, background: 'rgba(255,255,255,0.1)', marginLeft: 3.5 }}/>
            )}
            {alert.exit_time && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: sevColor, flexShrink: 0 }}/>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Exit / Alert</span>
                <span style={{ fontSize: '0.78rem', fontFamily: 'monospace', color: 'var(--text-main)', fontWeight: 600 }}>{fmtTime(alert.exit_time)}</span>
              </div>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 2 }}>
              <Clock size={8} style={{ color: 'var(--text-dim)', marginLeft: 0 }}/>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Alert fired</span>
              <span style={{ fontSize: '0.78rem', fontFamily: 'monospace', color: 'var(--text-main)', fontWeight: 600 }}>{fmtTime(alert.timestamp)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   COMPACT ALERT CARD (in the right panel list)
════════════════════════════════════════════════════════════════ */
function AlertCard({ alert, apiBase, onAcknowledge }) {
  const [showModal, setShowModal] = useState(false);
  const sevColor = SEV_COLOR[alert.severity] || '#94a3b8';
  const cfg = ALERT_CONFIG[alert.type] || DEFAULT_CFG;
  const hasThumb = !!alert.thumbnail_path;

  return (
    <>
      <div
        onClick={() => setShowModal(true)}
        style={{
          background: 'rgba(15,23,42,0.6)',
          border: `1px solid rgba(255,255,255,0.08)`,
          borderLeft: `4px solid ${sevColor}`,
          borderRadius: 10,
          padding: '0.75rem 0.85rem',
          cursor: 'pointer',
          animation: 'slideIn 0.3s ease-out',
          transition: 'background 0.15s, box-shadow 0.15s',
          position: 'relative',
        }}
        onMouseEnter={e => {
          e.currentTarget.style.background = `${sevColor}0d`;
          e.currentTarget.style.boxShadow = `0 2px 16px ${sevColor}20`;
        }}
        onMouseLeave={e => {
          e.currentTarget.style.background = 'rgba(15,23,42,0.6)';
          e.currentTarget.style.boxShadow = 'none';
        }}
      >
        {/* Top row: type badge + time */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
          <span style={{
            display: 'flex', alignItems: 'center', gap: 4,
            fontSize: '0.68rem', fontWeight: 700, color: sevColor,
            textTransform: 'uppercase', letterSpacing: '0.04em',
            background: `${sevColor}15`, padding: '0.18rem 0.5rem', borderRadius: 4,
          }}>
            {cfg.icon} {cfg.label}
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: '0.65rem', color: 'var(--text-dim)' }}>
            <Clock size={10}/> {fmtTime(alert.timestamp)}
          </span>
        </div>

        {/* Message */}
        <p style={{ fontSize: '0.78rem', lineHeight: 1.45, color: 'var(--text-main)', marginBottom: '0.5rem' }}>
          {alert.message}
        </p>

        {/* Footer: thumbnail hint + acknowledge */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.62rem', color: sevColor, display: 'flex', alignItems: 'center', gap: 3, opacity: hasThumb ? 1 : 0.4 }}>
            <Camera size={10}/> {hasThumb ? 'Image proof available — click to view' : 'No snapshot'}
          </span>
          <button
            onClick={e => { e.stopPropagation(); onAcknowledge(alert.alert_id); }}
            style={{
              background: 'transparent', border: `1px solid rgba(255,255,255,0.12)`,
              color: 'var(--text-dim)', fontSize: '0.65rem',
              padding: '0.18rem 0.55rem', borderRadius: 4, cursor: 'pointer', fontFamily: 'inherit',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.06)'; e.currentTarget.style.color = 'var(--text-main)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-dim)'; }}
          >
            Acknowledge
          </button>
        </div>
      </div>

      {showModal && (
        <EvidenceModal alert={alert} apiBase={apiBase} onClose={() => setShowModal(false)} />
      )}
    </>
  );
}

/* ════════════════════════════════════════════════════════════════
   MAIN AlertsFeed
════════════════════════════════════════════════════════════════ */
export default function AlertsFeed({ alerts, setAlerts, apiBase }) {
  const handleAcknowledge = async (alertId) => {
    try {
      await fetch(`${apiBase}/alerts/${alertId}/acknowledge`, { method: 'POST' });
      setAlerts(prev => prev.filter(a => a.alert_id !== alertId));
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
      {alerts.length === 0 ? (
        <div style={{ textAlign: 'center', color: 'var(--text-dim)', marginTop: '4rem' }}>
          <Package size={28} style={{ opacity: 0.2, marginBottom: 8 }} />
          <p>No alerts detected recently.</p>
          <p style={{ fontSize: '0.72rem', marginTop: 4 }}>Click any alert card to view evidence</p>
        </div>
      ) : (
        alerts.map(alert => (
          <AlertCard
            key={alert.alert_id}
            alert={alert}
            apiBase={apiBase}
            onAcknowledge={handleAcknowledge}
          />
        ))
      )}
    </div>
  );
}
