import React from 'react';
import { AlertTriangle, Clock, Package, PackageCheck, PackageMinus, ArrowRightLeft } from 'lucide-react';

/* ── Baggage-change alert card (special rendering) ── */
function BaggageAlertCard({ alert, onAcknowledge }) {
  const isCritical = alert.severity === 'critical';
  const borderColor = isCritical ? 'var(--critical)' : 'var(--high)';
  const bgColor     = isCritical ? 'rgba(239,68,68,0.08)' : 'rgba(249,115,22,0.08)';

  const typeConfig = {
    baggage_swap: {
      icon: <ArrowRightLeft size={13} />,
      label: 'BAGGAGE SWAP',
      color: 'var(--critical)',
    },
    baggage_taken: {
      icon: <PackageCheck size={13} />,
      label: 'ITEM TAKEN',
      color: 'var(--critical)',
    },
    baggage_left_behind: {
      icon: <PackageMinus size={13} />,
      label: 'ITEM LEFT',
      color: 'var(--high)',
    },
  };

  const tc = typeConfig[alert.type] || { icon: <Package size={13} />, label: alert.type.toUpperCase(), color: borderColor };

  const removed = alert.items_removed || [];
  const added   = alert.items_added   || [];

  const fmtTime = (iso) => {
    try { return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }
    catch { return iso || '—'; }
  };

  return (
    <div style={{
      background: bgColor,
      border: `1px solid ${borderColor}`,
      borderLeft: `4px solid ${borderColor}`,
      borderRadius: 12,
      padding: '0.85rem',
      animation: 'slideIn 0.3s ease-out',
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.55rem' }}>
        <span style={{
          display: 'flex', alignItems: 'center', gap: 5,
          fontSize: '0.72rem', fontWeight: 700,
          color: tc.color, textTransform: 'uppercase', letterSpacing: '0.04em',
          background: `${tc.color}18`, padding: '0.2rem 0.55rem', borderRadius: 5,
        }}>
          {tc.icon} {tc.label}
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.68rem', color: 'var(--text-dim)' }}>
          <Clock size={11} /> {fmtTime(alert.timestamp)}
        </span>
      </div>

      {/* Identity */}
      <div style={{ fontSize: '0.78rem', fontWeight: 600, marginBottom: '0.45rem', color: 'var(--text-main)' }}>
        {alert.global_id}
        {alert.session_index && (
          <span style={{ marginLeft: 6, fontSize: '0.68rem', color: 'var(--text-dim)', fontWeight: 400 }}>
            · visit #{alert.session_index}
          </span>
        )}
        <span style={{ marginLeft: 6, fontSize: '0.68rem', color: 'var(--text-dim)', fontWeight: 400 }}>
          · {alert.camera_id}
        </span>
      </div>

      {/* Timing */}
      {(alert.entry_time || alert.exit_time) && (
        <div style={{ display: 'flex', gap: 12, fontSize: '0.68rem', color: 'var(--text-dim)', marginBottom: '0.5rem' }}>
          {alert.entry_time && <span>🚪 In: <b style={{ color: 'var(--text-main)' }}>{fmtTime(alert.entry_time)}</b></span>}
          {alert.exit_time  && <span>🚶 Out: <b style={{ color: 'var(--text-main)' }}>{fmtTime(alert.exit_time)}</b></span>}
        </div>
      )}

      {/* Item diff */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: '0.55rem' }}>
        {removed.length > 0 && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: 'rgba(239,68,68,0.12)', borderRadius: 6, padding: '0.25rem 0.6rem',
          }}>
            <PackageMinus size={12} color="#ef4444" />
            <span style={{ fontSize: '0.72rem', color: '#fca5a5' }}>
              Left behind: <b>{removed.join(', ')}</b>
            </span>
          </div>
        )}
        {added.length > 0 && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: 'rgba(249,115,22,0.12)', borderRadius: 6, padding: '0.25rem 0.6rem',
          }}>
            <PackageCheck size={12} color="#fb923c" />
            <span style={{ fontSize: '0.72rem', color: '#fdba74' }}>
              Took with them: <b>{added.join(', ')}</b>
            </span>
          </div>
        )}
      </div>

      {/* Acknowledge */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button
          onClick={() => onAcknowledge(alert.alert_id)}
          style={{
            background: 'transparent', border: `1px solid ${borderColor}40`,
            color: 'var(--text-dim)', fontSize: '0.7rem',
            padding: '0.2rem 0.65rem', borderRadius: 4, cursor: 'pointer', fontFamily: 'inherit',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = `${borderColor}18`; e.currentTarget.style.color = 'var(--text-main)'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-dim)'; }}
        >
          Acknowledge
        </button>
      </div>
    </div>
  );
}

/* ── BAGGAGE ALERT TYPES ── */
const BAGGAGE_TYPES = new Set(['baggage_left_behind', 'baggage_taken', 'baggage_swap']);

/* ── Main AlertsFeed ── */
export default function AlertsFeed({ alerts, setAlerts, apiBase }) {

  const handleAcknowledge = async (alertId) => {
    try {
      await fetch(`${apiBase}/alerts/${alertId}/acknowledge`, { method: 'POST' });
      setAlerts(prev => prev.filter(a => a.alert_id !== alertId));
    } catch (e) {
      console.error(e);
    }
  };

  const getSeverityColor = (sev) => {
    switch(sev) {
      case 'critical': return 'var(--critical)';
      case 'high':     return 'var(--high)';
      case 'medium':   return 'var(--medium)';
      case 'low':      return 'var(--low)';
      default:         return 'var(--text-dim)';
    }
  };

  const getAlertBg = (sev) => {
    switch(sev) {
      case 'critical': return 'rgba(239, 68, 68, 0.1)';
      case 'high':     return 'rgba(249, 115, 22, 0.1)';
      default:         return 'rgba(255, 255, 255, 0.05)';
    }
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {alerts.length === 0 ? (
        <div style={{ textAlign: 'center', color: 'var(--text-dim)', marginTop: '4rem' }}>
          <Package size={28} style={{ opacity: 0.2, marginBottom: 8 }} />
          <p>No alerts detected recently.</p>
        </div>
      ) : (
        alerts.map(alert => (
          /* Render baggage alerts with the special card */
          BAGGAGE_TYPES.has(alert.type) ? (
            <BaggageAlertCard
              key={alert.alert_id}
              alert={alert}
              onAcknowledge={handleAcknowledge}
            />
          ) : (
            /* Standard alert card for all other types */
            <div key={alert.alert_id} style={{
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid var(--border-color)',
              borderLeft: `4px solid ${getSeverityColor(alert.severity)}`,
              borderRadius: 12,
              padding: '1rem',
              animation: 'slideIn 0.3s ease-out',
              position: 'relative',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>
                <span style={{
                  fontWeight: 600, padding: '0.25rem 0.5rem', borderRadius: 4,
                  backgroundColor: getAlertBg(alert.severity),
                  color: getSeverityColor(alert.severity),
                }}>
                  <AlertTriangle size={12} style={{ marginRight: 4, verticalAlign: 'text-top' }} />
                  {alert.type}
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <Clock size={12}/>{new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>

              <p style={{ fontSize: '0.9rem', lineHeight: 1.4, marginBottom: '0.75rem' }}>{alert.message}</p>

              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  onClick={() => handleAcknowledge(alert.alert_id)}
                  style={{
                    background: 'transparent', border: '1px solid var(--border-color)',
                    color: 'var(--text-dim)', fontSize: '0.75rem',
                    padding: '0.25rem 0.75rem', borderRadius: 4, cursor: 'pointer',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background='rgba(255,255,255,0.05)'; e.currentTarget.style.color='var(--text-main)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background='transparent'; e.currentTarget.style.color='var(--text-dim)'; }}
                >
                  Acknowledge
                </button>
              </div>
            </div>
          )
        ))
      )}
    </div>
  );
}
