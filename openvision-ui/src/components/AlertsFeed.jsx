import React from 'react';
import { AlertTriangle, Clock } from 'lucide-react';

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
      case 'high': return 'var(--high)';
      case 'medium': return 'var(--medium)';
      case 'low': return 'var(--low)';
      default: return 'var(--text-dim)';
    }
  };

  const getAlertBg = (sev) => {
    switch(sev) {
      case 'critical': return 'rgba(239, 68, 68, 0.1)';
      case 'high': return 'rgba(249, 115, 22, 0.1)';
      default: return 'rgba(255, 255, 255, 0.05)';
    }
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {alerts.length === 0 ? (
        <div style={{ textAlign: 'center', color: 'var(--text-dim)', marginTop: '4rem' }}>
          <p>No alerts detected recently.</p>
        </div>
      ) : (
        alerts.map(alert => (
          <div key={alert.alert_id} style={{
            background: 'rgba(15, 23, 42, 0.6)',
            border: '1px solid var(--border-color)',
            borderLeft: `4px solid ${getSeverityColor(alert.severity)}`,
            borderRadius: 12,
            padding: '1rem',
            animation: 'slideIn 0.3s ease-out',
            position: 'relative'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>
              <span style={{ 
                fontWeight: 600, padding: '0.25rem 0.5rem', borderRadius: 4, 
                backgroundColor: getAlertBg(alert.severity),
                color: getSeverityColor(alert.severity)
              }}>
                <AlertTriangle size={12} style={{ marginRight: 4, verticalAlign: 'text-top' }} />
                {alert.type}
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Clock size={12}/>{new Date(alert.timestamp).toLocaleTimeString()}</span>
            </div>
            
            <p style={{ fontSize: '0.9rem', lineHeight: 1.4, marginBottom: '0.75rem' }}>{alert.message}</p>
            
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button 
                onClick={() => handleAcknowledge(alert.alert_id)}
                style={{ background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-dim)', fontSize: '0.75rem', padding: '0.25rem 0.75rem', borderRadius: 4, cursor: 'pointer' }}
                onMouseEnter={e => { e.currentTarget.style.background='rgba(255,255,255,0.05)'; e.currentTarget.style.color='var(--text-main)'; }}
                onMouseLeave={e => { e.currentTarget.style.background='transparent'; e.currentTarget.style.color='var(--text-dim)'; }}
              >
                Acknowledge
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
