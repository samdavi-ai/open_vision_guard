import React, { useState, useRef, useEffect, useCallback } from 'react';
import { X, MessageSquare, Send, Zap, BarChart2, User, FileText, Loader, Bot, RefreshCw, AlertCircle } from 'lucide-react';

const QUICK = [
  { label: 'Suspicious activity?',  q: 'Any suspicious activity in the last hour?' },
  { label: 'Persons with bags',     q: 'Who entered with a bag and left without one?' },
  { label: 'High risk persons',     q: 'Who are the highest risk persons detected today?' },
  { label: 'Loitering incidents',   q: 'Show me all loitering alerts from today' },
  { label: 'CAM_01 summary',        q: 'What happened at CAM_01 in the last 30 minutes?' },
  { label: 'Baggage theft?',        q: 'Were there any baggage theft or swap incidents today?' },
];

/* ── Dot-animation pulse keyframes (injected once) ── */
if (typeof document !== 'undefined' && !document.getElementById('llm-chat-styles')) {
  const s = document.createElement('style');
  s.id = 'llm-chat-styles';
  s.textContent = `
    @keyframes llm-pulse { 0%,80%,100% { opacity:.2; transform:scale(.8) } 40% { opacity:1; transform:scale(1) } }
    @keyframes llm-slide-up { from { opacity:0; transform:translateY(24px) } to { opacity:1; transform:translateY(0) } }
    @keyframes llm-spin { to { transform:rotate(360deg) } }
    @keyframes llm-fade-in { from { opacity:0 } to { opacity:1 } }
    .llm-spin { animation: llm-spin 1s linear infinite; }
  `;
  document.head.appendChild(s);
}

/* ── Single message bubble ── */
function Message({ msg }) {
  const isUser   = msg.role === 'user';
  const isSystem = msg.role === 'system';

  if (isSystem) return (
    <div style={{ textAlign: 'center', color: 'rgba(255,255,255,0.35)', fontSize: '0.68rem', padding: '0.25rem 0' }}>
      {msg.content}
    </div>
  );

  return (
    <div style={{
      display: 'flex', gap: 8, alignItems: 'flex-start',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      animation: 'llm-fade-in 0.18s ease',
    }}>
      {!isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
          background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 2px 8px rgba(59,130,246,0.4)',
        }}>
          <Bot size={14} color="#fff" />
        </div>
      )}
      <div style={{
        maxWidth: '80%',
        background: isUser
          ? 'linear-gradient(135deg, #3b82f6, #6366f1)'
          : 'rgba(255,255,255,0.07)',
        border: isUser ? 'none' : '1px solid rgba(255,255,255,0.1)',
        color: '#e2e8f0',
        padding: '0.55rem 0.85rem',
        borderRadius: isUser ? '14px 14px 4px 14px' : '4px 14px 14px 14px',
        fontSize: '0.82rem',
        lineHeight: 1.6,
        whiteSpace: 'pre-wrap',
        boxShadow: isUser ? '0 2px 12px rgba(59,130,246,0.3)' : 'none',
      }}>
        {msg.content}
        {msg.loading && (
          <span style={{ display: 'inline-flex', gap: 3, marginLeft: 6, verticalAlign: 'middle' }}>
            {[0, 1, 2].map(i => (
              <span key={i} style={{
                width: 5, height: 5, borderRadius: '50%',
                background: 'rgba(255,255,255,0.5)',
                display: 'inline-block',
                animation: `llm-pulse 1.2s ${i * 0.2}s infinite`,
              }} />
            ))}
          </span>
        )}
      </div>
    </div>
  );
}

/* ── Result card (for correlate / report tabs) ── */
function ResultCard({ text }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 10, padding: '0.9rem',
      fontSize: '0.82rem', lineHeight: 1.65,
      color: '#e2e8f0', whiteSpace: 'pre-wrap',
      animation: 'llm-fade-in 0.25s ease',
    }}>
      {text}
    </div>
  );
}

/* ── Main component ── */
export default function LLMChat({ apiBase, onClose }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "👋 Hi! I'm your AI security analyst powered by Llama 4 Scout.\n\nAsk me anything about the surveillance feed — suspicious activity, persons of interest, baggage incidents, or request a shift report." }
  ]);
  const [input, setInput]               = useState('');
  const [loading, setLoading]           = useState(false);
  const [tab, setTab]                   = useState('chat');
  const [correlateResult, setCorrelate] = useState('');
  const [reportResult, setReport]       = useState('');
  const [status, setStatus]             = useState(null);   // { available, model_fast, model_smart, db_online }
  const [reinitiating, setReinit]       = useState(false);
  const bottomRef = useRef(null);

  /* Fetch LLM status */
  const fetchStatus = useCallback(() => {
    fetch(`${apiBase}/llm/status`)
      .then(r => r.json())
      .then(d => setStatus(d))
      .catch(() => setStatus({ available: false, db_online: false }));
  }, [apiBase]);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  /* Auto-scroll */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  /* Hot-reinit */
  const handleReinit = async () => {
    setReinit(true);
    try {
      const r = await fetch(`${apiBase}/llm/reinit`, { method: 'POST' });
      const d = await r.json();
      setStatus(prev => ({ ...prev, ...d }));
      if (d.available) {
        setMessages(p => [...p, { role: 'system', content: '✅ LLM reconnected successfully.' }]);
      } else {
        setMessages(p => [...p, { role: 'system', content: '❌ Reconnect failed — check GROQ_API_KEY in .env' }]);
      }
    } catch {
      setMessages(p => [...p, { role: 'system', content: '❌ Failed to reach backend.' }]);
    }
    setReinit(false);
  };

  /* Send chat message */
  const send = async (question) => {
    const q = (question ?? input).trim();
    if (!q || loading) return;
    setInput('');
    setMessages(p => [...p, { role: 'user', content: q }]);
    setLoading(true);

    const loadId = Date.now();
    setMessages(p => [...p, { role: 'assistant', content: '', loading: true, id: loadId }]);

    try {
      const r = await fetch(`${apiBase}/llm/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const d = await r.json();
      setMessages(p => p.map(m => m.id === loadId
        ? { role: 'assistant', content: d.answer || 'No answer available.' }
        : m
      ));
    } catch {
      setMessages(p => p.map(m => m.id === loadId
        ? { role: 'assistant', content: '⚠ Connection error — is the backend running?' }
        : m
      ));
    }
    setLoading(false);
  };

  /* Correlate */
  const runCorrelate = async () => {
    setCorrelate('');
    setLoading(true);
    try {
      const r = await fetch(`${apiBase}/llm/correlate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hours: 1 }),
      });
      const d = await r.json();
      setCorrelate(d.analysis || 'No patterns detected.');
    } catch {
      setCorrelate('Failed to run correlation.');
    }
    setLoading(false);
  };

  /* Shift report */
  const runReport = async () => {
    setReport('');
    setLoading(true);
    try {
      const r = await fetch(`${apiBase}/llm/shift-report?hours=8`);
      const d = await r.json();
      setReport(d.report || 'No report generated.');
    } catch {
      setReport('Failed to generate report.');
    }
    setLoading(false);
  };

  const TABS = [
    { id: 'chat',      icon: <MessageSquare size={13}/>, label: 'Chat' },
    { id: 'correlate', icon: <BarChart2 size={13}/>,     label: 'Correlate' },
    { id: 'report',    icon: <FileText size={13}/>,      label: 'Shift Report' },
  ];

  const llmOk  = status?.available === true;
  const llmBad = status?.available === false;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 3000,
        background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'flex-end', justifyContent: 'flex-end',
        padding: '70px 12px 12px 12px',
        animation: 'llm-fade-in 0.15s ease',
      }}
    >
      <div onClick={e => e.stopPropagation()} style={{
        width: 460, maxWidth: '98vw',
        height: 640, maxHeight: '88vh',
        background: 'linear-gradient(160deg, #0d1a2e 0%, #080f1e 100%)',
        border: '1px solid rgba(59,130,246,0.28)',
        borderRadius: 18,
        boxShadow: '0 28px 80px rgba(0,0,0,0.75), 0 0 60px rgba(59,130,246,0.08)',
        display: 'flex', flexDirection: 'column',
        animation: 'llm-slide-up 0.22s ease',
        overflow: 'hidden',
      }}>

        {/* ── Header ── */}
        <div style={{
          padding: '0.85rem 1rem 0.7rem',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          flexShrink: 0,
          background: 'rgba(59,130,246,0.07)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 34, height: 34, borderRadius: 10,
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 4px 14px rgba(59,130,246,0.45)',
            }}>
              <Bot size={17} color="#fff"/>
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#e2e8f0' }}>AI Security Analyst</div>
              <div style={{ fontSize: '0.62rem', color: 'rgba(255,255,255,0.45)', display: 'flex', alignItems: 'center', gap: 5 }}>
                <div style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: status === null ? '#f59e0b' : llmOk ? '#22c55e' : '#ef4444',
                  boxShadow: llmOk ? '0 0 6px #22c55e' : undefined,
                  flexShrink: 0,
                }}/>
                {status === null
                  ? 'Connecting…'
                  : llmOk
                    ? `${status.model_fast?.includes('llama-4') ? 'Llama 4 Scout' : 'Llama 3.1 8B'} · Connected`
                    : 'LLM Offline — check GROQ_API_KEY'}
                {!status?.db_online && status !== null && (
                  <span style={{ color: '#f59e0b', marginLeft: 4 }}>· DB offline</span>
                )}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 6 }}>
            {/* Reinit button */}
            {llmBad && (
              <button
                onClick={handleReinit}
                disabled={reinitiating}
                title="Retry LLM connection"
                style={{
                  background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)',
                  color: '#fca5a5', borderRadius: 8, width: 30, height: 30,
                  cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: 'background 0.15s',
                }}
              >
                <RefreshCw size={13} className={reinitiating ? 'llm-spin' : ''}/>
              </button>
            )}
            <button onClick={onClose} style={{
              background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
              color: 'rgba(255,255,255,0.5)', borderRadius: 8, width: 30, height: 30,
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <X size={14}/>
            </button>
          </div>
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.07)', flexShrink: 0 }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              flex: 1, padding: '0.5rem', background: 'transparent', border: 'none',
              borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
              color: tab === t.id ? '#60a5fa' : 'rgba(255,255,255,0.35)',
              fontSize: '0.72rem', fontWeight: 600, cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
              transition: 'color 0.15s',
            }}>
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        {/* ── CHAT TAB ── */}
        {tab === 'chat' && (
          <>
            <div style={{
              flex: 1, overflowY: 'auto', padding: '0.9rem',
              display: 'flex', flexDirection: 'column', gap: 10,
            }}>
              {messages.map((m, i) => <Message key={i} msg={m}/>)}
              <div ref={bottomRef}/>
            </div>

            {/* Quick actions */}
            <div style={{ padding: '0 0.8rem 0.55rem', display: 'flex', gap: 5, flexWrap: 'wrap', flexShrink: 0 }}>
              {QUICK.map(q => (
                <button key={q.q} onClick={() => send(q.q)} disabled={loading} style={{
                  background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.25)',
                  color: '#93c5fd', fontSize: '0.64rem', padding: '0.22rem 0.6rem',
                  borderRadius: 999, cursor: 'pointer', fontFamily: 'inherit',
                  transition: 'background 0.15s',
                  opacity: loading ? 0.45 : 1,
                }}>
                  {q.label}
                </button>
              ))}
            </div>

            {/* Input bar */}
            <div style={{
              padding: '0.65rem 0.8rem 0.8rem', display: 'flex', gap: 8, flexShrink: 0,
              borderTop: '1px solid rgba(255,255,255,0.06)',
            }}>
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
                placeholder="Ask about activity, persons, incidents…"
                disabled={loading}
                style={{
                  flex: 1, background: 'rgba(255,255,255,0.07)',
                  border: '1px solid rgba(255,255,255,0.13)',
                  borderRadius: 10, color: '#e2e8f0',
                  padding: '0.55rem 0.85rem', fontSize: '0.82rem', outline: 'none',
                  fontFamily: 'inherit',
                  transition: 'border-color 0.15s',
                }}
              />
              <button onClick={() => send()} disabled={!input.trim() || loading} style={{
                background: 'linear-gradient(135deg, #3b82f6, #6366f1)',
                border: 'none', color: '#fff',
                borderRadius: 10, width: 40, flexShrink: 0, cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                opacity: (!input.trim() || loading) ? 0.4 : 1,
                transition: 'opacity 0.15s',
                boxShadow: '0 4px 12px rgba(59,130,246,0.35)',
              }}>
                {loading
                  ? <Loader size={14} className="llm-spin"/>
                  : <Send size={14}/>}
              </button>
            </div>
          </>
        )}

        {/* ── CORRELATE TAB ── */}
        {tab === 'correlate' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ fontSize: '0.78rem', color: 'rgba(255,255,255,0.45)', lineHeight: 1.55 }}>
              Analyzes all alerts from the last hour using <b style={{ color: '#e2e8f0' }}>Llama 3.3 70B</b> to detect coordinated activity,
              shoplifting teams, or suspicious patterns across cameras.
            </div>
            <button onClick={runCorrelate} disabled={loading} style={{
              background: 'linear-gradient(135deg, #3b82f6, #6366f1)',
              border: 'none', color: '#fff', padding: '0.65rem', borderRadius: 9,
              cursor: 'pointer', fontWeight: 700, fontSize: '0.82rem', fontFamily: 'inherit',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              opacity: loading ? 0.7 : 1,
              boxShadow: '0 4px 14px rgba(59,130,246,0.35)',
            }}>
              {loading ? <Loader size={14} className="llm-spin"/> : <Zap size={14}/>}
              {loading ? 'Analyzing patterns…' : 'Run Correlation Analysis'}
            </button>
            {correlateResult && <ResultCard text={correlateResult}/>}
          </div>
        )}

        {/* ── REPORT TAB ── */}
        {tab === 'report' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ fontSize: '0.78rem', color: 'rgba(255,255,255,0.45)', lineHeight: 1.55 }}>
              Generate a professional <b style={{ color: '#e2e8f0' }}>end-of-shift security report</b> summarizing the last 8 hours of surveillance data.
              Uses <b style={{ color: '#e2e8f0' }}>Llama 3.3 70B</b> for deep analysis.
            </div>
            <button onClick={runReport} disabled={loading} style={{
              background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
              border: 'none', color: '#fff', padding: '0.65rem', borderRadius: 9,
              cursor: 'pointer', fontWeight: 700, fontSize: '0.82rem', fontFamily: 'inherit',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              opacity: loading ? 0.7 : 1,
              boxShadow: '0 4px 14px rgba(139,92,246,0.35)',
            }}>
              {loading ? <Loader size={14} className="llm-spin"/> : <FileText size={14}/>}
              {loading ? 'Generating report…' : 'Generate 8-Hour Shift Report'}
            </button>
            {reportResult && <ResultCard text={reportResult}/>}
          </div>
        )}
      </div>
    </div>
  );
}
