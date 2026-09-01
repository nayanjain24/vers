import React, { useState, useEffect, useRef } from 'react';
import { 
  Shield, 
  Activity, 
  Video, 
  VideoOff, 
  AlertTriangle, 
  Cpu, 
  Volume2, 
  VolumeX, 
  Sparkles, 
  Flame, 
  Ambulance, 
  CheckCircle2, 
  Radio, 
  HelpCircle,
  Siren,
  BookOpen,
  X,
  Search,
  MessageSquare,
  Hand
} from 'lucide-react';
import './index.css';

const NORMAL_SIGNS = [
  { word: "HELLO", meaning: "Friendly greeting", gesture: "Open hand waving side to side near temple", category: "Conversational" },
  { word: "THANK_YOU", meaning: "Expressing gratitude", gesture: "Flat hand moving from chin outward towards person", category: "Conversational" },
  { word: "PLEASE", meaning: "Polite request modifier", gesture: "Flat palm held against chest in circular motion", category: "Conversational" },
  { word: "YES", meaning: "Affirmative response / agreement", gesture: "Closed fist (ASL 'S') nodding up and down", category: "Conversational" },
  { word: "NO", meaning: "Negative response / denial", gesture: "Index + Middle finger snapping against thumb", category: "Conversational" },
  { word: "WATER", meaning: "Requesting water or drink", gesture: "W-handshape (3 middle fingers up) tapping against chin", category: "Needs" },
  { word: "FOOD", meaning: "Requesting food or meal", gesture: "Flat-O handshape (fingertips touching thumb) tapping lips", category: "Needs" },
  { word: "WANT", meaning: "Expressing a want or need", gesture: "Both hands open clawed palms up, pulling inward", category: "Needs" },
  { word: "MORE", meaning: "Requesting additional item", gesture: "Both hands forming flat-O, tapping fingertips together", category: "Needs" },
  { word: "FRIEND", meaning: "Referring to a friend or ally", gesture: "Index fingers hooked into each other and reversed", category: "Social" },
  { word: "FAMILY", meaning: "Referring to family or relatives", gesture: "F-handshapes touching at thumbs and circling outward", category: "Social" },
  { word: "NAME", meaning: "Asking for or giving name", gesture: "H-handshapes (index+middle) tapping across each other", category: "Social" },
  { word: "GOOD", meaning: "Positive status or approval", gesture: "Fingers of dominant hand on chin moving down to flat palm", category: "Feedback" },
  { word: "BAD", meaning: "Negative status or disapproval", gesture: "Fingers on chin moving down while flipping palm downwards", category: "Feedback" },
  { word: "SORRY", meaning: "Apology or regret", gesture: "Closed A-fist rubbing in a circle over the heart/chest", category: "Feedback" },
  { word: "UNDERSTAND", meaning: "Confirming comprehension", gesture: "Index finger flicking up like a lightbulb near forehead", category: "Conversational" },
  { word: "PHONE", meaning: "Requesting a call or mobile device", gesture: "Y-handshape (thumb and pinky out) held up to the ear", category: "Needs" },
  { word: "WHERE", meaning: "Location inquiry", gesture: "Index finger held upright wagging side-to-side", category: "Conversational" },
  { word: "FINISHED", meaning: "Task completed or done", gesture: "Open hands flipped outward away from body", category: "Feedback" },
];

const EMERGENCY_SIGNS_LIST = [
  { word: "HELP", meaning: "Distress / SOS signal", gesture: "Open hand with 5 fingers spread wide or fist resting on flat palm", level: "Critical" },
  { word: "MEDICAL", meaning: "Need ambulance / doctor", gesture: "4 fingers extended and grouped with thumb tucked sharply across palm", level: "Critical" },
  { word: "FIRE", meaning: "Fire hazard / Smoke alarm", gesture: "Index finger pointing straight up, waving/flickering upward motion", level: "High" },
  { word: "POLICE", meaning: "Security threat / Crime", gesture: "Index and middle fingers in V-shape pointing upward", level: "High" },
  { word: "AMBULANCE", meaning: "Extreme medical crisis", gesture: "Index, middle, and ring fingers extended (3 fingers), sweeping", level: "Critical" },
  { word: "ACCIDENT", meaning: "Traffic or physical crash", gesture: "Tight closed fist, all fingers curled tightly covering thumb", level: "Critical" },
  { word: "DANGER", meaning: "Imminent threat", gesture: "Closed fist with wrist rotated heavily on z-axis", level: "High" },
  { word: "PAIN", meaning: "Physical injury / Distress", gesture: "Partially clenched claw shape curved inward over injury site", level: "Medium" },
  { word: "FALL", meaning: "Slip, trip, or collapse", gesture: "Open hand with fingers pointing straight down vertically", level: "High" },
  { word: "STOP", meaning: "Halt ongoing action", gesture: "Flat palm facing forward with fingers held tightly together", level: "Medium" },
  { word: "SAFE", meaning: "All clear / Safe status", gesture: "Thumbs up with tight fist held steady", level: "Low" },
];

export default function App() {
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState(null);
  const [telemetry, setTelemetry] = useState({
    fps: 0,
    gesture: 'NONE',
    confidence: 0,
    distress_score: 0,
    dominant_emotion: 'neutral',
    threat_level: 'NONE',
  });
  const [alerts, setAlerts] = useState([]);
  const [browserCamActive, setBrowserCamActive] = useState(false);
  const [signMode, setSignMode] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [lastTriggered, setLastTriggered] = useState(null);
  const [showGlossary, setShowGlossary] = useState(false);
  const [glossaryTab, setGlossaryTab] = useState('normal'); // 'normal' | 'emergency'
  const [searchQuery, setSearchQuery] = useState('');

  const signModeRef = useRef(signMode);
  useEffect(() => {
    signModeRef.current = signMode;
  }, [signMode]);

  const ws = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [appMode, setAppMode] = useState('emergency'); // 'emergency' | 'conversation'
  const [sentenceWords, setSentenceWords] = useState([]);
  const [conversationHistory, setConversationHistory] = useState([]);
  const lastSpokenWordRef = useRef(null);

  // Sync signMode with appMode
  useEffect(() => {
    setSignMode(appMode === 'conversation');
  }, [appMode]);

  // Handle conversational word detection for sentence building & TTS
  useEffect(() => {
    if (appMode === 'conversation' && telemetry.gesture && telemetry.gesture !== 'NONE') {
      const word = telemetry.gesture;
      if (lastSpokenWordRef.current !== word) {
        lastSpokenWordRef.current = word;
        // Append to sentence
        setSentenceWords(prev => [...prev, word]);
        // Add to transcript
        setConversationHistory(prev => [
          { word, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) },
          ...prev.slice(0, 19)
        ]);

        // Speak word naturally if sound enabled
        if (soundEnabled && 'speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(word.toLowerCase().replace('_', ' '));
          utterance.rate = 1.0;
          utterance.pitch = 1.0;
          window.speechSynthesis.speak(utterance);
        }
      }
    }
  }, [telemetry.gesture, appMode, soundEnabled]);

  const speakCurrentSentence = () => {
    if (sentenceWords.length > 0 && 'speechSynthesis' in window) {
      const text = sentenceWords.map(w => w.toLowerCase().replace('_', ' ')).join(' ');
      const utterance = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(utterance);
    }
  };

  const clearSentence = () => {
    setSentenceWords([]);
    lastSpokenWordRef.current = null;
  };

  // WebSocket Connection with exponential backoff
  useEffect(() => {
    let reconnectDelay = 1000;
    const MAX_RECONNECT_DELAY = 15000;

    const connectWs = () => {
      ws.current = new WebSocket('ws://localhost:8000/api/v1/stream');

      ws.current.onopen = () => {
        setConnected(true);
        reconnectDelay = 1000; // Reset on successful connection
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.image) setFrame(data.image);
          if (data.telemetry) setTelemetry(data.telemetry);
          if (data.alerts) setAlerts(data.alerts);
        } catch (e) {
          console.error("Error parsing websocket message", e);
        }
      };

      ws.current.onclose = () => {
        setConnected(false);
        setTimeout(connectWs, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT_DELAY);
      };
    };

    connectWs();

    return () => {
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current);
        sendIntervalRef.current = null;
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  // Direct Browser Webcam Management
  const startBrowserCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setBrowserCamActive(true);

      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
        canvasRef.current.width = 640;
        canvasRef.current.height = 360;
      }

      if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = setInterval(() => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN && videoRef.current && videoRef.current.readyState >= 2) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.65);
          ws.current.send(JSON.stringify({
            image: dataUrl,
            sign_mode: signModeRef.current,
          }));
        }
      }, 65);
    } catch (err) {
      console.error("Could not access browser camera:", err);
      alert("Could not access camera: " + err.message + "\nPlease click allow on your browser's camera prompt.");
    }
  };

  const stopBrowserCamera = () => {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setBrowserCamActive(false);
  };

  // Trigger test alerts via REST
  const triggerTestAlert = async (gesture, threatLevel = 'Critical') => {
    try {
      setLastTriggered(gesture);
      setTimeout(() => setLastTriggered(null), 1500);
      const res = await fetch(`http://localhost:8000/api/v1/trigger?gesture=${gesture}&threat_level=${threatLevel}`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.payload) {
        setAlerts(prev => [data.payload, ...prev.slice(0, 14)]);
      }
    } catch (e) {
      console.error("Failed to trigger alert:", e);
    }
  };

  const getThreatColor = (level) => {
    switch (level) {
      case 'Critical':
      case 'High': return 'high-threat';
      case 'Medium': return 'medium-threat';
      case 'Low': return 'safe';
      default: return '';
    }
  };

  const getThreatHex = (level) => {
    switch (level) {
      case 'Critical':
      case 'High': return '#ef4444';
      case 'Medium': return '#f59e0b';
      default: return '#10b981';
    }
  };

  const formatLocation = (loc) => {
    if (!loc) return null;
    if (typeof loc === 'string') return loc;
    if (typeof loc === 'object') {
      return loc.label || (loc.latitude ? `${loc.latitude.toFixed(4)}, ${loc.longitude.toFixed(4)}` : JSON.stringify(loc));
    }
    return String(loc);
  };

  const filteredNormalSigns = NORMAL_SIGNS.filter(s => 
    s.word.toLowerCase().includes(searchQuery.toLowerCase()) || 
    s.meaning.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.gesture.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredEmergencySigns = EMERGENCY_SIGNS_LIST.filter(s => 
    s.word.toLowerCase().includes(searchQuery.toLowerCase()) || 
    s.meaning.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.gesture.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="dashboard-container">
      <video ref={videoRef} playsInline muted style={{ display: 'none' }} />

      {/* Header */}
      <header className="header">
        <div className="flex items-center gap-3">
          <div className="logo-badge">
            <Shield size={26} className="text-accent-blue" />
          </div>
          <div>
            <h1>VERS v5.0 Command Center</h1>
            <p className="subtitle">Visual Emergency Response System & Multimodal Sign AI</p>
          </div>
        </div>

        <div className="header-controls">
          {/* Mode Switcher Tabs */}
          <div className="app-mode-switcher">
            <button 
              className={`mode-tab-btn ${appMode === 'emergency' ? 'active-tab-emergency' : ''}`}
              onClick={() => setAppMode('emergency')}
            >
              <AlertTriangle size={15} />
              Emergency Mode
            </button>
            <button 
              className={`mode-tab-btn ${appMode === 'conversation' ? 'active-tab-conversation' : ''}`}
              onClick={() => setAppMode('conversation')}
            >
              <MessageSquare size={15} />
              Conversational Mode
            </button>
          </div>

          <button 
            className="control-btn active-purple"
            onClick={() => setShowGlossary(true)}
            title="Open Sign Language Dictionary"
          >
            <BookOpen size={16} />
            Glossary ({NORMAL_SIGNS.length + EMERGENCY_SIGNS_LIST.length})
          </button>

          <button 
            className={`control-btn ${browserCamActive ? 'active-green' : ''}`}
            onClick={browserCamActive ? stopBrowserCamera : startBrowserCamera}
          >
            {browserCamActive ? <VideoOff size={16} /> : <Video size={16} />}
            {browserCamActive ? 'Stop Camera' : 'Start Camera'}
          </button>

          <button 
            className="control-btn"
            onClick={() => setSoundEnabled(!soundEnabled)}
            title="Toggle Audio Feedback"
          >
            {soundEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
          </button>

          <div className={`status-badge ${!connected ? 'disconnected' : ''}`}>
            <div className="pulse" />
            {connected ? 'AI Online' : 'Connecting...'}
          </div>
        </div>
      </header>

      {/* Main Vision & Stats Grid */}
      <main className="main-content">
        {/* Main Video Panel */}
        <div className="glass-panel p-4">
          <div className="camera-container">
            {frame ? (
              <img src={`data:image/jpeg;base64,${frame}`} alt="AI Camera Stream" className="camera-feed" />
            ) : (
              <div className="camera-placeholder">
                <Video size={48} opacity={0.4} />
                <p className="font-semibold text-lg">Live AI Gesture & Sign Feed</p>
                <p className="text-sm text-text-muted max-w-md text-center">
                  Click <strong>Start Camera</strong> to stream your webcam directly through the MediaPipe AI {appMode === 'conversation' ? 'Conversational Sign Translator' : 'Emergency Detector'}.
                </p>
                <button className="primary-action-btn" onClick={startBrowserCamera}>
                  <Video size={18} /> Enable Camera Now
                </button>
              </div>
            )}
            {frame && (
              <div className="camera-overlay-stats">
                FPS: {telemetry.fps > 0 ? telemetry.fps.toFixed(1) : '20.0'} | Mode: {appMode === 'conversation' ? 'Conversational Translation' : 'Emergency Monitor'}
              </div>
            )}
          </div>
        </div>

        {/* Dynamic Mode-Specific Content */}
        {appMode === 'conversation' ? (
          /* ============================================================ */
          /* CONVERSATIONAL SIGN MODE VIEW                                */
          /* ============================================================ */
          <div className="flex flex-col gap-4">
            {/* Live Sentence Builder & Real-time Translation */}
            <div className="glass-panel p-4 sentence-builder-panel">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Sparkles size={18} className="text-accent-purple" />
                  <span className="font-semibold">Live Sign-to-Speech Sentence Translator</span>
                </div>
                <div className="flex items-center gap-2">
                  <button 
                    className="action-pill-btn btn-speak"
                    onClick={speakCurrentSentence}
                    disabled={sentenceWords.length === 0}
                    title="Speak whole sentence aloud"
                  >
                    <Volume2 size={14} /> Speak Sentence
                  </button>
                  <button 
                    className="action-pill-btn btn-clear"
                    onClick={clearSentence}
                    disabled={sentenceWords.length === 0}
                    title="Clear sentence"
                  >
                    <Trash2 size={14} /> Clear
                  </button>
                </div>
              </div>

              <div className="sentence-display-box">
                {sentenceWords.length > 0 ? (
                  <div className="flex flex-wrap gap-2 items-center">
                    {sentenceWords.map((w, idx) => (
                      <span key={idx} className="sentence-word-pill">
                        {w}
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-text-muted italic text-sm">
                    Perform conversational sign gestures in front of the camera (or click signs below) to form a sentence...
                  </span>
                )}
              </div>
            </div>

            {/* Conversational Telemetry Cards */}
            <div className="telemetry-grid">
              <div className="glass-panel stat-card">
                <div className="stat-title"><Activity size={18} className="text-accent-purple" /> Translated Sign Word</div>
                <div className="stat-value text-accent-purple">
                  {telemetry.gesture !== 'NONE' ? telemetry.gesture : 'Listening...'}
                </div>
                <div className="text-sm text-text-muted mt-1">
                  Confidence: {(telemetry.confidence * 100).toFixed(0)}%
                </div>
              </div>

              <div className="glass-panel stat-card">
                <div className="stat-title"><MessageSquare size={18} className="text-accent-blue" /> Mode Status</div>
                <div className="stat-value text-accent-green">
                  Conversational Active
                </div>
                <div className="text-sm text-text-muted mt-1">
                  Emergency Alarms: <strong>Silenced</strong>
                </div>
              </div>

              <div className="glass-panel stat-card">
                <div className="stat-title"><Cpu size={18} className="text-accent-green" /> Facial State</div>
                <div className="stat-value" style={{ textTransform: 'capitalize' }}>
                  {telemetry.dominant_emotion || 'neutral'}
                </div>
                <div className="text-sm text-text-muted mt-1">
                  Emotion: {telemetry.dominant_emotion || 'neutral'}
                </div>
              </div>
            </div>

            {/* All Conversational Signs Interactive Pad */}
            <div className="glass-panel trigger-panel">
              <div className="trigger-header">
                <div className="flex items-center gap-2">
                  <MessageSquare size={18} className="text-accent-purple" />
                  <span className="font-semibold">Conversational Sign Language Quick Pad</span>
                  <span className="mode-pill pill-on">19 Conversational Signs</span>
                </div>
                <button className="text-xs text-accent-blue underline cursor-pointer bg-transparent border-none" onClick={() => setShowGlossary(true)}>
                  View 3D Geometries & Meaning →
                </button>
              </div>

              <div className="normal-signs-grid">
                {NORMAL_SIGNS.map((sign) => (
                  <button 
                    key={sign.word}
                    className={`normal-sign-btn ${sign.word === 'NO' ? 'btn-no-highlight' : ''} ${lastTriggered === sign.word ? 'triggered' : ''}`}
                    onClick={() => triggerTestAlert(sign.word, 'Low')}
                    title={sign.gesture}
                  >
                    <span className="sign-word">{sign.word}</span>
                    <span className="sign-meaning">{sign.meaning}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* ============================================================ */
          /* EMERGENCY RESPONSE MODE VIEW                                 */
          /* ============================================================ */
          <div className="flex flex-col gap-4">
            {/* Telemetry Stat Cards */}
            <div className="telemetry-grid">
              <div className="glass-panel stat-card">
                <div className="stat-title"><Activity size={18} className="text-accent-blue" /> Emergency Gesture</div>
                <div className={`stat-value ${telemetry.gesture !== 'NONE' && telemetry.gesture !== 'No gesture' ? 'high-threat' : ''}`}>
                  {telemetry.gesture || 'NONE'}
                </div>
                <div className="text-sm text-text-muted mt-1">
                  Confidence: {(telemetry.confidence * 100).toFixed(0)}%
                </div>
              </div>
              
              <div className="glass-panel stat-card">
                <div className="stat-title"><AlertTriangle size={18} className="text-accent-orange" /> Threat Assessment</div>
                <div className={`stat-value ${getThreatColor(telemetry.threat_level)}`}>
                  {telemetry.threat_level || 'NONE'}
                </div>
                <div className="progress-bg">
                  <div 
                    className="progress-fill" 
                    style={{ 
                      width: `${Math.min(100, Math.max(8, telemetry.distress_score * 100))}%`,
                      backgroundColor: getThreatHex(telemetry.threat_level)
                    }} 
                  />
                </div>
              </div>

              <div className="glass-panel stat-card">
                <div className="stat-title"><Cpu size={18} className="text-accent-green" /> Facial Distress Score</div>
                <div className="stat-value" style={{ textTransform: 'capitalize' }}>
                  {telemetry.dominant_emotion || 'neutral'}
                </div>
                <div className="text-sm text-text-muted mt-1">
                  Distress: {(telemetry.distress_score * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            {/* Emergency Signals Simulator */}
            <div className="glass-panel trigger-panel">
              <div className="trigger-header">
                <div className="flex items-center gap-2">
                  <Radio size={18} className="text-accent-red animate-pulse" />
                  <span className="font-semibold">Critical Emergency Signals (Siren Dispatch)</span>
                  <span className="mode-pill pill-off">Emergency Mode Active</span>
                </div>
                <span className="text-xs text-text-muted">High priority alerts with instant audio siren dispatch</span>
              </div>

              <div className="trigger-buttons-grid">
                <button 
                  className={`trigger-btn btn-sos ${lastTriggered === 'SOS' ? 'triggered' : ''}`}
                  onClick={() => triggerTestAlert('SOS', 'Critical')}
                >
                  <ShieldAlert size={20} />
                  <span>SOS / HELP</span>
                </button>
                <button 
                  className={`trigger-btn btn-medical ${lastTriggered === 'MEDICAL' ? 'triggered' : ''}`}
                  onClick={() => triggerTestAlert('MEDICAL', 'Critical')}
                >
                  <Activity size={20} />
                  <span>MEDICAL</span>
                </button>
                <button 
                  className={`trigger-btn btn-fire ${lastTriggered === 'FIRE' ? 'triggered' : ''}`}
                  onClick={() => triggerTestAlert('FIRE', 'High')}
                >
                  <Flame size={20} />
                  <span>FIRE ALARM</span>
                </button>
                <button 
                  className={`trigger-btn btn-police ${lastTriggered === 'POLICE' ? 'triggered' : ''}`}
                  onClick={() => triggerTestAlert('POLICE', 'High')}
                >
                  <Radio size={20} />
                  <span>POLICE ASSIST</span>
                </button>
                <button 
                  className={`trigger-btn btn-safe ${lastTriggered === 'SAFE' ? 'triggered' : ''}`}
                  onClick={() => triggerTestAlert('SAFE', 'Low')}
                >
                  <CheckCircle2 size={20} />
                  <span>ALL CLEAR</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Sidebar Alerts Panel */}
      <aside className="sidebar">
        <div className="glass-panel alerts-panel">
          <div className="alerts-header">
            <Shield size={20} className="text-accent-blue" />
            <span>Live Dispatch Log</span>
            <span className="alert-badge">{alerts.length}</span>
          </div>

          <div className="alerts-list">
            {alerts.length === 0 ? (
              <div className="empty-alerts">
                <Shield size={36} opacity={0.3} />
                <p className="font-medium text-sm">System Guard Active</p>
                <p className="text-xs text-text-muted text-center">
                  Perform a normal sign or emergency gesture in front of the camera to see real-time detection.
                </p>
              </div>
            ) : (
              alerts.map((alert, idx) => (
                <div key={idx} className={`alert-item ${alert.ThreatLevel || alert.Severity || 'Low'}`}>
                  <div className="alert-item-header">
                    <span className="alert-title">{alert.MainGesture || alert.Gesture || 'EMERGENCY'}</span>
                    <span className="alert-time">
                      {alert.Timestamp ? alert.Timestamp.split('T')[1]?.substring(0, 8) : new Date().toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="alert-details">
                    <div className="flex justify-between">
                      <span>Threat: <strong>{alert.ThreatLevel || alert.Severity || 'Critical'}</strong></span>
                      <span>Score: {((alert.SeverityScore || 0.9) * 100).toFixed(0)}%</span>
                    </div>
                    {alert.Location && <div className="text-xs text-text-muted">📍 {formatLocation(alert.Location)}</div>}
                    {alert.Message && <div className="text-xs text-text-muted italic">{alert.Message}</div>}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </aside>

      {/* Sign Language Glossary Modal */}
      {showGlossary && (
        <div className="modal-backdrop" onClick={() => setShowGlossary(false)}>
          <div className="modal-content glass-panel" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="flex items-center gap-2">
                <BookOpen size={22} className="text-accent-purple" />
                <h2>VERS Sign Language Dictionary & Geometry Guide</h2>
              </div>
              <button className="close-btn" onClick={() => setShowGlossary(false)}>
                <X size={20} />
              </button>
            </div>

            <div className="modal-controls">
              <div className="tab-buttons">
                <button 
                  className={`tab-btn ${glossaryTab === 'normal' ? 'active' : ''}`}
                  onClick={() => setGlossaryTab('normal')}
                >
                  <Hand size={16} /> Normal / Conversational Signs ({NORMAL_SIGNS.length})
                </button>
                <button 
                  className={`tab-btn ${glossaryTab === 'emergency' ? 'active' : ''}`}
                  onClick={() => setGlossaryTab('emergency')}
                >
                  <AlertTriangle size={16} /> Emergency Signs ({EMERGENCY_SIGNS_LIST.length})
                </button>
              </div>

              <div className="search-bar">
                <Search size={16} className="text-text-muted" />
                <input 
                  type="text" 
                  placeholder="Search signs, meanings, or hand movements..." 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            <div className="glossary-grid">
              {glossaryTab === 'normal' ? (
                filteredNormalSigns.map((s) => (
                  <div key={s.word} className="glossary-card">
                    <div className="glossary-card-header">
                      <span className="card-word">{s.word}</span>
                      <span className="card-badge">{s.category}</span>
                    </div>
                    <p className="card-meaning">{s.meaning}</p>
                    <div className="card-gesture-box">
                      <span className="text-xs text-text-muted font-medium">Hand Geometry:</span>
                      <p className="card-gesture">{s.gesture}</p>
                    </div>
                    <button 
                      className="card-test-btn"
                      onClick={() => {
                        triggerTestAlert(s.word, 'Low');
                      }}
                    >
                      ⚡ Simulate Sign
                    </button>
                  </div>
                ))
              ) : (
                filteredEmergencySigns.map((s) => (
                  <div key={s.word} className="glossary-card emergency-card">
                    <div className="glossary-card-header">
                      <span className="card-word text-accent-red">{s.word}</span>
                      <span className={`card-badge ${s.level}`}>{s.level} Threat</span>
                    </div>
                    <p className="card-meaning">{s.meaning}</p>
                    <div className="card-gesture-box">
                      <span className="text-xs text-text-muted font-medium">Hand Geometry:</span>
                      <p className="card-gesture">{s.gesture}</p>
                    </div>
                    <button 
                      className="card-test-btn btn-sos"
                      onClick={() => {
                        triggerTestAlert(s.word, s.level);
                      }}
                    >
                      🚨 Dispatch {s.word}
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
