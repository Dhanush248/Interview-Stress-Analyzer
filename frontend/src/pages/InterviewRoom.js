import React, { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import io from 'socket.io-client';
import VideoCall from '../components/VideoCall';
import StressAnalytics from '../components/StressAnalytics';
import './InterviewRoom.css';

const InterviewRoom = () => {
  const { roomId } = useParams();
  const [searchParams] = useSearchParams();
  const role = searchParams.get('role');
  const userName = searchParams.get('name');

  const [socket, setSocket] = useState(null);
  const [participants, setParticipants] = useState([]);
  const [stressData, setStressData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');

  const localVideoRef = useRef();
  const remoteVideoRef = useRef();
  const peerConnectionRef = useRef();
  const localStreamRef = useRef();

  useEffect(() => {
    const newSocket = io('http://localhost:3000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      setIsConnected(true);
      newSocket.emit('join-room', { roomId, role, userName });
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
    });

    newSocket.on('user-joined', (data) => {
      setParticipants(prev => [...prev, data]);
    });

    newSocket.on('user-left', (data) => {
      setParticipants(prev => prev.filter(p => p.socketId !== data.socketId));
    });

    newSocket.on('room-participants', (data) => {
      setParticipants(data);
    });

    newSocket.on('stress_analysis', (data) => {
      if (role === 'interviewer') {
        setStressData(data.data);
      }
    });

    newSocket.on('chat-message', (data) => {
      setChatMessages(prev => [...prev, data]);
    });

    return () => {
      newSocket.disconnect();
    };
  }, [roomId, role, userName]);

  const sendChatMessage = () => {
    if (newMessage.trim() && socket) {
      socket.emit('chat-message', { message: newMessage, userName: userName });
      setChatMessages(prev => [...prev, {
        message: newMessage,
        sender: userName,
        timestamp: Date.now()
      }]);
      setNewMessage('');
    }
  };

  const resetAnalysis = () => {
    if (socket) {
      socket.emit('reset-analysis');
      setStressData(null);
    }
  };

  return (
    <div className="interview-room">
      <div className="room-header">
        <div className="header-left">
          <div className="room-icon">🎯</div>
          <div>
            <h2>Room: {roomId}</h2>
            <span className="room-subtitle">{userName} • {role}</span>
          </div>
        </div>
        <div className="header-right">
          <span className={`status-badge ${isConnected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="room-content">
        <div className="left-panel">
          <div className="chat-section">
            <div className="chat-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z" fill="currentColor"/>
              </svg>
              <h3>Chat</h3>
            </div>
            <div className="chat-messages">
              {chatMessages.map((msg, index) => (
                <div key={index} className={`chat-message ${msg.sender === userName ? 'own' : 'other'}`}>
                  <div className="message-avatar">{msg.sender.charAt(0).toUpperCase()}</div>
                  <div className="message-content">
                    <div className="message-header">
                      <span className="sender-name">{msg.sender}</span>
                      <span className="message-time">{new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                    </div>
                    <div className="message-text">{msg.message}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="chat-input">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                placeholder="Type a message..."
              />
              <button onClick={sendChatMessage} className="send-btn">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>
                </svg>
              </button>
            </div>
          </div>
        </div>

        <div className="center-panel">
          <VideoCall
            socket={socket}
            role={role}
            participants={participants}
            localVideoRef={localVideoRef}
            remoteVideoRef={remoteVideoRef}
            peerConnectionRef={peerConnectionRef}
            localStreamRef={localStreamRef}
          />
        </div>

        {role === 'interviewer' && (
          <div className="right-panel">
            <StressAnalytics 
              stressData={stressData}
              onReset={resetAnalysis}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default InterviewRoom;