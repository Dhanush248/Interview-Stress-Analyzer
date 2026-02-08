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
    // Initialize socket connection
    const newSocket = io('http://localhost:3000');
    setSocket(newSocket);

    // Socket event listeners
    newSocket.on('connect', () => {
      setIsConnected(true);
      console.log('Connected to server');
      
      // Join room
      newSocket.emit('join-room', {
        roomId,
        role,
        userName
      });
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
      console.log('Disconnected from server');
    });

    newSocket.on('user-joined', (data) => {
      console.log('User joined:', data);
      setParticipants(prev => [...prev, data]);
    });

    newSocket.on('user-left', (data) => {
      console.log('User left:', data);
      setParticipants(prev => prev.filter(p => p.socketId !== data.socketId));
    });

    newSocket.on('room-participants', (data) => {
      setParticipants(data);
    });

    // AI Analysis results (only for interviewers)
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
      socket.emit('chat-message', { message: newMessage });
      setChatMessages(prev => [...prev, {
        message: newMessage,
        sender: 'You',
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
        <h2>Interview Room: {roomId}</h2>
        <div className="user-info">
          <span>{userName} ({role})</span>
          <span className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="room-content">
        <div className="video-section">
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
          <div className="analytics-section">
            <StressAnalytics 
              stressData={stressData}
              onReset={resetAnalysis}
            />
          </div>
        )}

        <div className="chat-section">
          <div className="chat-header">
            <h3>Chat</h3>
          </div>
          <div className="chat-messages">
            {chatMessages.map((msg, index) => (
              <div key={index} className="chat-message">
                <span className="sender">{msg.sender}:</span>
                <span className="message">{msg.message}</span>
                <span className="timestamp">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
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
            <button onClick={sendChatMessage}>Send</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InterviewRoom;