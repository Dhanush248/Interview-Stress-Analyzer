import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

const HomePage = () => {
  const [roomId, setRoomId] = useState('');
  const [userName, setUserName] = useState('');
  const [role, setRole] = useState('interviewer');
  const navigate = useNavigate();

  const generateRoomId = () => {
    const id = Math.random().toString(36).substring(2, 15);
    setRoomId(id);
  };

  const joinRoom = () => {
    if (!roomId.trim() || !userName.trim()) {
      alert('Please enter both room ID and your name');
      return;
    }

    navigate(`/interview/${roomId}?role=${role}&name=${encodeURIComponent(userName)}`);
  };

  return (
    <div className="home-container">
      <div className="home-content">
        <h1>Interview Stress Analysis System</h1>
        <p>Real-time AI-powered stress and confidence analysis for interviews</p>
        
        <div className="form-container">
          <div className="form-group">
            <label>Your Name:</label>
            <input
              type="text"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              placeholder="Enter your name"
            />
          </div>

          <div className="form-group">
            <label>Role:</label>
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="interviewer">Interviewer</option>
              <option value="interviewee">Interviewee</option>
            </select>
          </div>

          <div className="form-group">
            <label>Room ID:</label>
            <div className="room-input-group">
              <input
                type="text"
                value={roomId}
                onChange={(e) => setRoomId(e.target.value)}
                placeholder="Enter room ID or generate new"
              />
              <button onClick={generateRoomId} className="generate-btn">
                Generate
              </button>
            </div>
          </div>

          <button onClick={joinRoom} className="join-btn">
            Join Interview Room
          </button>
        </div>

        <div className="info-section">
          <h3>How it works:</h3>
          <ul>
            <li><strong>Interviewer:</strong> Can see real-time stress and confidence analytics</li>
            <li><strong>Interviewee:</strong> Participates in the interview (analytics hidden)</li>
            <li><strong>AI Analysis:</strong> Processes facial expressions and voice patterns</li>
            <li><strong>Privacy:</strong> Only interviewers see the AI analysis results</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default HomePage;