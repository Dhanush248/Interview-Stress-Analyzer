import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import './StressAnalytics.css';

const StressAnalytics = ({ stressData, onReset }) => {
  const [historicalData, setHistoricalData] = useState([]);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);

  useEffect(() => {
    if (stressData) {
      setCurrentAnalysis(stressData);
      
      // Add to historical data
      const timestamp = new Date().toLocaleTimeString();
      const newDataPoint = {
        time: timestamp,
        stressLevel: getStressLevelValue(stressData.stress_level),
        confidence: Math.round(stressData.confidence_score * 100),
        timestamp: Date.now()
      };

      setHistoricalData(prev => {
        const updated = [...prev, newDataPoint];
        // Keep only last 20 data points
        return updated.slice(-20);
      });
    }
  }, [stressData]);

  const getStressLevelValue = (level) => {
    switch (level) {
      case 'Low Stress': return 1;
      case 'Medium Stress': return 2;
      case 'High Stress': return 3;
      default: return 0;
    }
  };

  const getStressColor = (level) => {
    switch (level) {
      case 'Low Stress': return '#4CAF50';
      case 'Medium Stress': return '#FF9800';
      case 'High Stress': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  const getConfidenceColor = (score) => {
    if (score >= 0.7) return '#4CAF50';
    if (score >= 0.4) return '#FF9800';
    return '#F44336';
  };

  const clearHistory = () => {
    setHistoricalData([]);
    setCurrentAnalysis(null);
    if (onReset) {
      onReset();
    }
  };

  if (!currentAnalysis) {
    return (
      <div className="stress-analytics">
        <div className="analytics-header">
          <h3>AI Stress Analysis</h3>
          <button onClick={clearHistory} className="reset-btn">Reset</button>
        </div>
        <div className="no-data">
          <p>Waiting for analysis data...</p>
          <div className="loading-spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="stress-analytics">
      <div className="analytics-header">
        <h3>AI Stress Analysis</h3>
        <button onClick={clearHistory} className="reset-btn">Reset</button>
      </div>

      {/* Current Status */}
      <div className="current-status">
        <div className="status-card">
          <h4>Current Stress Level</h4>
          <div 
            className="stress-indicator"
            style={{ backgroundColor: getStressColor(currentAnalysis.stress_level) }}
          >
            {currentAnalysis.stress_level}
          </div>
        </div>

        <div className="status-card">
          <h4>Confidence Score</h4>
          <div 
            className="confidence-indicator"
            style={{ backgroundColor: getConfidenceColor(currentAnalysis.confidence_score) }}
          >
            {Math.round(currentAnalysis.confidence_score * 100)}%
          </div>
        </div>

        <div className="status-card">
          <h4>Detection Status</h4>
          <div className="detection-status">
            <div className={`status-item ${currentAnalysis.face_detected ? 'active' : 'inactive'}`}>
              👤 Face: {currentAnalysis.face_detected ? 'Detected' : 'Not Detected'}
            </div>
            <div className={`status-item ${currentAnalysis.audio_processed ? 'active' : 'inactive'}`}>
              🎤 Audio: {currentAnalysis.audio_processed ? 'Processing' : 'No Audio'}
            </div>
          </div>
        </div>
      </div>

      {/* Stress Probability Distribution */}
      <div className="probability-section">
        <h4>Stress Level Probabilities</h4>
        <div className="probability-bars">
          {currentAnalysis.stress_probability.map((prob, index) => {
            const labels = ['Low Stress', 'Medium Stress', 'High Stress'];
            const colors = ['#4CAF50', '#FF9800', '#F44336'];
            
            return (
              <div key={index} className="probability-bar">
                <div className="bar-label">{labels[index]}</div>
                <div className="bar-container">
                  <div 
                    className="bar-fill"
                    style={{ 
                      width: `${prob * 100}%`,
                      backgroundColor: colors[index]
                    }}
                  ></div>
                </div>
                <div className="bar-value">{Math.round(prob * 100)}%</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Historical Charts */}
      {historicalData.length > 1 && (
        <div className="charts-section">
          <div className="chart-container">
            <h4>Stress Level Over Time</h4>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 3]} tickFormatter={(value) => {
                  const labels = ['', 'Low', 'Medium', 'High'];
                  return labels[value] || '';
                }} />
                <Tooltip formatter={(value) => {
                  const labels = ['Unknown', 'Low Stress', 'Medium Stress', 'High Stress'];
                  return [labels[value] || 'Unknown', 'Stress Level'];
                }} />
                <Line 
                  type="monotone" 
                  dataKey="stressLevel" 
                  stroke="#FF6B6B" 
                  strokeWidth={2}
                  dot={{ fill: '#FF6B6B', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-container">
            <h4>Confidence Score Over Time</h4>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 100]} />
                <Tooltip formatter={(value) => [`${value}%`, 'Confidence']} />
                <Line 
                  type="monotone" 
                  dataKey="confidence" 
                  stroke="#4ECDC4" 
                  strokeWidth={2}
                  dot={{ fill: '#4ECDC4', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Analysis Summary */}
      <div className="analysis-summary">
        <h4>Analysis Summary</h4>
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label">Total Readings:</span>
            <span className="stat-value">{historicalData.length}</span>
          </div>
          {historicalData.length > 0 && (
            <>
              <div className="stat-item">
                <span className="stat-label">Avg Confidence:</span>
                <span className="stat-value">
                  {Math.round(historicalData.reduce((sum, item) => sum + item.confidence, 0) / historicalData.length)}%
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Session Duration:</span>
                <span className="stat-value">
                  {historicalData.length > 1 
                    ? Math.round((historicalData[historicalData.length - 1].timestamp - historicalData[0].timestamp) / 1000 / 60)
                    : 0} min
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Recommendations */}
      <div className="recommendations">
        <h4>Recommendations</h4>
        <div className="recommendation-list">
          {currentAnalysis.stress_level === 'High Stress' && (
            <div className="recommendation high-stress">
              ⚠️ High stress detected. Consider taking a short break or asking easier questions.
            </div>
          )}
          {currentAnalysis.confidence_score < 0.3 && (
            <div className="recommendation low-confidence">
              💡 Low confidence detected. Try encouraging the candidate or providing more context.
            </div>
          )}
          {!currentAnalysis.face_detected && (
            <div className="recommendation technical">
              📹 Face not detected. Ask the candidate to adjust their camera position.
            </div>
          )}
          {!currentAnalysis.audio_processed && (
            <div className="recommendation technical">
              🎤 Audio not being processed. Check microphone settings.
            </div>
          )}
          {currentAnalysis.stress_level === 'Low Stress' && currentAnalysis.confidence_score > 0.7 && (
            <div className="recommendation positive">
              ✅ Candidate appears comfortable and confident. Good interview flow!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StressAnalytics;