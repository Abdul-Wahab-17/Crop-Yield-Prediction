import React, { useState, useEffect } from 'react';
import './../App.css';

function Regional() {
  const [metadata, setMetadata] = useState({ crops: [], states: [], available_models: [] });
  const [counties, setCounties] = useState([]);
  const [validStates, setValidStates] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  const [formData, setFormData] = useState({
    State: '',
    County: '',
    Year: new Date().getFullYear(),
    prcp_growing_total: '',
    tavg_growing_avg: '',
    tmin_growing_avg: '',
    tmax_growing_avg: '',
    rh_growing_avg: '',
    gdd_total: '',
    heat_stress_days: '',
    temp_range_avg: '',
    clay_pct: '',
    sand_pct: '',
    silt_pct: '',
    organic_matter_pct: '',
    ph: '',
    bulk_density: '',
    cec: '',
    awc: '',
    Latitude: '',
    Longitude: ''
  });

  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load metadata
  useEffect(() => {
    fetch('http://localhost:5000/metadata')
      .then(res => res.json())
      .then(data => {
        setMetadata(data);
        setSelectedModel(data.current_model);
      })
      .catch(err => console.error("Failed to load metadata", err));
  }, []);

  // Reset state if invalid
  useEffect(() => {
    if (formData.State && !metadata.states.includes(formData.State)) {
      setFormData(prev => ({ ...prev, State: '' }));
    }
  }, [metadata.states]);

  // Fetch valid states when crop changes — here crop is iterated, so we just consider all crops
  useEffect(() => {
    if (metadata.states.length > 0) {
      setValidStates(metadata.states);
    }
  }, [metadata.states]);

  // Fetch counties when state changes
  useEffect(() => {
    if (formData.State) {
      fetch(`http://localhost:5000/counties/${formData.State}`)
        .then(res => res.json())
        .then(data => setCounties(data.counties))
        .catch(err => console.error("Failed to load counties", err));
      setFormData(prev => ({ ...prev, County: '' }));
    } else {
      setCounties([]);
    }
  }, [formData.State]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);
    fetch(`http://localhost:5000/models/${newModel}`, { method: 'POST' })
      .then(res => res.json())
      .then(data => console.log(data.message))
      .catch(err => console.error("Failed to switch model", err));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults([]);

    if (!formData.State || !formData.County) {
      setError("State and County are required");
      setLoading(false);
      return;
    }

    try {
      const cropList = metadata.crops;
      const predictions = [];

      for (const crop of cropList) {
        const requestData = { ...formData, Crop: crop, model: selectedModel };
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestData)
        });
        const data = await response.json();
        if (response.ok) {
          predictions.push({ crop, ...data });
        } else {
          predictions.push({ crop, error: data.error || "Prediction failed" });
        }
      }

      setResults(predictions);
    } catch (err) {
      setError("Network error. Ensure Flask backend is running.");
    }

    setLoading(false);
  };

  return (
    <div className="app-container">
      <header>
        <h1>🌱 Regional Crop Yield Prediction</h1>
        <p>Enter State and County to get predictions for all crops</p>
      </header>

      <div className="main-content">
        <form onSubmit={handleSubmit} className="prediction-form">

          {/* Model Selection */}
          <div className="form-section">
            <h3>Model Selection</h3>
            <div className="input-group">
              <label>ML Model</label>
              <select value={selectedModel} onChange={handleModelChange}>
                {metadata.available_models.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Required Fields */}
          <div className="form-section">
            <h3>Location (Required)</h3>
            <div className="input-group">
              <label>State</label>
              <select name="State" value={formData.State} onChange={handleChange} required>
                <option value="">-- Select State --</option>
                {validStates.map(state => (
                  <option key={state} value={state}>{state}</option>
                ))}
              </select>
            </div>
            <div className="input-group">
              <label>County</label>
              <select
                name="County"
                value={formData.County}
                onChange={handleChange}
                required
                disabled={!formData.State}
              >
                <option value="">-- Select County --</option>
                {counties.map(county => (
                  <option key={county} value={county}>{county}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Optional Fields */}
          <div className="form-section">
            <h3>Optional Inputs</h3>
            <div className="input-grid">
              {Object.keys(formData)
                .filter(key => !['State', 'County'].includes(key))
                .map(key => (
                  <div className="input-group" key={key}>
                    <label>{key.replace(/_/g, ' ')}</label>
                    <input
                      type="number"
                      name={key}
                      value={formData[key]}
                      placeholder={`Default: ${formData[key] || ''}`}
                      onChange={handleChange}
                      step="any"
                    />
                  </div>
                ))}
            </div>
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Calculating...' : 'Predict All Crops'}
          </button>
        </form>

        {/* Results */}
        {results.length > 0 && (
          <div className="result-card">
            <h2>Predictions</h2>
            {results.map((res, idx) => (
              <div key={idx} className="prediction-item">
                <strong>{res.crop}</strong>: {res.error ? res.error : `${res.prediction.toFixed(2)} ${res.units}`}
                {res.warnings && res.warnings.length > 0 && (
                  <div className="warnings">
                    {res.warnings.map((w, i) => <div key={i}>⚠️ {w}</div>)}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {error && <div className="error-msg">{error}</div>}
      </div>
    </div>
  );
}

export default Regional;
