import React, { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import './Metrics.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function Metrics() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://localhost:5000/metrics')
      .then(res => res.json())
      .then(data => {
        if (data.error) setError(data.error);
        else setData(data);
      })
      .catch(err => setError("Failed to fetch metrics"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading metrics...</div>;
  if (error) return <p style={{ color: 'red' }}>{error}</p>;

  // Color mapping by model type
  const colorMap = {
    'Baseline': '#9e9e9e',
    'Classical ML': '#4fc3f7',
    'Ensemble/Advanced': '#43a047'
  };

  // Prepare data for charts (exclude baselines from main charts)
  const mlModels = data.filter(d => d.Type !== 'Baseline');
  const allModels = [...data];

  // Sort for R² chart (descending)
  const r2Data = [...mlModels].sort((a, b) => b['R²'] - a['R²']);

  // Sort for training time (ascending)
  const timeData = [...mlModels].sort((a, b) => a['Training Time (s)'] - b['Training Time (s)']);

  // Sort for MAE (ascending - lower is better)
  const maeData = [...mlModels].sort((a, b) => a['MAE'] - b['MAE']);

  // Sort for RMSE (ascending - lower is better)
  const rmseData = [...mlModels].sort((a, b) => a['RMSE'] - b['RMSE']);

  // Chart 1: R² Score Bar Chart
  const r2ChartData = {
    labels: r2Data.map(d => d.Model),
    datasets: [{
      label: 'R² Score',
      data: r2Data.map(d => d['R²']),
      backgroundColor: r2Data.map(d => colorMap[d.Type]),
      borderRadius: 8,
    }]
  };

  const r2Options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: { display: true, text: 'R² Score' }
      },
      x: {
        ticks: {
          autoSkip: false,
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 }
        }
      }
    }
  };

  // Chart 2: Training Time Bar Chart
  const timeChartData = {
    labels: timeData.map(d => d.Model),
    datasets: [{
      label: 'Training Time (s)',
      data: timeData.map(d => d['Training Time (s)']),
      backgroundColor: timeData.map(d => colorMap[d.Type]),
      borderRadius: 8,
    }]
  };

  const timeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Training Time (seconds)' }
      },
      x: {
        ticks: {
          autoSkip: false,
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 }
        }
      }
    }
  };

  // Chart 3: MAE Bar Chart
  const maeChartData = {
    labels: maeData.map(d => d.Model),
    datasets: [{
      label: 'MAE',
      data: maeData.map(d => d['MAE']),
      backgroundColor: maeData.map(d => colorMap[d.Type]),
      borderRadius: 8,
    }]
  };

  const maeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Mean Absolute Error' }
      },
      x: {
        ticks: {
          autoSkip: false,
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 }
        }
      }
    }
  };

  // Chart 4: RMSE Bar Chart
  const rmseChartData = {
    labels: rmseData.map(d => d.Model),
    datasets: [{
      label: 'RMSE',
      data: rmseData.map(d => d['RMSE']),
      backgroundColor: rmseData.map(d => colorMap[d.Type]),
      borderRadius: 8,
    }]
  };

  const rmseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Root Mean Square Error' }
      },
      x: {
        ticks: {
          autoSkip: false,
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 }
        }
      }
    }
  };

  return (
    <div className="metrics-container">
      <h2>Model Performance Comparison</h2>
      <p className="metrics-subtitle">Comparing {data.length} models across multiple performance metrics</p>

      {/* Charts Section */}
      <div className="charts-grid">
        {/* Chart 1: R² Score Comparison */}
        <div className="chart-card">
          <h3>Model Accuracy (R² Score)</h3>
          <div style={{ height: '300px' }}>
            <Bar data={r2ChartData} options={r2Options} />
          </div>
        </div>

        {/* Chart 2: Training Time Comparison */}
        <div className="chart-card">
          <h3>Training Time (seconds)</h3>
          <div style={{ height: '300px' }}>
            <Bar data={timeChartData} options={timeOptions} />
          </div>
        </div>

        {/* Chart 3: MAE Comparison */}
        <div className="chart-card">
          <h3>Mean Absolute Error (MAE)</h3>
          <div style={{ height: '300px' }}>
            <Bar data={maeChartData} options={maeOptions} />
          </div>
        </div>

        {/* Chart 4: RMSE Comparison */}
        <div className="chart-card">
          <h3>Root Mean Square Error (RMSE)</h3>
          <div style={{ height: '300px' }}>
            <Bar data={rmseChartData} options={rmseOptions} />
          </div>
        </div>
      </div>

      {/* Detailed Table */}
      <div className="table-section">
        <h3>Detailed Metrics Table</h3>
        <div className="table-container">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>MAPE (%)</th>
                <th>Within 10%</th>
                <th>Training Time (s)</th>
                <th>Complexity</th>
                <th>Interpretability</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {allModels.map((row, idx) => (
                <tr key={idx} className={`row - type - ${row.Type.replace(/\s+/g, '-').toLowerCase()} `}>
                  <td><strong>{row.Model}</strong></td>
                  <td>{row["R²"].toFixed(3)}</td>
                  <td>{row.RMSE.toFixed(2)}</td>
                  <td>{row.MAE.toFixed(2)}</td>
                  <td>{row["MAPE (%)"].toFixed(2)}</td>
                  <td>{row["Within 10%"].toFixed(2)}</td>
                  <td>{row["Training Time (s)"].toFixed(2)}</td>
                  <td>{row.Complexity}</td>
                  <td>{row.Interpretability}</td>
                  <td><span className={`badge badge - ${row.Type.replace(/\s+/g, '-').toLowerCase()} `}>{row.Type}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Legend */}
      <div className="legend-section">
        <h4>Model Type Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: colorMap['Baseline'] }}></span>
            <span>Baseline Models</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: colorMap['Classical ML'] }}></span>
            <span>Classical ML</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: colorMap['Ensemble/Advanced'] }}></span>
            <span>Ensemble/Advanced</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Metrics;
