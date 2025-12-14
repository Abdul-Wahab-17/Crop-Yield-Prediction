import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav className="navbar">
      <h2>🌱 Crop Yield AI</h2>
      <div>
        <Link to="/">Home</Link>
        <Link to="/predict">Predict</Link>
        <Link to="/regional">Regional</Link>
        <Link to="/models">Models</Link>
      </div>
    </nav>
  );
}
