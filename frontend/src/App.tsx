import Dashboard from "./components/Dashboard";
import MapPage from "./components/MapPage";

const App = () => {
  const path = window.location.pathname;
  if (path.startsWith("/dashboard/maps")) {
    return <MapPage />;
  }
  return <Dashboard />;
};

export default App;
