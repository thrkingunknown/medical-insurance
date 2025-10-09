import { InsuranceForm } from "./components/InsuranceForm";
import { ThemeProvider } from "./components/ThemeProvider";
import "./App.css";

function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="medical-insurance-theme">
      <InsuranceForm />
    </ThemeProvider>
  );
}

export default App;
