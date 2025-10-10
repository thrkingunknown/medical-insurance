import { InsuranceForm } from "./components/InsuranceForm";
import { ThemeProvider } from "./components/ThemeProvider";
import { Analytics } from "@vercel/analytics/react";
import "./App.css";

function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="medical-insurance-theme">
      <Analytics />
      <InsuranceForm />
    </ThemeProvider>
  );
}

export default App;
