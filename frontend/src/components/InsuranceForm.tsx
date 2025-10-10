import * as React from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { ThemeToggle } from "@/components/ThemeToggle";
import { GithubButton } from "@/components/GithubButton";
import {
  User,
  TrendingUp,
  Users,
  Cigarette,
  MapPin,
  Crown,
  Star,
  Sparkles,
} from "lucide-react";

interface FormData {
  age: string;
  sex: string;
  bmi: string;
  children: string;
  smoker: string;
  region: string;
}

interface PredictionResult {
  ensemble: number;
  modelAccuracies: {
    rf: number;
    gb: number;
    lgb: number;
    ensemble: number;
  };
}

export function InsuranceForm() {
  const [formData, setFormData] = React.useState<FormData>({
    age: "",
    sex: "",
    bmi: "",
    children: "",
    smoker: "",
    region: "",
  });

  const [errors, setErrors] = React.useState<Partial<FormData>>({});
  const [prediction, setPrediction] = React.useState<PredictionResult | null>(
    null
  );
  const [isLoading, setIsLoading] = React.useState(false);

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    if (errors[name as keyof FormData]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
  };

  const handleSelectChange = (name: keyof FormData, value: string) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
  };

  const validateForm = (): boolean => {
    const newErrors: Partial<FormData> = {};

    if (
      !formData.age ||
      parseInt(formData.age) < 0 ||
      parseInt(formData.age) > 120
    ) {
      newErrors.age = "Please enter a valid age (0-120)";
    }

    if (!formData.sex) {
      newErrors.sex = "Please select a sex";
    }

    if (
      !formData.bmi ||
      parseFloat(formData.bmi) < 10 ||
      parseFloat(formData.bmi) > 60
    ) {
      newErrors.bmi = "Please enter a valid BMI (10-60)";
    }

    if (!formData.children || parseInt(formData.children) < 0) {
      newErrors.children = "Please enter a valid number of children";
    }

    if (!formData.smoker) {
      newErrors.smoker = "Please select smoking status";
    }

    if (!formData.region) {
      newErrors.region = "Please select a region";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      setPrediction({
        ensemble: data.ensemble,
        modelAccuracies: data.modelAccuracies,
      });
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to get prediction. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      age: "",
      sex: "",
      bmi: "",
      children: "",
      smoker: "",
      region: "",
    });
    setErrors({});
    setPrediction(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-900 dark:to-slate-900 py-8 sm:py-12 px-4 sm:px-6 lg:px-8 transition-colors duration-300">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8 sm:mb-12 relative">
          <div className="absolute top-0 right-0 flex gap-2 animate-in fade-in slide-in-from-right-4 duration-500">
            <GithubButton />
            <ThemeToggle />
          </div>
          <div className="flex justify-center mb-4 animate-in fade-in zoom-in duration-500">
            <img
              src="/LOGO.png"
              className="h-16 sm:h-20 md:h-24 transition-transform duration-300 hover:scale-110"
              alt="MediPredict Logo"
            />
          </div>
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 dark:text-gray-100 mb-3 animate-in fade-in slide-in-from-bottom-2 duration-500 delay-100">
            MediPredict
          </h1>
          <p className="text-base sm:text-lg text-gray-600 dark:text-gray-400 px-4 animate-in fade-in slide-in-from-bottom-3 duration-500 delay-200">
            Enter your information to get an estimated insurance cost prediction
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl dark:shadow-2xl p-6 sm:p-8 mb-8 transition-all duration-300 hover:shadow-2xl dark:hover:shadow-3xl">
          <form onSubmit={handleSubmit} className="space-y-6 sm:space-y-8">
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
              <h2 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 sm:mb-6 flex items-center transition-colors duration-200">
                <User className="h-5 w-5 sm:h-6 sm:w-6 mr-2 text-blue-600 dark:text-blue-400 transition-transform duration-200 hover:scale-110" />
                Personal Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                <div className="space-y-2 transition-all duration-200 hover:translate-y-[-2px]">
                  <Label
                    htmlFor="age"
                    className="text-gray-700 dark:text-gray-300"
                  >
                    Age
                  </Label>
                  <Input
                    id="age"
                    name="age"
                    type="number"
                    placeholder="e.g., 30"
                    value={formData.age}
                    onChange={handleInputChange}
                    className={errors.age ? "border-red-500" : ""}
                  />
                  {errors.age && (
                    <p className="text-sm text-red-500 dark:text-red-400 animate-in fade-in slide-in-from-top-1 duration-200">
                      {errors.age}
                    </p>
                  )}
                </div>

                <div className="space-y-2 transition-all duration-200 hover:translate-y-[-2px]">
                  <Label
                    htmlFor="sex"
                    className="text-gray-700 dark:text-gray-300"
                  >
                    Sex
                  </Label>
                  <Select
                    id="sex"
                    name="sex"
                    value={formData.sex}
                    onChange={(value) => handleSelectChange("sex", value)}
                    options={[
                      { value: "male", label: "Male" },
                      { value: "female", label: "Female" },
                    ]}
                    placeholder="Select sex"
                    hasError={!!errors.sex}
                  />
                  {errors.sex && (
                    <p className="text-sm text-red-500 dark:text-red-400 animate-in fade-in slide-in-from-top-1 duration-200">
                      {errors.sex}
                    </p>
                  )}
                </div>
              </div>
            </div>

            <div className="animate-in fade-in slide-in-from-bottom-3 duration-500 delay-100">
              <h2 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 sm:mb-6 flex items-center transition-colors duration-200">
                <TrendingUp className="h-5 w-5 sm:h-6 sm:w-6 mr-2 text-blue-600 dark:text-blue-400 transition-transform duration-200 hover:scale-110" />
                Health Metrics
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                <div className="space-y-2 transition-all duration-200 hover:translate-y-[-2px]">
                  <Label
                    htmlFor="bmi"
                    className="text-gray-700 dark:text-gray-300"
                  >
                    BMI (Body Mass Index)
                  </Label>
                  <Input
                    id="bmi"
                    name="bmi"
                    type="number"
                    step="0.1"
                    placeholder="e.g., 25.5"
                    value={formData.bmi}
                    onChange={handleInputChange}
                    className={errors.bmi ? "border-red-500" : ""}
                  />
                  {errors.bmi && (
                    <p className="text-sm text-red-500 dark:text-red-400 animate-in fade-in slide-in-from-top-1 duration-200">
                      {errors.bmi}
                    </p>
                  )}
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    BMI = weight(kg) / height(m)²
                  </p>
                </div>

                <div className="space-y-2 transition-all duration-200 hover:translate-y-[-2px]">
                  <Label
                    htmlFor="smoker"
                    className="text-gray-700 dark:text-gray-300 flex items-center"
                  >
                    <Cigarette className="h-4 w-4 mr-1" />
                    Smoking Status
                  </Label>
                  <Select
                    id="smoker"
                    name="smoker"
                    value={formData.smoker}
                    onChange={(value) => handleSelectChange("smoker", value)}
                    options={[
                      { value: "yes", label: "Yes" },
                      { value: "no", label: "No" },
                    ]}
                    placeholder="Select status"
                    hasError={!!errors.smoker}
                  />
                  {errors.smoker && (
                    <p className="text-sm text-red-500 dark:text-red-400 animate-in fade-in slide-in-from-top-1 duration-200">
                      {errors.smoker}
                    </p>
                  )}
                </div>
              </div>
            </div>

            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 delay-200">
              <h2 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 sm:mb-6 flex items-center transition-colors duration-200">
                <Users className="h-5 w-5 sm:h-6 sm:w-6 mr-2 text-blue-600 dark:text-blue-400 transition-transform duration-200 hover:scale-110" />
                Family & Location
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                <div className="space-y-2 transition-all duration-200 hover:translate-y-[-2px]">
                  <Label
                    htmlFor="children"
                    className="text-gray-700 dark:text-gray-300"
                  >
                    Number of Children
                  </Label>
                  <Input
                    id="children"
                    name="children"
                    type="number"
                    placeholder="e.g., 2"
                    value={formData.children}
                    onChange={handleInputChange}
                    className={errors.children ? "border-red-500" : ""}
                  />
                  {errors.children && (
                    <p className="text-sm text-red-500 dark:text-red-400 animate-in fade-in slide-in-from-top-1 duration-200">
                      {errors.children}
                    </p>
                  )}
                </div>

                <div className="space-y-2 transition-all duration-200 hover:translate-y-[-2px]">
                  <Label
                    htmlFor="region"
                    className="text-gray-700 dark:text-gray-300 flex items-center"
                  >
                    <MapPin className="h-4 w-4 mr-1" />
                    Region
                  </Label>
                  <Select
                    id="region"
                    name="region"
                    value={formData.region}
                    onChange={(value) => handleSelectChange("region", value)}
                    options={[
                      { value: "northeast", label: "Northeast" },
                      { value: "northwest", label: "Northwest" },
                      { value: "southeast", label: "Southeast" },
                      { value: "southwest", label: "Southwest" },
                    ]}
                    placeholder="Select region"
                    hasError={!!errors.region}
                  />
                  {errors.region && (
                    <p className="text-sm text-red-500 dark:text-red-400 animate-in fade-in slide-in-from-top-1 duration-200">
                      {errors.region}
                    </p>
                  )}
                </div>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 pt-6 border-t dark:border-gray-700">
              <Button
                type="submit"
                className="flex-1 h-11 sm:h-10"
                disabled={isLoading}
              >
                {isLoading ? "Calculating..." : "Calculate Insurance Cost"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={handleReset}
                disabled={isLoading}
                className="h-11 sm:h-10"
              >
                Reset Form
              </Button>
            </div>
          </form>
        </div>

        {prediction && (
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl dark:shadow-2xl p-6 sm:p-8 animate-in fade-in slide-in-from-bottom-4 duration-500 transition-all hover:shadow-2xl dark:hover:shadow-3xl">
            <h2 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 text-center animate-in fade-in slide-in-from-top-2 duration-300">
              Predicted Insurance Cost
            </h2>

            <div className="mb-6 sm:mb-8 bg-gradient-to-br from-amber-50 to-yellow-100 dark:from-amber-900/30 dark:to-yellow-900/30 rounded-xl p-6 sm:p-8 border-2 border-amber-300 dark:border-amber-700 relative overflow-hidden transition-all duration-300 hover:scale-[1.02] hover:shadow-lg">
              <div className="absolute top-2 right-2 animate-in spin-in duration-500 delay-200">
                <Crown className="h-8 w-8 sm:h-10 sm:w-10 text-amber-600 dark:text-amber-400" />
              </div>
              <div className="flex items-center gap-2 mb-3 animate-in fade-in slide-in-from-left-3 duration-400">
                <Star className="h-5 w-5 sm:h-6 sm:w-6 text-amber-600 dark:text-amber-400 fill-amber-600 dark:fill-amber-400 animate-pulse" />
                <h3 className="text-lg sm:text-xl font-bold text-amber-900 dark:text-amber-100">
                  Ensemble Model Prediction
                </h3>
              </div>
              <p className="text-4xl sm:text-5xl font-bold text-amber-900 dark:text-amber-100 mb-3 animate-in zoom-in duration-500 delay-100">
                $
                {prediction.ensemble.toLocaleString("en-US", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </p>
              <div className="flex items-center gap-2 mt-4 animate-in fade-in slide-in-from-bottom-2 duration-400 delay-150">
                <Sparkles className="h-5 w-5 text-amber-600 dark:text-amber-400 animate-pulse" />
                <p className="text-base font-semibold text-amber-800 dark:text-amber-200">
                  Model Accuracy:{" "}
                  {prediction.modelAccuracies.ensemble.toFixed(2)}%
                </p>
              </div>
              <p className="text-sm text-amber-700 dark:text-amber-300 mt-3 animate-in fade-in duration-500 delay-200">
                This prediction combines Random Forest, Gradient Boosting, and
                LightGBM using an ensemble stacking approach for maximum
                accuracy and reliability.
              </p>
              <p className="text-xs text-amber-600 dark:text-amber-400 mt-2 italic animate-in fade-in duration-500 delay-300">
                Estimated annual insurance cost
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
