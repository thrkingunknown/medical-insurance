import * as React from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Activity,
  User,
  TrendingUp,
  Users,
  Cigarette,
  MapPin,
  Award,
  Zap,
  Target,
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
  linearRegression: number;
  randomForest: number;
  gradientBoosting: number;
  xgboost: number;
  lightgbm: number;
  ensemble: number;
  modelAccuracies: {
    lr: number;
    rf: number;
    gb: number;
    xgb: number;
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
      const response = await fetch("http://localhost:8000/predict", {
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
        linearRegression: data.linearRegression,
        randomForest: data.randomForest,
        gradientBoosting: data.gradientBoosting,
        xgboost: data.xgboost,
        lightgbm: data.lightgbm,
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <Activity className="h-16 w-16 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            Medical Insurance Cost Predictor
          </h1>
          <p className="text-lg text-gray-600">
            Enter your information to get an estimated insurance cost prediction
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          <form onSubmit={handleSubmit} className="space-y-8">
            <div>
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <User className="h-6 w-6 mr-2 text-blue-600" />
                Personal Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="age" className="text-gray-700">
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
                    <p className="text-sm text-red-500">{errors.age}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="sex" className="text-gray-700">
                    Sex
                  </Label>
                  <select
                    id="sex"
                    name="sex"
                    value={formData.sex}
                    onChange={handleInputChange}
                    className={`flex h-10 w-full rounded-md border ${
                      errors.sex ? "border-red-500" : "border-input"
                    } bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2`}
                  >
                    <option value="">Select sex</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                  {errors.sex && (
                    <p className="text-sm text-red-500">{errors.sex}</p>
                  )}
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <TrendingUp className="h-6 w-6 mr-2 text-blue-600" />
                Health Metrics
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="bmi" className="text-gray-700">
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
                    <p className="text-sm text-red-500">{errors.bmi}</p>
                  )}
                  <p className="text-xs text-gray-500">
                    BMI = weight(kg) / height(m)²
                  </p>
                </div>

                <div className="space-y-2">
                  <Label
                    htmlFor="smoker"
                    className="text-gray-700 flex items-center"
                  >
                    <Cigarette className="h-4 w-4 mr-1" />
                    Smoking Status
                  </Label>
                  <select
                    id="smoker"
                    name="smoker"
                    value={formData.smoker}
                    onChange={handleInputChange}
                    className={`flex h-10 w-full rounded-md border ${
                      errors.smoker ? "border-red-500" : "border-input"
                    } bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2`}
                  >
                    <option value="">Select status</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                  {errors.smoker && (
                    <p className="text-sm text-red-500">{errors.smoker}</p>
                  )}
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <Users className="h-6 w-6 mr-2 text-blue-600" />
                Family & Location
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="children" className="text-gray-700">
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
                    <p className="text-sm text-red-500">{errors.children}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label
                    htmlFor="region"
                    className="text-gray-700 flex items-center"
                  >
                    <MapPin className="h-4 w-4 mr-1" />
                    Region
                  </Label>
                  <select
                    id="region"
                    name="region"
                    value={formData.region}
                    onChange={handleInputChange}
                    className={`flex h-10 w-full rounded-md border ${
                      errors.region ? "border-red-500" : "border-input"
                    } bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2`}
                  >
                    <option value="">Select region</option>
                    <option value="northeast">Northeast</option>
                    <option value="northwest">Northwest</option>
                    <option value="southeast">Southeast</option>
                    <option value="southwest">Southwest</option>
                  </select>
                  {errors.region && (
                    <p className="text-sm text-red-500">{errors.region}</p>
                  )}
                </div>
              </div>
            </div>

            <div className="flex gap-4 pt-6 border-t">
              <Button type="submit" className="flex-1" disabled={isLoading}>
                {isLoading ? "Calculating..." : "Calculate Insurance Cost"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={handleReset}
                disabled={isLoading}
              >
                Reset Form
              </Button>
            </div>
          </form>
        </div>

        {prediction && (
          <div className="bg-white rounded-2xl shadow-xl p-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6 text-center">
              Predicted Insurance Costs
            </h2>

            <div className="mb-8 bg-gradient-to-br from-amber-50 to-yellow-100 rounded-xl p-6 border-2 border-amber-300 relative overflow-hidden">
              <div className="absolute top-2 right-2">
                <Crown className="h-8 w-8 text-amber-600" />
              </div>
              <div className="flex items-center gap-2 mb-2">
                <Star className="h-5 w-5 text-amber-600 fill-amber-600" />
                <h3 className="text-lg font-bold text-amber-900">
                  Best Model: Ensemble Stacking
                </h3>
              </div>
              <p className="text-4xl font-bold text-amber-900 mb-2">
                $
                {prediction.ensemble.toLocaleString("en-US", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </p>
              <div className="flex items-center gap-2 mt-3">
                <Sparkles className="h-4 w-4 text-amber-600" />
                <p className="text-sm font-semibold text-amber-800">
                  Accuracy: {prediction.modelAccuracies.ensemble.toFixed(2)}%
                </p>
              </div>
              <p className="text-xs text-amber-700 mt-2">
                Combines Random Forest, Gradient Boosting & LightGBM for maximum
                accuracy
              </p>
            </div>

            <h3 className="text-lg font-semibold text-gray-700 mb-4">
              All Model Predictions
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-5 border border-blue-200 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-blue-800">
                    Ridge Regression
                  </h3>
                  <Target className="h-4 w-4 text-blue-600" />
                </div>
                <p className="text-2xl font-bold text-blue-900 mb-2">
                  $
                  {prediction.linearRegression.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
                <div className="flex items-center justify-between">
                  <p className="text-xs text-blue-600">per year</p>
                  <div className="flex items-center gap-1">
                    <Award className="h-3 w-3 text-blue-600" />
                    <p className="text-xs font-semibold text-blue-700">
                      {prediction.modelAccuracies.lr.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-5 border border-purple-200 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-purple-800">
                    Random Forest
                  </h3>
                  <Target className="h-4 w-4 text-purple-600" />
                </div>
                <p className="text-2xl font-bold text-purple-900 mb-2">
                  $
                  {prediction.randomForest.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
                <div className="flex items-center justify-between">
                  <p className="text-xs text-purple-600">per year</p>
                  <div className="flex items-center gap-1">
                    <Award className="h-3 w-3 text-purple-600" />
                    <p className="text-xs font-semibold text-purple-700">
                      {prediction.modelAccuracies.rf.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-5 border border-green-200 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-green-800">
                    Gradient Boosting
                  </h3>
                  <Zap className="h-4 w-4 text-green-600" />
                </div>
                <p className="text-2xl font-bold text-green-900 mb-2">
                  $
                  {prediction.gradientBoosting.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
                <div className="flex items-center justify-between">
                  <p className="text-xs text-green-600">per year</p>
                  <div className="flex items-center gap-1">
                    <Award className="h-3 w-3 text-green-600" />
                    <p className="text-xs font-semibold text-green-700">
                      {prediction.modelAccuracies.gb.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl p-5 border border-orange-200 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-orange-800">
                    XGBoost
                  </h3>
                  <Zap className="h-4 w-4 text-orange-600" />
                </div>
                <p className="text-2xl font-bold text-orange-900 mb-2">
                  $
                  {prediction.xgboost.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
                <div className="flex items-center justify-between">
                  <p className="text-xs text-orange-600">per year</p>
                  <div className="flex items-center gap-1">
                    <Award className="h-3 w-3 text-orange-600" />
                    <p className="text-xs font-semibold text-orange-700">
                      {prediction.modelAccuracies.xgb.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-teal-50 to-teal-100 rounded-xl p-5 border border-teal-200 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-teal-800">
                    LightGBM
                  </h3>
                  <Zap className="h-4 w-4 text-teal-600" />
                </div>
                <p className="text-2xl font-bold text-teal-900 mb-2">
                  $
                  {prediction.lightgbm.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
                <div className="flex items-center justify-between">
                  <p className="text-xs text-teal-600">per year</p>
                  <div className="flex items-center gap-1">
                    <Award className="h-3 w-3 text-teal-600" />
                    <p className="text-xs font-semibold text-teal-700">
                      {prediction.modelAccuracies.lgb.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-amber-50 to-amber-100 rounded-xl p-5 border-2 border-amber-300 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-amber-800 flex items-center gap-1">
                    Ensemble
                    <Star className="h-3 w-3 text-amber-600 fill-amber-600" />
                  </h3>
                  <Crown className="h-4 w-4 text-amber-600" />
                </div>
                <p className="text-2xl font-bold text-amber-900 mb-2">
                  $
                  {prediction.ensemble.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
                <div className="flex items-center justify-between">
                  <p className="text-xs text-amber-600">per year</p>
                  <div className="flex items-center gap-1">
                    <Award className="h-3 w-3 text-amber-600" />
                    <p className="text-xs font-semibold text-amber-700">
                      {prediction.modelAccuracies.ensemble.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-sm text-gray-600 text-center">
                <strong>Note:</strong> These are estimated predictions based on
                6 different machine learning models. The Ensemble model combines
                the best performing models for maximum accuracy. Actual
                insurance costs may vary based on additional factors.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
