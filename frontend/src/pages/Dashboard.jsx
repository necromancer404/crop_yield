import { useEffect, useMemo, useState } from "react";
import { CropYieldBarChart, YieldVsRainfallChart } from "../components/ChartsSection.jsx";
import Field from "../components/Field.jsx";
import Panel from "../components/Panel.jsx";
import {
  fetchHealth,
  fetchYieldByCrop,
  fetchYieldVsRainfall,
  predictCrop,
  predictYield,
  recommendFull,
} from "../services/api.js";

const inputStyle = {
  padding: "10px 12px",
  borderRadius: 10,
  border: "1px solid var(--border)",
  background: "var(--panel-2)",
  color: "var(--text)",
};

const btnPrimary = {
  padding: "12px 18px",
  borderRadius: 12,
  border: "none",
  background: "linear-gradient(135deg, var(--accent), var(--accent-dim))",
  color: "#0b0f12",
  fontWeight: 700,
  cursor: "pointer",
};

const btnGhost = {
  ...btnPrimary,
  background: "transparent",
  color: "var(--text)",
  border: "1px solid var(--border)",
};

const defaults = {
  N: 85,
  P: 45,
  K: 42,
  temperature: 24,
  humidity: 72,
  ph: 6.5,
  rainfall: 210,
  State: "Karnataka",
  District: "BANGALORE RURAL",
  Crop: "Rice",
  Season: "Kharif",
  Temperature: 28,
  Humidity: 65,
  Soil_Moisture: 48,
  Area: 2.5,
  Crop_Year: 2014,
};

export default function Dashboard() {
  const [form, setForm] = useState(defaults);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState("");
  const [cropResult, setCropResult] = useState(null);
  const [yieldResult, setYieldResult] = useState(null);
  const [recResult, setRecResult] = useState(null);
  const [scatter, setScatter] = useState([]);
  const [bars, setBars] = useState([]);
  const [chartErr, setChartErr] = useState({ s: "", b: "" });
  const [chartsLoading, setChartsLoading] = useState(true);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "error" }));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setChartsLoading(true);
    Promise.all([fetchYieldVsRainfall(), fetchYieldByCrop()])
      .then(([s, b]) => {
        if (cancelled) return;
        setScatter(s.points ?? []);
        setBars(b.points ?? []);
        setChartErr({ s: "", b: "" });
      })
      .catch((e) => {
        if (!cancelled) setChartErr({ s: String(e.message), b: String(e.message) });
      })
      .finally(() => {
        if (!cancelled) setChartsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const onChange = (key) => (e) => {
    const v = e.target.type === "number" ? parseFloat(e.target.value) : e.target.value;
    setForm((f) => ({ ...f, [key]: Number.isFinite(v) ? v : e.target.value }));
  };

  const payloadCrop = useMemo(
    () => ({
      N: form.N,
      P: form.P,
      K: form.K,
      temperature: form.temperature,
      humidity: form.humidity,
      ph: form.ph,
      rainfall: form.rainfall,
    }),
    [form]
  );

  const payloadYield = useMemo(
    () => ({
      State: form.State,
      District: form.District,
      Crop: form.Crop,
      Season: form.Season,
      Temperature: form.Temperature,
      Humidity: form.Humidity,
      Soil_Moisture: form.Soil_Moisture,
      Area: form.Area,
      Crop_Year: form.Crop_Year || undefined,
    }),
    [form]
  );

  const runCrop = async () => {
    setError("");
    setLoading("crop");
    try {
      setCropResult(await predictCrop(payloadCrop));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading("");
    }
  };

  const runYield = async () => {
    setError("");
    setLoading("yield");
    try {
      setYieldResult(await predictYield(payloadYield));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading("");
    }
  };

  const runRecommend = async () => {
    setError("");
    setLoading("rec");
    try {
      setRecResult(await recommendFull({ ...payloadCrop, ...payloadYield }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading("");
    }
  };

  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "1.5rem 1.25rem 3rem" }}>
      <header style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ margin: 0, fontSize: "1.75rem", letterSpacing: -0.5 }}>
          Smart crop yield & recommendation
        </h1>
        <p style={{ margin: "0.35rem 0 0", color: "var(--muted)", maxWidth: 640 }}>
          Classifier suggests a crop from soil NPK, pH, and weather; regressors estimate yield from farm
          records. Train models from the backend before calling the API.
        </p>
        <p style={{ marginTop: "0.75rem", fontSize: 13, color: health?.models_loaded ? "var(--accent)" : "var(--warn)" }}>
          API: {health ? (health.models_loaded ? "models loaded" : "models missing — run training") : "checking…"}
        </p>
      </header>

      <div className="dashboard-grid">
        <Panel title="Inputs">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.85rem" }}>
            <Field label="N (nitrogen)">
              <input type="number" style={inputStyle} value={form.N} onChange={onChange("N")} />
            </Field>
            <Field label="P (phosphorus)">
              <input type="number" style={inputStyle} value={form.P} onChange={onChange("P")} />
            </Field>
            <Field label="K (potassium)">
              <input type="number" style={inputStyle} value={form.K} onChange={onChange("K")} />
            </Field>
            <Field label="Temperature (°C)">
              <input type="number" style={inputStyle} value={form.temperature} onChange={onChange("temperature")} />
            </Field>
            <Field label="Humidity (%)">
              <input type="number" style={inputStyle} value={form.humidity} onChange={onChange("humidity")} />
            </Field>
            <Field label="pH">
              <input type="number" step="0.1" style={inputStyle} value={form.ph} onChange={onChange("ph")} />
            </Field>
            <Field label="Rainfall (mm)">
              <input type="number" style={inputStyle} value={form.rainfall} onChange={onChange("rainfall")} />
            </Field>
            <Field label="State">
              <input type="text" style={inputStyle} value={form.State} onChange={onChange("State")} />
            </Field>
            <Field label="District">
              <input type="text" style={inputStyle} value={form.District} onChange={onChange("District")} />
            </Field>
            <Field label="Current crop">
              <input type="text" style={inputStyle} value={form.Crop} onChange={onChange("Crop")} />
            </Field>
            <Field label="Season">
              <input type="text" style={inputStyle} value={form.Season} onChange={onChange("Season")} />
            </Field>
            <Field label="Farm temperature (°C)">
              <input type="number" style={inputStyle} value={form.Temperature} onChange={onChange("Temperature")} />
            </Field>
            <Field label="Farm humidity (%)">
              <input type="number" style={inputStyle} value={form.Humidity} onChange={onChange("Humidity")} />
            </Field>
            <Field label="Soil moisture (%)">
              <input type="number" style={inputStyle} value={form.Soil_Moisture} onChange={onChange("Soil_Moisture")} />
            </Field>
            <Field label="Area (ha)">
              <input type="number" step="0.1" style={inputStyle} value={form.Area} onChange={onChange("Area")} />
            </Field>
            <Field label="Crop year (optional)" hint="Used if the trained model includes year.">
              <input type="number" style={inputStyle} value={form.Crop_Year} onChange={onChange("Crop_Year")} />
            </Field>
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: "1rem" }}>
            <button type="button" style={btnGhost} disabled={loading === "crop"} onClick={runCrop}>
              Predict crop
            </button>
            <button type="button" style={btnGhost} disabled={loading === "yield"} onClick={runYield}>
              Predict yield
            </button>
            <button type="button" style={btnPrimary} disabled={loading === "rec"} onClick={runRecommend}>
              Full recommend
            </button>
          </div>

          {error ? (
            <p style={{ color: "var(--danger)", marginTop: "0.85rem", fontSize: 14 }}>{error}</p>
          ) : null}
        </Panel>

        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <Panel title="Results">
            {cropResult ? (
              <p style={{ margin: "0 0 0.5rem" }}>
                <strong>Recommended crop:</strong> {cropResult.recommended_crop}
              </p>
            ) : (
              <p style={{ margin: 0, color: "var(--muted)" }}>Run predict crop or full recommend.</p>
            )}
            {yieldResult ? (
              <div style={{ marginTop: "0.5rem" }}>
                <p style={{ margin: "0.25rem 0" }}>
                  <strong>Predicted yield (ensemble):</strong> {yieldResult.predicted_yield.toFixed(3)}
                </p>
                <p style={{ margin: "0.25rem 0", fontSize: 13, color: "var(--muted)" }}>
                  RF: {yieldResult.rf_contribution?.toFixed(3)} · XGB: {yieldResult.xgb_contribution?.toFixed(3)}
                </p>
              </div>
            ) : null}
            {recResult ? (
              <div style={{ marginTop: "0.75rem", borderTop: "1px solid var(--border)", paddingTop: "0.75rem" }}>
                <p style={{ margin: "0.25rem 0" }}>
                  <strong>Classifier:</strong> {recResult.recommended_crop}{" "}
                  <span style={{ color: "var(--muted)" }}>(yield row: {recResult.mapped_crop_for_yield})</span>
                </p>
                <p style={{ margin: "0.25rem 0" }}>
                  <strong>Yield (your crop):</strong> {recResult.predicted_yield.toFixed(3)}
                </p>
                <p style={{ margin: "0.25rem 0" }}>
                  <strong>Yield (recommended mapping):</strong>{" "}
                  {recResult.predicted_yield_for_recommended?.toFixed(3)}
                </p>
                <div style={{ marginTop: "0.5rem" }}>
                  <strong style={{ fontSize: 13 }}>Summary</strong>
                  <ul style={{ margin: "0.35rem 0 0", paddingLeft: "1.1rem", color: "var(--muted)", fontSize: 13 }}>
                    {recResult.suggestions?.map((s) => (
                      <li key={s} style={{ marginBottom: 4 }}>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
                <div style={{ marginTop: "0.5rem" }}>
                  <strong style={{ fontSize: 13 }}>Fertilizer / soil</strong>
                  <ul style={{ margin: "0.35rem 0 0", paddingLeft: "1.1rem", color: "var(--muted)", fontSize: 13 }}>
                    {recResult.fertilizer_suggestions?.map((s) => (
                      <li key={s} style={{ marginBottom: 4 }}>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
                <div style={{ marginTop: "0.5rem" }}>
                  <strong style={{ fontSize: 13 }}>Yield insights</strong>
                  <ul style={{ margin: "0.35rem 0 0", paddingLeft: "1.1rem", color: "var(--muted)", fontSize: 13 }}>
                    {recResult.yield_insights?.map((s) => (
                      <li key={s} style={{ marginBottom: 4 }}>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : null}
          </Panel>

          <Panel title="Yield vs rainfall (matched crops)">
            <YieldVsRainfallChart
              data={scatter}
              loading={chartsLoading}
              error={chartErr.s}
            />
          </Panel>

          <Panel title="Crop comparison (avg yield)">
            <CropYieldBarChart data={bars} loading={chartsLoading} error={chartErr.b} />
          </Panel>
        </div>
      </div>
    </div>
  );
}
