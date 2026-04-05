import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const axisStyle = { fill: "var(--muted)", fontSize: 11 };
const gridStyle = { stroke: "#2f3d4d", strokeDasharray: "3 6" };

export function YieldVsRainfallChart({ data, loading, error }) {
  if (loading) return <p style={{ color: "var(--muted)" }}>Loading chart…</p>;
  if (error) return <p style={{ color: "var(--danger)" }}>{error}</p>;
  if (!data?.length) return <p style={{ color: "var(--muted)" }}>No data.</p>;

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 12, right: 12, left: 0, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 6" stroke="#2f3d4d" />
          <XAxis
            type="number"
            dataKey="rainfall"
            name="Rainfall"
            unit=" mm"
            tick={axisStyle}
            label={{ value: "Avg rainfall (mm)", position: "bottom", offset: 24, fill: "#8fa3b8" }}
          />
          <YAxis
            type="number"
            dataKey="avg_yield"
            name="Yield"
            tick={axisStyle}
            width={60}
            label={{ value: "Avg yield", angle: -90, position: "insideLeft", fill: "#8fa3b8" }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ background: "#1a222c", border: "1px solid #2f3d4d", borderRadius: 8 }}
            formatter={(value, name) => [value, name === "avg_yield" ? "Avg yield" : name]}
            labelFormatter={(_, p) => (p?.[0]?.payload?.crop ? `Crop: ${p[0].payload.crop}` : "")}
          />
          <Legend />
          <Scatter name="Crops" data={data} fill="#3dd68c" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CropYieldBarChart({ data, loading, error }) {
  if (loading) return <p style={{ color: "var(--muted)" }}>Loading chart…</p>;
  if (error) return <p style={{ color: "var(--danger)" }}>{error}</p>;
  if (!data?.length) return <p style={{ color: "var(--muted)" }}>No data.</p>;

  const short = data.map((d) => ({
    ...d,
    label: d.crop?.length > 14 ? `${d.crop.slice(0, 12)}…` : d.crop,
  }));

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        <BarChart data={short} margin={{ top: 8, right: 12, left: 0, bottom: 56 }}>
          <CartesianGrid {...gridStyle} />
          <XAxis dataKey="label" interval={0} angle={-28} textAnchor="end" height={70} tick={axisStyle} />
          <YAxis tick={axisStyle} width={56} />
          <Tooltip
            contentStyle={{ background: "#1a222c", border: "1px solid #2f3d4d", borderRadius: 8 }}
            formatter={(v) => [v, "Avg yield"]}
            labelFormatter={(_, i) => `Crop: ${data[i]?.crop ?? ""}`}
          />
          <Legend />
          <Bar dataKey="avg_yield" name="Avg yield" fill="#f4b860" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
