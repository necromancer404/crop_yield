export default function Field({ label, hint, children }) {
  return (
    <label
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        fontSize: 13,
        color: "var(--muted)",
      }}
    >
      <span style={{ fontWeight: 600, color: "var(--text)" }}>{label}</span>
      {children}
      {hint ? <span style={{ fontSize: 12, opacity: 0.85 }}>{hint}</span> : null}
    </label>
  );
}
