export default function Panel({ title, children, style }) {
  return (
    <section
      style={{
        background: "var(--panel)",
        border: "1px solid var(--border)",
        borderRadius: 14,
        padding: "1.1rem 1.25rem",
        ...style,
      }}
    >
      {title ? (
        <h2 style={{ margin: "0 0 0.75rem", fontSize: "1.05rem", fontWeight: 700 }}>{title}</h2>
      ) : null}
      {children}
    </section>
  );
}
