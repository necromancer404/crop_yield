const BASE = import.meta.env.VITE_API_URL ?? "";

async function parseError(res) {
  const text = await res.text();
  try {
    const j = JSON.parse(text);
    return j.detail ?? text;
  } catch {
    return text;
  }
}

export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function predictCrop(body) {
  const res = await fetch(`${BASE}/predict-crop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function predictYield(body) {
  const res = await fetch(`${BASE}/predict-yield`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function recommendFull(body) {
  const res = await fetch(`${BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchYieldVsRainfall() {
  const res = await fetch(`${BASE}/analytics/yield-vs-rainfall`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchYieldByCrop() {
  const res = await fetch(`${BASE}/analytics/yield-by-crop`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}
