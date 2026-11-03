import React from "react";

export default function StatusPill({ status, label }) {
  return <span className={`pill pill-${status || "pending"}`}>{label || status || "pending"}</span>;
}
