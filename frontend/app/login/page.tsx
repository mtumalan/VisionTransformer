"use client";

import { useState } from "react";

export default function LoginPage() {
  // Form state
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [success, setSuccess] = useState<string | null>(null);

  // Typed as React.FormEvent<HTMLFormElement>—no more `any`
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000"}/api/users/login/`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ username, password }),
        }
      );

      if (res.ok) {
        setSuccess("Login successful!");
        // You might store a token or redirect here.
      } else {
        const data = await res.json();
        setError(JSON.stringify(data, null, 2));
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError("Network error: " + err.message);
      } else {
        setError("An unknown error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: 420,
        margin: "2rem auto",
        padding: "1rem",
        border: "1px solid #eaeaea",
        borderRadius: 8,
      }}
    >
      <h1>Login</h1>
      {error && (
        <pre
          style={{
            background: "#ffe8e8",
            color: "#b00020",
            padding: "0.75rem",
            borderRadius: 4,
          }}
        >
          {error}
        </pre>
      )}
      {success && (
        <p
          style={{
            background: "#e6ffe6",
            color: "#006600",
            padding: "0.75rem",
            borderRadius: 4,
          }}
        >
          {success}
        </p>
      )}
      <form
        onSubmit={handleSubmit}
        style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
      >
        <label style={{ display: "flex", flexDirection: "column" }}>
          Username
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={{ padding: "0.5rem", fontSize: "1rem", marginTop: "0.25rem" }}
          />
        </label>

        <label style={{ display: "flex", flexDirection: "column" }}>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ padding: "0.5rem", fontSize: "1rem", marginTop: "0.25rem" }}
          />
        </label>

        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "0.75rem",
            fontSize: "1rem",
            background: loading ? "#ccc" : "#0070f3",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: loading ? "default" : "pointer",
          }}
        >
          {loading ? "Logging in…" : "Log In"}
        </button>
      </form>

      <p style={{ marginTop: "1rem", textAlign: "center" }}>
        Don’t have an account?{" "}
        <a href="/register" style={{ color: "#0070f3", textDecoration: "underline" }}>
          Register
        </a>
      </p>
    </div>
  );
}
