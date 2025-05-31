// frontend/app/layout.tsx
import React from "react";
import Link from "next/link";       // ← import Next’s Link

export const metadata = {
  title: "My App",
  description: "A Next.js + Django integrated project",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Add any <meta> or <link> tags here if needed */}
      </head>
      <body>
        <header style={{ padding: "1rem", borderBottom: "1px solid #ddd" }}>
          <nav style={{ display: "flex", gap: "1rem" }}>
            {/* Replace <a> with <Link> */}
            <Link href="/">Home</Link>
            <Link href="/register">Register</Link>
            <Link href="/login">Login</Link>
          </nav>
        </header>

        <main style={{ padding: "2rem" }}>{children}</main>

        <footer
          style={{
            textAlign: "center",
            padding: "1rem",
            borderTop: "1px solid #ddd",
          }}
        >
          &copy; 2025 MyCompany
        </footer>
      </body>
    </html>
  );
}
