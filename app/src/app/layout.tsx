// src/app/layout.tsx
import "../styles/globals.css";

export const metadata = {
  title: "Chess UI (dev)",
  description: "Minimal chess UI for engine testing",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 min-h-screen">
        {/* <header className="p-4 border-b bg-white/60 sticky top-0 z-10">
          <div className="max-w-4xl mx-auto">🧩 Chess UI — engine test</div>
        </header> */}
        <main className="max-w-4xl mx-auto p-6">{children}</main>
      </body>
    </html>
  );
}
