import type { Metadata, Viewport } from "next";
import { AppShell } from "@/components/app-shell";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Scalping Studio",
  description: "Sentetik veriyle çalışan scalping simülatörü ve eğitim paneli.",
};

export const viewport: Viewport = {
  themeColor: "#0B0F17",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="tr" className="dark">
      <body className="bg-[#0B0F17] text-slate-200 antialiased">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
