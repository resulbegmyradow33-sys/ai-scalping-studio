import type { Metadata } from "next";

// The page is a client component and cannot export metadata itself; the route
// layout owns the document title, matching the rest of the dashboard.
export const metadata: Metadata = {
  title: "Backtesting — AI Scalping Studio",
};

export default function BacktestingLayout({ children }: { children: React.ReactNode }) {
  return children;
}
