import type { Metadata } from "next";

// The page is a client component and cannot export metadata itself; the route
// layout names the document so WCAG 2.4.2 (Page Titled) holds on direct loads.
export const metadata: Metadata = {
  title: "Paper Trading — AI Scalping Studio",
};

export default function PaperLayout({ children }: { children: React.ReactNode }) {
  return children;
}
