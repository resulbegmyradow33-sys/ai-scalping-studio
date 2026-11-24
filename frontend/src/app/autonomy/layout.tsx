import type { Metadata } from "next";
import type { ReactNode } from "react";

/** A distinct document title for the autonomy route. */
export const metadata: Metadata = {
  title: "Otonomi — AI Scalping Studio",
};

export default function AutonomyLayout({ children }: { children: ReactNode }) {
  return children;
}
