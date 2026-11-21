import Link from "next/link";
import { Nav } from "./nav";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-dvh bg-surface">
      <a
        href="#main"
        className="sr-only focus-visible:not-sr-only focus-visible:fixed focus-visible:left-4 focus-visible:top-4 focus-visible:z-50 focus-visible:rounded-sm focus-visible:bg-slate-800 focus-visible:px-3 focus-visible:py-2 focus-visible:text-sm focus-visible:text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        İçeriğe geç
      </a>
      <header className="border-b border-slate-800">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <Link
            href="/"
            className="rounded-sm text-sm font-semibold tracking-wide text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            AI Scalping Studio
          </Link>
          <Nav />
        </div>
      </header>
      <main id="main" tabIndex={-1} className="mx-auto max-w-7xl px-4 py-6">
        {children}
      </main>
    </div>
  );
}
