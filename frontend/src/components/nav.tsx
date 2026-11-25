"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "Simülatör" },
  { href: "/training", label: "Eğitim" },
  { href: "/labs/ml", label: "ML Laboratuvarı" },
  { href: "/performance", label: "Performans" },
  { href: "/backtesting", label: "Backtesting" },
  { href: "/paper", label: "Paper Trading" },
  { href: "/autonomy", label: "Otonomi" },
  { href: "/intelligence", label: "Piyasa Zekâsı" },
  { href: "/strategies", label: "Strateji Zekâsı" },
  { href: "/data", label: "Veri" },
] as const;

export function Nav() {
  const pathname = usePathname();

  return (
    <nav aria-label="Ana gezinme">
      <ul className="flex flex-wrap items-center gap-1">
        {items.map((item) => {
          const active =
            pathname === item.href || pathname.startsWith(`${item.href}/`);

          return (
            <li key={item.href}>
              <Link
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={[
                  "inline-block rounded-sm border-b-2 px-3 py-2 text-sm transition-colors",
                  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
                  active
                    ? "border-accent bg-slate-800/60 font-medium text-slate-100"
                    : "border-transparent text-slate-400 hover:text-slate-100",
                ].join(" ")}
              >
                {item.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
