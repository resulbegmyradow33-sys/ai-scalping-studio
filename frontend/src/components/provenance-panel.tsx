/**
 * Provenance of the stored dataset: where it came from and whether it changed.
 *
 * `GET /api/data/status` returns the sidecar verbatim, or `null` for a store
 * written before provenance existed. The null state says exactly that instead
 * of guessing a source or checksum; a page that invents a dataset id is worse
 * than one that admits it has none.
 */

import type { ReactNode } from "react";

import { formatTs } from "@/lib/format";
import type { Provenance } from "@/lib/types";

export interface ProvenancePanelProps {
  provenance: Provenance | null | undefined;
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-xs text-slate-400">{label}</dt>
      <dd className="text-sm text-slate-200">{children}</dd>
    </div>
  );
}

export function ProvenancePanel({ provenance }: ProvenancePanelProps) {
  return (
    <section aria-labelledby="provenance-heading" className="flex flex-col gap-2">
      <h2 id="provenance-heading" className="text-sm font-semibold text-slate-100">
        Veri kökeni
      </h2>

      {provenance === null || provenance === undefined ? (
        <p data-testid="provenance-missing" className="text-sm text-slate-400">
          Bu store için köken kaydı yok. Eski store’lar dataset kimliği, checksum
          ve oluşturma zamanı taşımaz; bu değerler tahmin edilmez.
        </p>
      ) : (
        <div className="flex flex-col gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3">
          <header className="flex flex-wrap items-center gap-2">
            <span
              data-testid="provenance-kind"
              data-kind={provenance.synthetic ? "synthetic" : "real"}
              className={
                provenance.synthetic
                  ? "rounded-sm border border-amber-600/60 bg-amber-500/10 px-2 py-0.5 text-xs font-medium text-amber-200"
                  : "rounded-sm border border-emerald-600/60 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-200"
              }
            >
              {provenance.synthetic ? "Sentetik" : "Gerçek"}
            </span>
            <span className="text-xs text-slate-400">
              {provenance.symbol} · {provenance.timeframe}
            </span>
          </header>

          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Row label="Kaynak">
              <span data-testid="provenance-source" className="break-all">
                {provenance.source}
              </span>
            </Row>
            <Row label="Dataset">
              <span data-testid="provenance-dataset-id" className="break-all font-mono text-xs">
                {provenance.dataset_id}
              </span>
            </Row>
            <Row label="Kontrol toplamı (checksum)">
              <span data-testid="provenance-checksum" className="break-all font-mono text-xs">
                {provenance.checksum}
              </span>
            </Row>
            <Row label="Oluşturuldu">
              <span data-testid="provenance-created-at" className="tabular-nums">
                {formatTs(provenance.created_at)}
              </span>
            </Row>
          </dl>

          {provenance.notes !== "" && (
            <p data-testid="provenance-notes" className="text-xs text-slate-400">
              {provenance.notes}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
