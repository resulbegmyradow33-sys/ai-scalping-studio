"use client";

/**
 * Strategy Intelligence: what the system is running, how the regime
 * distribution looks for one stored slice, and what the wire does not expose.
 *
 * Every value comes from an existing read-only endpoint — `GET /api/regime`,
 * `GET /api/strategy/card` and `GET /api/autonomy/champions`. The strategy ×
 * regime table is not exposed yet: its inputs are simulator trade rows produced
 * inside an evaluation pass, so this page states that gap instead of deriving a
 * table in the browser.
 */

import { useState } from "react";
import type { ReactNode } from "react";

import { useAutonomyChampions } from "@/hooks/use-autonomy-champions";
import { useRegime, useTimeframes } from "@/hooks/use-regime";
import { useStrategyCard } from "@/hooks/use-strategy-card";
import { ApiError } from "@/lib/api";
import { formatNumber, formatTs } from "@/lib/format";
import type { AutonomyChampion, AutonomyChampionList, StrategyCard } from "@/lib/types";

const FIELD_CLASS =
  "rounded-sm border border-slate-700 bg-slate-950/60 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const LABEL_CLASS = "text-xs text-slate-400";

type Tone = "legacy" | "fallback" | "dsl" | "muted";

const TONE_CLASS: Record<Tone, string> = {
  legacy: "border-amber-500/50 bg-amber-500/10 text-amber-200",
  fallback: "border-amber-600/40 bg-amber-600/10 text-amber-200",
  dsl: "border-profit/50 bg-profit/10 text-profit",
  muted: "border-slate-700 bg-slate-800/40 text-slate-400",
};

function errorDetail(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return "Veri alınamadı.";
}

function Badge({
  tone,
  testId,
  children,
}: {
  tone: Tone;
  testId: string;
  children: ReactNode;
}) {
  return (
    <span
      data-testid={testId}
      className={`rounded-sm border px-1.5 py-0.5 text-xs font-medium ${TONE_CLASS[tone]}`}
    >
      {children}
    </span>
  );
}

/** Short, copy-pasteable identity for a champion row. */
function championIdentity(champion: AutonomyChampion): string {
  const id = champion.policy_id ?? champion.policy_digest ?? champion.candidate_id;
  return id.length > 16 ? `${id.slice(0, 16)}…` : id;
}

function CardClassification({
  card,
  error,
  loading,
}: {
  card: StrategyCard | undefined;
  error: unknown;
  loading: boolean;
}) {
  if (loading && card === undefined) {
    return <p className="text-xs text-slate-500">Strateji kartı okunuyor…</p>;
  }
  if (error !== undefined && error !== null) {
    return (
      <p data-testid="strategy-card-error" className="text-sm text-amber-200">
        Strateji kartı alınamadı ({errorDetail(error)}). Sınıflandırma bilinmiyor.
      </p>
    );
  }
  if (card === undefined) {
    return (
      <p data-testid="strategy-card-missing" className="text-sm text-slate-400">
        Strateji kartı yanıtı boş; sınıflandırma bilinmiyor.
      </p>
    );
  }

  if (card.active) {
    return (
      <div data-testid="classification-card" className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="legacy" testId="classification-legacy">
            legacy
          </Badge>
          <span className="font-mono text-sm text-slate-100">{card.strategy}</span>
          {card.params?.timeframe ? (
            <span className="text-xs text-slate-400">· {card.params.timeframe}</span>
          ) : null}
        </div>
        <p className="text-xs text-slate-400">
          flow_momentum kartı legacy incumbent olarak sınıflandırılır; DSL arama
          şablonu ya da varsayılan şampiyon değildir.
          {card.created_at ? ` Kart ${formatTs(card.created_at)} tarihinde yazıldı.` : ""}
        </p>
      </div>
    );
  }

  return (
    <div data-testid="classification-card" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="fallback" testId="classification-fallback">
          fallback
        </Badge>
        <span className="font-mono text-sm text-slate-100">{card.strategy}</span>
      </div>
      <p className="text-xs text-slate-400">
        Kart yok: panel legacy M1 momentumunu oynatıyor. Otonom yol bir DSL
        şampiyonu olmadan işlem açmaz (NO TRADE).
      </p>
    </div>
  );
}

function ChampionClassification({
  data,
  error,
}: {
  data: AutonomyChampionList | undefined;
  error: unknown;
}) {
  if (error !== undefined && error !== null) {
    return (
      <p data-testid="champions-error" className="text-sm text-amber-200">
        Şampiyon arşivi alınamadı ({errorDetail(error)}). DSL sınıflandırması
        bilinmiyor.
      </p>
    );
  }
  if (data === undefined) {
    return <p className="text-xs text-slate-500">Şampiyon arşivi okunuyor…</p>;
  }
  if (!data.configured) {
    return (
      <div data-testid="champions-unconfigured" className="flex flex-col gap-1">
        <Badge tone="muted" testId="classification-dsl-missing">
          dsl yok
        </Badge>
        <p className="text-xs text-slate-400">
          Keşif arşivi yapılandırılmadı (dosya yok); hiçbir şampiyon okunmadı.
        </p>
      </div>
    );
  }
  if (!data.readable) {
    return (
      <div data-testid="champions-unreadable" className="flex flex-col gap-1">
        <Badge tone="muted" testId="classification-dsl-unknown">
          okunamadı
        </Badge>
        <p className="text-xs text-slate-400">
          Arşiv var ama şampiyon tablosu okunamadı; bu &quot;şampiyon yok&quot;
          anlamına gelmez.
        </p>
      </div>
    );
  }

  const items = data.items ?? [];
  if (items.length === 0) {
    return (
      <div data-testid="champions-empty" className="flex flex-col gap-1">
        <Badge tone="muted" testId="classification-dsl-missing">
          dsl yok
        </Badge>
        <p className="text-xs text-slate-400">
          Arşiv okunabilir, kayıtlı şampiyon yok. Bu durumda otonom sonuç NO
          TRADE&apos;dir.
        </p>
      </div>
    );
  }

  return (
    <div data-testid="champions-list" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="dsl" testId="classification-dsl">
          dsl
        </Badge>
        <span className="text-xs text-slate-400">
          {items.length} kapsamda şampiyon
        </span>
      </div>
      <ul className="flex flex-col gap-1">
        {items.map((champion) => (
          <li
            key={`${champion.scope}:${champion.candidate_id}`}
            data-testid="champion-row"
            className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-xs"
          >
            <span className="font-mono text-slate-200">{champion.scope}</span>
            <span className="font-mono text-slate-400">{champion.mode ?? "—"}</span>
            <span className="truncate font-mono text-slate-500" title={champion.policy_id ?? champion.candidate_id}>
              {championIdentity(champion)}
            </span>
            <span className="tabular-nums text-slate-500">
              {formatTs(champion.promoted_at ?? null)}
            </span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-slate-500">
        mode arşivde yoksa backend <span className="font-mono">dsl</span> varsayar;
        satır bunu kanıtlamaz.
      </p>
    </div>
  );
}

export default function StrategiesPage() {
  const [symbol, setSymbol] = useState("XAUUSD");
  const [timeframe, setTimeframe] = useState("M1");

  const cleanSymbol = symbol.trim();
  const timeframesState = useTimeframes();
  const timeframes =
    timeframesState.data !== undefined && timeframesState.data.length > 0
      ? timeframesState.data.map((item) => item.timeframe)
      : ["M1"];

  const regimeState = useRegime(cleanSymbol, timeframe);
  const cardState = useStrategyCard();
  const championsState = useAutonomyChampions();

  const data = regimeState.data;
  const rows = data?.rows ?? 0;
  const counts = data?.counts ?? [];

  const regimeError =
    regimeState.error === undefined ? null : errorDetail(regimeState.error);

  /**
   * One polite live region for the page's headline state. Loading is not
   * announced (it is an interstitial); only settled outcomes are.
   */
  const statusMessage = (() => {
    if (cleanSymbol === "") return null;
    if (data === undefined) {
      return regimeError === null ? null : `Rejim verisi alınamadı (${regimeError}).`;
    }
    if (rows === 0) return "Bu sembol ve zaman diliminde rejim verisi yok.";
    return `${formatNumber(rows, 0)} bar sınıflandırıldı, ${counts.length} rejim.`;
  })();

  return (
    <section aria-labelledby="strategies-heading" className="flex flex-col gap-5">
      <div>
        <h1 id="strategies-heading" className="text-xl font-semibold text-slate-100">
          Strateji Zekâsı
        </h1>
        <p className="text-xs text-slate-400">
          Rejim dağılımı ve mevcut strateji sınıflandırması; yalnızca API&apos;nin
          döndürdüğü değerler. Terfi ya da kârlılık iddiası değildir.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label htmlFor="strategies-symbol" className={LABEL_CLASS}>
            Sembol
          </label>
          <input
            id="strategies-symbol"
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
            autoComplete="off"
            spellCheck={false}
            className={FIELD_CLASS}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="strategies-timeframe" className={LABEL_CLASS}>
            Zaman dilimi
          </label>
          <select
            id="strategies-timeframe"
            value={timeframe}
            onChange={(event) => setTimeframe(event.target.value)}
            className={FIELD_CLASS}
          >
            {timeframes.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <span className="pb-2 text-xs tabular-nums text-slate-500">
          {data === undefined ? "" : `${formatNumber(rows, 0)} bar`}
        </span>
      </div>

      <p role="status" data-testid="strategy-intel-status" className="text-sm text-slate-300">
        {statusMessage ?? ""}
      </p>

      <section aria-labelledby="classification-heading" className="flex flex-col gap-2">
        <h2 id="classification-heading" className="text-sm font-semibold text-slate-100">
          Strateji sınıflandırması
        </h2>
        <div className="flex flex-col gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3">
          <div className="flex flex-col gap-1">
            <h3 className="text-xs font-medium text-slate-300">Panel kartı</h3>
            <CardClassification
              card={cardState.card}
              error={cardState.error}
              loading={cardState.isLoading}
            />
          </div>
          <div className="flex flex-col gap-1">
            <h3 className="text-xs font-medium text-slate-300">DSL şampiyonları</h3>
            <ChampionClassification
              data={championsState.data}
              error={championsState.error}
            />
          </div>
        </div>
      </section>

      <section aria-labelledby="regime-dist-heading" className="flex flex-col gap-2">
        <h2 id="regime-dist-heading" className="text-sm font-semibold text-slate-100">
          Rejim dağılımı
        </h2>

        {regimeError !== null ? (
          <p data-testid="regime-error" className="text-sm text-amber-200">
            Rejim verisi alınamadı ({regimeError}).
          </p>
        ) : cleanSymbol === "" ? (
          <p className="text-sm text-slate-400">Sembol girerek rejimi görüntüle.</p>
        ) : data === undefined ? (
          <p data-testid="regime-loading" className="text-sm text-slate-400">
            Yükleniyor…
          </p>
        ) : rows === 0 ? (
          <p data-testid="regime-empty" className="text-sm text-slate-400">
            Bu sembol ve zaman diliminde veri yok — Veri sayfasından içe aktar.
          </p>
        ) : (
          <div
            className="overflow-x-auto"
            tabIndex={0}
            role="region"
            aria-labelledby="regime-dist-heading"
          >
            <table className="w-full border-collapse text-xs">
              <caption className="pb-1 text-left text-xs text-slate-500">
                {data?.symbol} {data?.timeframe} · tüm store dilimi, {formatNumber(rows, 0)} bar
              </caption>
              <thead>
                <tr className="border-b border-slate-800 text-left text-slate-400">
                  <th scope="col" className="py-1.5 pr-3 font-medium">
                    Rejim
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                    Bar
                  </th>
                  <th scope="col" className="py-1.5 text-right font-medium">
                    Pay
                  </th>
                </tr>
              </thead>
              <tbody>
                {counts.map((row) => (
                  <tr
                    key={row.regime}
                    data-testid={`regime-count-${row.regime}`}
                    className="border-b border-slate-800/60"
                  >
                    <th scope="row" className="py-1.5 pr-3 text-left font-mono font-normal text-slate-200">
                      {row.regime}
                    </th>
                    <td className="py-1.5 pr-3 text-right tabular-nums text-slate-300">
                      {formatNumber(row.bars, 0)}
                    </td>
                    <td className="py-1.5 text-right tabular-nums text-slate-400">
                      {formatNumber(row.share * 100, 1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section aria-labelledby="strategy-regime-heading" className="flex flex-col gap-2">
        <h2 id="strategy-regime-heading" className="text-sm font-semibold text-slate-100">
          Strateji × rejim
        </h2>
        <div
          data-testid="strategy-regime-unavailable"
          className="flex flex-col gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
        >
          <p className="text-sm text-slate-300">
            Bu tablo henüz API&apos;de açık değil.
          </p>
          <p className="text-xs text-slate-400">
            <span className="font-mono">regime/strategy.py</span> içindeki{" "}
            <span className="font-mono">strategy_regime_table</span> kapalı işlem
            satırları ve rejim etiketleri ister. İşlem listesi yalnızca bir
            simülatör/ML değerlendirme geçişinde üretilir; okuma amaçlı bir uç
            yok. Bu sayfa kırılımı tarayıcıda türetmez — uydurulmuş bir tablo
            göstermektense durumu açıkça yazar.
          </p>
          <p className="text-xs text-slate-500">
            Bitmiş bir ML koşusu için <span className="font-mono">POST /api/ml/evaluate</span>{" "}
            yanıtı <span className="font-mono">regime.rows</span> altında koşu
            bazlı rejim kırılımı döndürür (trades, win_rate, expectancy_r,
            total_r, pnl, profit_factor ve unmatched_trades). Bu, tek bir koşunun
            değerlendirmesidir; kart × rejim tablosu değildir.
          </p>
        </div>
      </section>

      <p className="text-xs text-slate-500">
        Rejim etiketi geçmiş yolun tanımıdır, fiyat tahmini değildir. Sınıflandırma
        bir terfi kanıtı değildir; terfi yalnızca doğrulanmış DSL şampiyonuyla olur.
      </p>
    </section>
  );
}
