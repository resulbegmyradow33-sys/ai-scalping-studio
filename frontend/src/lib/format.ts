const DASH = "—";
const LOCALE = "tr-TR";

const numberFormatCache = new Map<number, Intl.NumberFormat>();

function numberFormatter(digits: number): Intl.NumberFormat {
  const safeDigits = Math.min(20, Math.max(0, Math.trunc(digits)));
  const cached = numberFormatCache.get(safeDigits);
  if (cached) return cached;

  const formatter = new Intl.NumberFormat(LOCALE, {
    minimumFractionDigits: safeDigits,
    maximumFractionDigits: safeDigits,
    useGrouping: true,
  });
  numberFormatCache.set(safeDigits, formatter);
  return formatter;
}

// Fixed zone: server and browser must agree or the rendered text hydration-mismatches.
const timestampFormatter = new Intl.DateTimeFormat(LOCALE, {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
  timeZone: "Europe/Istanbul",
});

function isPresent(value: number | null): value is number {
  return value !== null && Number.isFinite(value);
}

export function formatNumber(value: number | null, digits = 2): string {
  if (!isPresent(value)) return DASH;
  return numberFormatter(digits).format(value);
}

export function formatPct(value: number | null): string {
  if (!isPresent(value)) return DASH;
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}%${numberFormatter(2).format(Math.abs(value))}`;
}

export function formatTs(ts: string | null): string {
  if (!ts) return DASH;
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return DASH;
  return timestampFormatter.format(date);
}

export function formatMoney(value: number | null): string {
  if (!isPresent(value)) return DASH;
  const sign = value < 0 ? "-" : "";
  return `${sign}$${numberFormatter(2).format(Math.abs(value))}`;
}
