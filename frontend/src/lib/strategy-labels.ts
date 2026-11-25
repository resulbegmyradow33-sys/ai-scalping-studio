/**
 * Which strategy labels belong to the legacy panel path.
 *
 * The backend now classifies the card payload, but paper sessions and decision
 * messages still carry a bare label. A known legacy label must be shown with a
 * "legacy baseline" badge instead of reading like the current champion.
 */

const LEGACY_STRATEGIES: ReadonlySet<string> = new Set(["momentum", "flow_momentum"]);

/** Reports written before the classification field used a prefix instead. */
const LEGACY_PREFIX = "legacy_baseline:";

export function isLegacyStrategy(label: string | null | undefined): boolean {
  if (label === null || label === undefined) return false;
  return LEGACY_STRATEGIES.has(label) || label.startsWith(LEGACY_PREFIX);
}
