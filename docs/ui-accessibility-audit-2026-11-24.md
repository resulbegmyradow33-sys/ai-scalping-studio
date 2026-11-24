# UI accessibility and honesty audit — 2026-11-24

## Scope

| Page | Path | Revision audited |
|---|---|---|
| ML laboratory | `/labs/ml` | `feat/a11y` from `main` `693f8a3` |
| Paper trading | `/paper` | `feat/a11y` from `main` `693f8a3` |
| Autonomy | `/autonomy` | branch `feat/autonomyui` @ `75fc843` (not on `main`) |
| Shared shell | `app-shell.tsx`, `nav.tsx`, `stat-card.tsx` | `main` `693f8a3` |

`/autonomy` does not exist on `main`; the page and its tests exist only on
`feat/autonomyui` (`frontend/src/app/autonomy/page.tsx`, 1065 lines). It is
audited here read-only. No fix was applied to it because the file is not on this
branch and editing another agent's active branch would conflict.

Standard: WCAG 2.2 Level AA. Severity: Critical (blocks the task), Serious
(major barrier, workaround needed), Moderate (difficulty, workaround exists),
Minor (annoyance).

## Method and limits

Performed:

- Source review of both pages, their tests, the shell and shared components.
- Colour-contrast arithmetic from the design tokens in `src/app/globals.css`
  (`surface #0b0f17`, `slate-900 #0f172a`, `loss #dc2626`, `profit #16a34a`,
  `accent #3b82f6`), compositing the `/10` and `/40` alpha layers.
- `npm run lint`, `npx tsc --noEmit`, `npm test`, `npm run build`.
- Assertions in jsdom for labels, roles, live regions and focus moves.

Not performed (recorded rather than claimed):

- No live VoiceOver/NVDA/JAWS session and no 200%/400% zoom pass were run in this
  environment.
- No axe-core or Lighthouse scan was run (no browser session).
- jsdom does not apply CSS, so contrast findings come from the token arithmetic
  below, not from a rendered screenshot.

These gaps mean this report supports "no known AA blocker in the audited code",
not a conformance claim. A live AT pass is still required before release.

## Summary

| Severity | Found | Fixed on `/labs/ml` + `/paper` | Still open |
|---|---:|---:|---:|
| Critical | 0 | 0 | 0 |
| Serious | 2 | 2 | 2 (shared `StatCard`/token, `/autonomy`) |
| Moderate | 7 | 5 | 2 (F7, F8) |
| Minor | 7 | 1 | 6 |

The serious fixes are page-scoped: the same contrast values remain in
`stat-card.tsx`, the `/autonomy` branch and any other page using
`text-slate-500`/`text-loss` as text.

Conformance: **PARTIALLY CONFORMS** on the audited pages. No critical barrier
was found; the failures are contrast, one invalid ARIA attribute, and focus and
announcement gaps.

## Findings

### F1 — Serious — Hint and explanatory text fails contrast (WCAG 1.4.3)

`text-slate-500` (`#64748b`) on the card (`slate-900/40` over `#0b0f17`,
effective `#0d121f`) is **3.93:1**; on `slate-900` it is **3.75:1**. Normal text
needs 4.5:1. At `text-xs` (12 px) this is not large text. The affected strings
are the explanatory text that makes the pages honest: "tahmin metrikleri alım
satış sonucu değildir", "Test bölümü okunmadı…", "sayfa metrik uydurmaz",
"Kapatma oturumu `closed` yapar…", and every `<dt>` label.

Locations before the fix:

- `frontend/src/app/labs/ml/page.tsx:68` (`HINT_CLASS`) plus 18 inline uses.
- `frontend/src/app/paper/page.tsx:121,199,203,212,529,546,655,708,853`.
- `frontend/src/components/stat-card.tsx:45` (`hint`).
- `feat/autonomyui:frontend/src/app/autonomy/page.tsx:263,409,659,827,965,1059`.

Fix applied in `/labs/ml` and `/paper`: `text-slate-500` → `text-slate-400`
(`#94a3b8`, **7.29:1** on the card). Hierarchy is now carried by size and
weight. `stat-card.tsx` and `/autonomy` still use `slate-500`.

### F2 — Serious — Loss red fails contrast as text (WCAG 1.4.3)

`text-loss` (`#dc2626`) on `bg-loss/10` is **3.66:1**; on the card it is
**3.87:1**. It labels the `failed` status chip in both pages and negative P&L in
the paper history table. The sign is present, so information is not colour-only,
but the text itself is unreadable at AA for low-vision users.

Locations before the fix:

- `frontend/src/app/labs/ml/page.tsx:55` (`STATUS_CLASS.failed`).
- `frontend/src/app/paper/page.tsx:45` (`STATUS_TONE.failed`).
- `frontend/src/app/paper/page.tsx:1058` (negative P&L).
- `src/components/stat-card.tsx:23` (`TONE_CLASS.loss`).
- `feat/autonomyui:.../autonomy/page.tsx:56,112`.

Fix applied in the two audited pages: `text-loss` → `text-rose-300`
(`#fda4af`, **9.07:1** on `rose-500/10`). The global `--color-loss` token is
still too dark for text and should gain a separate text token (see "Remaining
work").

### F3 — Moderate — Invalid `aria-selected` on a table row (WCAG 4.1.2)

`RunsTable` put `aria-selected={selected}` on a `<tr>` inside a plain
`<table>`. ARIA 1.2 supports `aria-selected` on `row` only inside a
`grid`/`treegrid`; on a native table it is ignored by browsers and rejected by
validators. The button already carried `aria-current="true"`.

Location: originally `frontend/src/app/labs/ml/page.tsx:1753`.

Fix applied: removed `aria-selected` from the row, kept `aria-current` on the
button and the visual `bg-slate-800/40` selection tint. Regression test added.

### F4 — Moderate — Stress-multiplier error was not announced (WCAG 4.1.3)

Pressing "Değerlendir" with a bad stress value set `stressError` but left focus
on the submit button. The field error was only reachable through
`aria-describedby` when the user later focused the input, and the form had no
summary alert.

Location: `frontend/src/app/labs/ml/page.tsx:1196-1206`.

Fix applied: the message paragraph now has `role="alert"`; `aria-invalid` was
already set. Regression test asserts both.

### F5 — Moderate — Close confirmation dropped focus to `<body>` (WCAG 2.4.3)

The close step is an inline confirm group, not a modal. Opening it moved focus
to "Kapatmayı onayla" (good). Cancelling or completing it unmounted the group
without a destination, so keyboard focus fell to the document. After a
successful close the trigger is disabled, so returning focus to it is not
possible.

Location: `frontend/src/app/paper/page.tsx:389-391` (old effect), `682-689`
(old cancel handler).

Fix applied: a deliberate return target. Cancel returns focus to "Oturumu
kapat"; successful close and the 409 "already closed" path move focus to the
"Oturum ayrıntısı" heading (`tabIndex={-1}`), where the status notice is read
next. Tests assert both destinations.

### F6 — Moderate — Pagination range changed silently (WCAG 4.1.3)

After paging, the range text ("1–20 / 25" → "21–25 / 25") changed with focus
still on the pager button and no live region, so a screen-reader user had no
confirmation that new rows arrived.

Location: `frontend/src/app/paper/page.tsx:212`.

Fix applied: `aria-live="polite"` on the range span. Regression test asserts it.

### F7 — Moderate — Field errors that appear on blur are not announced (open)

Each field error is a plain `<p>` referenced by `aria-describedby`. On blur,
`visibleErrors` reveals the message while focus moves to the next control, so
nothing is announced until submit (where the summary `role="alert"` and
`focusFirstError` do work). The summary path is sound; the blur path is silent.

Location: `frontend/src/app/labs/ml/page.tsx:694-696` and the seven sibling
blocks (`727,758,792,832,867,896,927`).

Suggested fix: validate on submit only, or add `role="alert"` to field errors
and accept the extra chatter, or route blur errors into one shared polite live
region. This is a behaviour choice, so it was not changed here.

### F8 — Moderate — Disabling the focused control loses focus (open)

`disabled={starting}` on "Koşuyu başlat" (`labs/ml/page.tsx:983`) and
`disabled={refreshing}` on the autonomy refresh button remove the focused
element from the tab order the moment it is used. The next Tab starts from the
document. The pager buttons have the same effect at the first/last page.

Suggested fix: use `aria-disabled` plus the existing guard in `handleSubmit`
(it already returns early when `starting`), and add `aria-disabled:` styles.
For the pager, either keep the button enabled and ignore the click, or move
focus to the table caption after the page settles.

### F9 — Moderate — Every route shared one document title (WCAG 2.4.2) — fixed for two routes

`app/layout.tsx` sets `title: "AI Scalping Studio"` and no nested layout
overrode it, so `/labs/ml` and `/paper` were indistinguishable in the tab, the
window list and the screen-reader page announcement. Both pages are client
components, which cannot export `metadata`.

Fix applied: new route layouts
`frontend/src/app/labs/ml/layout.tsx` and `frontend/src/app/paper/layout.tsx`
export `metadata.title`. Verified in the build output:
`<title>ML Laboratuvarı — AI Scalping Studio</title>` and
`<title>Paper Trading — AI Scalping Studio</title>`.

Still open for `/autonomy` (branch) and the other routes.

### F10 — Minor — Checkbox and select hints were not programmatically attached (WCAG 3.3.2)

"Varsayılan olarak düz satırlar etiketlenmez…", "Test bölünmesi aday başına
yalnızca bir kez okunmalı." and "Giriş barındaki kural etiketiyle eşlenir." were
visible but not referenced by their controls.

Location: `frontend/src/app/labs/ml/page.tsx:934-953,1154-1168,1212-1226`.

Fix applied: stable hint ids plus `aria-describedby` on
`#ml-include-flat`, `#ml-split` and `#ml-regime`.

### F11 — Minor — Row selection uses `aria-pressed` for single selection (open)

`paper/page.tsx:629` puts `aria-pressed` on the session button. A pressed
toggle implies the user can unpress it; selecting another session does not
unpress the first, it replaces it. `aria-current="true"` (as used in the ML
runs table) or grid selection semantics would be more accurate. This is not a
hard ARIA violation and matches the existing `run-table.tsx` pattern, so it is
reported rather than changed.

### F12 — Minor — Empty detail page can render a reversed range (open)

`pageRange` (`paper/page.tsx:86-91`) computes `offset + 1` to
`offset + items.length`. If a detail table returns an empty page while
`total > 0`, the pager reads "21–20 / 25". The session table has a dedicated
empty-page state; the other four tables do not. Suggested fix: return
`0 / {total}` when `items.length === 0` and show the same "page is empty" note
for all tables.

### F13 — Minor — ECE/MCE fallback provenance is unclear (open)

`predictionRows` (`labs/ml/page.tsx:256-269`) prefers
`validation.calibration.ece` and falls back to the run-level
`detail.calibration.ece`, while the column is labelled "Doğrulama". The fallback
is probably the same validation calibration, but the page does not say so. No
number is invented; the provenance is just ambiguous. Suggested fix: label the
fallback explicitly or drop it.

### F14 — Minor — `type="number"` plus `noValidate` defeats the native constraints and the comma handling (open)

The form sets `noValidate` (`labs/ml/page.tsx:652`), so `required`, `min`,
`max` and `step` are inert; validation is entirely in `validateForm`, which is
consistent with the messages. Separately, `toNumber` accepts a decimal comma,
but the fraction fields are `type="number"` (`770,809,844`), where Chrome
discards "0,7" and reports an empty value. A Turkish-keyboard user therefore
gets "0 ile 1 arasında olmalı" for a value they typed correctly. Not a WCAG
criterion, but a real locale barrier. Suggested fix: `type="text"
inputMode="decimal"` for the fraction fields, as the stress field already does.

### F15 — Minor — Selected run detail appears above its trigger (open)

Clicking a row in the runs table (bottom of the page) renders the run panel
above the table. The `sr-only` status sentence announces the new run and status,
so the state change is not silent, but there is no focus move and the new
content is behind the user's reading position. Consider focusing the "Koşu"
heading after selection. Not a WCAG failure.

### F16 — Minor — Redundant live-region configuration on `/autonomy` (branch)

`role="status"` already implies `aria-live="polite"`; the branch sets both
(`feat/autonomyui:.../autonomy/page.tsx:427-434`). Harmless, but redundant and
easy to misread as a second live region.

## `/autonomy` (branch `feat/autonomyui`) — read-only audit

Structure is good: one `h1`, labelled `h2` sections, a focusable named scroll
region per table, `<caption>`, `scope` on every header, a labelled window-size
`select` with `aria-describedby`, explicit "not configured" versus "unreadable"
sentences, explicit empty states, and a polite status region for the last
successful read. No fake zeros: `count()` returns a dash for absent numbers and
`storeState()` separates missing from unreadable.

Open items on that branch:

1. F1 and F2 contrast: `text-slate-500` at lines `263,409,659,827,965,1059`;
   `text-loss` at `56` (failed job chip) and `112` (unknown champion mode);
   `StatCard` `tone="loss"` at `627-630`.
2. F8: `disabled={refreshing}` on the refresh button (line `403`).
3. F9: no route `metadata`/`layout.tsx`, so the document title is still the app
   name.
4. F16: redundant `aria-live` on `role="status"`.

Apply these on `feat/autonomyui` before that page merges; the same patterns can
be copied from this branch's two pages.

## What is working well

- Every input has a `<label htmlFor>`. No placeholder-as-label anywhere.
- Related controls are grouped with `<fieldset>`/`<legend>`; the ML form exposes
  `required`, `min`, `max` and `step` to AT even though it validates in JS.
- Submit-time validation is announced twice over: a `role="alert"` summary with
  anchor links, and focus moved to the first invalid control, which carries
  `aria-invalid` and an `aria-describedby` pointing at its message.
- Every table has a `<caption>`, `scope="col"`/`scope="row"`, and a
  `tabIndex={0}` `role="region"` wrapper named by the caption, so horizontal
  (and, where bounded, vertical) scrolling is keyboard-operable.
- Status is never colour-only: chips carry Turkish text labels, and money always
  carries its sign.
- Empty, missing (404), unreadable, and error states are written out per section
  on all three pages. A missing number is a dash ("—"), never a zero.
- The paper page states plainly that it reads `/api/paper` and that closing a
  session is its only write; the autonomy page states the same for
  `/api/autonomy/*`.
- Global affordances are in place: skip link to `#main`, `lang="tr"`, one `main`
  landmark, visible `focus-visible` outlines, `prefers-reduced-motion` handling,
  and `color-scheme: dark`.

## Verification

Run from `frontend/` on `feat/a11y`:

```text
npm test                 → 25 files, 224 tests passed
npx tsc --noEmit         → no output (clean)
npm run lint             → clean
npm run build            → compiled successfully, 11 static routes
```

Build-output title check:

```text
.next/server/app/labs/ml.html → <title>ML Laboratuvarı — AI Scalping Studio</title>
.next/server/app/paper.html   → <title>Paper Trading — AI Scalping Studio</title>
```

Tests added:

- `labs/ml/page.test.tsx`: selected run exposes `aria-current` and the row does
  not expose `aria-selected`; stress error has `role="alert"` and the field is
  `aria-invalid`.
- `paper/page.test.tsx`: the pager range is a polite live region; focus lands on
  "Kapatmayı onayla" when the confirm opens; cancel returns focus to "Oturumu
  kapat"; a successful close moves focus to the section heading.

## Remaining work before an AA claim

1. Run axe-core and Lighthouse on both routes, then a keyboard-only and
   VoiceOver/NVDA pass on the critical paths (train a run, select a run,
   evaluate, page the ledger, close a session).
2. Check 200% and 400% zoom, and forced-colors mode.
3. Change the shared `stat-card.tsx` hint and loss tones, or add
   `--color-loss-text`; then re-check every page that uses `text-loss` as text.
4. Decide F7 (blur announcements) and F8 (`aria-disabled`) as a shared pattern
   in `docs/ui-accessibility-checklist.md` rather than per page.
5. Apply F1, F2, F8, F9 and F16 on `feat/autonomyui`.
6. Re-audit after those changes; the checklist in
   `docs/ui-accessibility-checklist.md` is the gate for new pages.
