# UI accessibility checklist for new pages

Short gate for any new page or interactive component. It encodes the failures
found in `docs/ui-accessibility-audit-2026-11-24.md`, so a page that passes this
list should not repeat them. WCAG 2.2 AA is the target; the full reason for each
item is in the audit.

## Page shell

- [ ] Route has its own document title. Client components cannot export
      `metadata`; add a route `layout.tsx` with `export const metadata`.
- [ ] One `h1`, then `h2`/`h3` without skipped levels.
- [ ] Each visual section is a `<section aria-labelledby>` pointing at its
      heading.
- [ ] The page renders inside the existing `AppShell` (skip link, `main`, nav
      with `aria-current`).

## Forms

- [ ] Every control has a `<label htmlFor>`; no placeholder is used as a label.
- [ ] Related controls sit in `<fieldset>` with a `<legend>`.
- [ ] Helper text has an `id` and is referenced with `aria-describedby`.
- [ ] Errors set `aria-invalid` and point `aria-describedby` at the message.
- [ ] Field messages are announced: `role="alert"` for a single message, or one
      polite live region for a group. Validate-on-submit needs a `role="alert"`
      summary plus focus on the first invalid field.
- [ ] Submitting does not disable the focused button without moving focus
      (`aria-disabled` plus an early return in the handler is preferred).
- [ ] Currency, fraction and decimal fields accept the Turkish decimal comma
      (`type="text" inputMode="decimal"`, parse with `,` → `.`).

## Tables

- [ ] `<caption>` names the table and doubles as the region label.
- [ ] Scroll container: `role="region"`, `aria-labelledby` = caption id,
      `tabIndex={0}`, visible focus ring.
- [ ] Header cells use `scope="col"`; the first cell of each row uses
      `scope="row"`.
- [ ] Do not put `aria-selected` on a `<tr>` outside a `grid`/`treegrid`. Use
      `aria-current` on the row's button, or real grid semantics.
- [ ] `aria-pressed` only for a control the user can actually unpress. Single
      selection is `aria-current` or a radio/grid selection.
- [ ] Paging changes are announced (`aria-live="polite"` on the range) and the
      range never reads backwards for an empty page.

## Status and colour

- [ ] Status has a text label; colour is never the only carrier.
- [ ] Money and P&L carry their sign.
- [ ] Text contrast is at least 4.5:1 (3:1 for >= 24 px or >= 18.66 px bold).
      Check `text-slate-500` and `text-loss` on the dark surface before use;
      they currently fail. `text-slate-400` and `text-rose-300` pass.
- [ ] Icons next to text are `aria-hidden`; icon-only buttons have
      `aria-label`.

## Async and states

- [ ] Loading, empty, missing (404), unreadable and error states are written
      out per section; an absent number is "—", never `0`.
- [ ] Long operations expose `aria-busy` and a status sentence.
- [ ] Polling has a documented interval, stops on terminal state, and clears its
      timer on unmount.
- [ ] Inline confirm steps return focus on cancel and after success. If the
      trigger becomes disabled, move focus to the section heading
      (`tabIndex={-1}`).
- [ ] Nothing on the page implies paper data is real money or validated
      evidence. Preview/provisional values are labelled as such.

## Keyboard and visual

- [ ] Every interactive element is reachable and operable by keyboard, in DOM
      order that matches the visual order.
- [ ] No keyboard trap; `Escape` closes overlays and returns focus.
- [ ] Focus is always visible (`focus-visible:outline-*`).
- [ ] At 200% and 400% zoom the page itself does not scroll horizontally; wide
      tables scroll inside their regions.
- [ ] Motion respects `prefers-reduced-motion` (already global; do not add
      unguarded animation).
- [ ] Content remains visible in forced-colors mode.

## Tests

- [ ] Query by role and label (`getByRole`, `getByLabelText`), not by class.
- [ ] Assert `aria-invalid`, `aria-describedby`, live regions and focus moves
      for the error and confirm paths.
- [ ] Assert the explicit empty/missing/unreadable sentence and that no zero is
      invented.
- [ ] Run `npm test`, `npx tsc --noEmit`, `npm run lint`, `npm run build`.
- [ ] Before an AA claim: axe-core/Lighthouse plus a keyboard-only and
      screen-reader pass on the critical path.
