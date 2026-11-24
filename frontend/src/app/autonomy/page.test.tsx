import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  AutonomyEventItem,
  AutonomyEventPage,
  AutonomyJobItem,
  AutonomyJobPage,
  AutonomyStatus,
} from "@/lib/types";

/**
 * The page has exactly three boundaries: the aggregate status, the bounded jobs
 * window and the bounded events window. Replacing the client keeps these tests
 * about the page's honest states: loaded, missing store, unreadable store,
 * readable-but-empty, error, and champion modes.
 */
const apiMock = vi.hoisted(() => ({
  autonomyStatus: vi.fn(),
  autonomyJobs: vi.fn(),
  autonomyEvents: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, api: apiMock };
});

const { default: AutonomyPage } = await import("./page");

const NOW = "2026-11-24T09:30:00Z";

function status(overrides: Partial<AutonomyStatus> = {}): AutonomyStatus {
  return {
    paper: {
      configured: true,
      readable: true,
      sessions: 2,
      open_positions: 1,
      closed_trades: 7,
    },
    orchestrator: {
      configured: true,
      readable: true,
      queued: 3,
      running: 1,
      failed: 0,
      paused: true,
      safe_mode: true,
      research_disabled: false,
      last_event: { id: 9, ts: NOW, event: "paused", job_id: null },
    },
    champions: [
      {
        scope: "XAUUSD:M5",
        candidate_id: "cand-evidence",
        mode: "evidence",
        promoted_at: NOW,
        policy_id: "policy-1",
        policy_digest: "digest-1",
      },
    ],
    champions_readable: true,
    discoveries: { configured: true, readable: true, runs: 4, decisions: 2 },
    data: { configured: true, readable: true, symbols: ["XAUUSD"] },
    ...overrides,
  };
}

function job(overrides: Partial<AutonomyJobItem> = {}): AutonomyJobItem {
  return {
    id: 12,
    kind: "paper.advance",
    status: "queued",
    attempts: 0,
    defer_count: 1,
    priority: 5,
    created_at: NOW,
    scheduled_at: NOW,
    started_at: null,
    finished_at: null,
    ...overrides,
  };
}

function jobsPage(overrides: Partial<AutonomyJobPage> = {}): AutonomyJobPage {
  return {
    configured: true,
    readable: true,
    items: [job(), job({ id: 11, status: "failed", attempts: 2, defer_count: 0 })],
    limit: 25,
    ...overrides,
  };
}

function event(overrides: Partial<AutonomyEventItem> = {}): AutonomyEventItem {
  return { id: 5, ts: NOW, event: "paused", job_id: null, ...overrides };
}

function eventsPage(overrides: Partial<AutonomyEventPage> = {}): AutonomyEventPage {
  return {
    configured: true,
    readable: true,
    items: [event(), event({ id: 4, event: "enqueued", job_id: 12 })],
    limit: 25,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  apiMock.autonomyStatus.mockResolvedValue(status());
  apiMock.autonomyJobs.mockResolvedValue(jobsPage());
  apiMock.autonomyEvents.mockResolvedValue(eventsPage());
});

afterEach(cleanup);

describe("AutonomyPage", () => {
  it("renders every section from the API payload and the paper-only notice", async () => {
    render(<AutonomyPage />);

    expect(await screen.findByTestId("autonomy-paper-sessions")).toHaveTextContent("2");
    expect(screen.getByTestId("autonomy-paper-open-positions")).toHaveTextContent("1");
    expect(screen.getByTestId("autonomy-paper-closed-trades")).toHaveTextContent("7");

    expect(screen.getByTestId("autonomy-orchestrator-queued")).toHaveTextContent("3");
    expect(screen.getByTestId("autonomy-orchestrator-running")).toHaveTextContent("1");
    expect(screen.getByTestId("autonomy-orchestrator-failed")).toHaveTextContent("0");
    expect(screen.getByTestId("autonomy-orchestrator-paused")).toHaveTextContent("Evet");
    expect(screen.getByTestId("autonomy-orchestrator-safe-mode")).toHaveTextContent("Etkin");
    expect(screen.getByTestId("autonomy-orchestrator-research-disabled")).toHaveTextContent(
      "Açık",
    );
    expect(screen.getByTestId("autonomy-orchestrator-last-event")).toHaveTextContent(
      "Son olay: paused",
    );

    expect(screen.getByTestId("autonomy-discoveries-runs")).toHaveTextContent("4");
    expect(screen.getByTestId("autonomy-discoveries-decisions")).toHaveTextContent("2");
    expect(screen.getByTestId("autonomy-data-symbols")).toHaveTextContent("XAUUSD");
    expect(screen.getByTestId("autonomy-paper-notice")).toHaveTextContent(
      "Yalnızca paper: gerçek para yok.",
    );

    // Jobs and events are bounded windows with status chips and row headers.
    expect(await screen.findByTestId("autonomy-job-status-12")).toHaveTextContent("Sırada");
    expect(screen.getByTestId("autonomy-job-status-11")).toHaveTextContent("Hata");
    expect(screen.getByTestId("autonomy-events-table")).toHaveTextContent("enqueued");
    expect(screen.getByRole("region", { name: /İşler — en yeni önce/ })).toBeInTheDocument();
  });

  it("asks for a bounded window when the operator changes the page size", async () => {
    render(<AutonomyPage />);

    await screen.findByTestId("autonomy-job-status-12");
    fireEvent.change(screen.getByLabelText("Gösterilecek iş sayısı"), {
      target: { value: "100" },
    });

    await waitFor(() => expect(apiMock.autonomyJobs).toHaveBeenLastCalledWith(100));
    expect(apiMock.autonomyJobs).toHaveBeenCalledTimes(2);

    fireEvent.change(screen.getByLabelText("Gösterilecek olay sayısı"), {
      target: { value: "50" },
    });
    await waitFor(() => expect(apiMock.autonomyEvents).toHaveBeenLastCalledWith(50));
  });

  it("renders missing stores as not configured, never as zeros", async () => {
    apiMock.autonomyStatus.mockResolvedValue(
      status({
        paper: { configured: false, readable: false },
        orchestrator: { configured: false, readable: false },
        champions: [],
        champions_readable: false,
        discoveries: { configured: false, readable: false },
        data: { configured: false, readable: false },
      }),
    );
    apiMock.autonomyJobs.mockResolvedValue({
      configured: false,
      readable: false,
      limit: 25,
    });
    apiMock.autonomyEvents.mockResolvedValue({
      configured: false,
      readable: false,
      limit: 25,
    });

    render(<AutonomyPage />);

    const paperState = await screen.findByTestId("autonomy-paper-state");
    expect(paperState).toHaveTextContent("yapılandırılmadı");
    expect(screen.getByTestId("autonomy-orchestrator-state")).toHaveTextContent(
      "yapılandırılmadı",
    );
    expect(screen.getByTestId("autonomy-champions-state")).toHaveTextContent(
      "yapılandırılmadı",
    );
    expect(screen.getByTestId("autonomy-discoveries-state")).toHaveTextContent(
      "yapılandırılmadı",
    );
    expect(screen.getByTestId("autonomy-data-state")).toHaveTextContent("yapılandırılmadı");
    expect(await screen.findByTestId("autonomy-jobs-state")).toHaveTextContent(
      "yapılandırılmadı",
    );
    expect(await screen.findByTestId("autonomy-events-state")).toHaveTextContent(
      "yapılandırılmadı",
    );

    // No paper count card exists at all, so a zero cannot be read as data.
    expect(screen.queryByTestId("autonomy-paper-sessions")).not.toBeInTheDocument();
    expect(screen.queryByTestId("autonomy-orchestrator-queued")).not.toBeInTheDocument();
    expect(screen.queryByTestId("autonomy-discoveries-runs")).not.toBeInTheDocument();
  });

  it("renders unreadable stores as unreadable rather than empty", async () => {
    apiMock.autonomyStatus.mockResolvedValue(
      status({
        paper: { configured: true, readable: false },
        orchestrator: { configured: true, readable: false },
        champions: [],
        champions_readable: false,
        discoveries: { configured: true, readable: false },
        data: { configured: true, readable: false },
      }),
    );
    apiMock.autonomyJobs.mockResolvedValue({
      configured: true,
      readable: false,
      limit: 25,
    });
    apiMock.autonomyEvents.mockResolvedValue({
      configured: true,
      readable: false,
      limit: 25,
    });

    render(<AutonomyPage />);

    expect(await screen.findByTestId("autonomy-paper-state")).toHaveTextContent("okunamıyor");
    expect(screen.getByTestId("autonomy-orchestrator-state")).toHaveTextContent("okunamıyor");
    expect(screen.getByTestId("autonomy-champions-state")).toHaveTextContent("okunamıyor");
    expect(screen.getByTestId("autonomy-discoveries-state")).toHaveTextContent("okunamıyor");
    expect(screen.getByTestId("autonomy-data-state")).toHaveTextContent("okunamıyor");
    expect(await screen.findByTestId("autonomy-jobs-state")).toHaveTextContent("okunamıyor");
    expect(await screen.findByTestId("autonomy-events-state")).toHaveTextContent("okunamıyor");

    expect(screen.queryByTestId("autonomy-paper-sessions")).not.toBeInTheDocument();
    expect(screen.queryByTestId("autonomy-jobs-empty")).not.toBeInTheDocument();
    expect(screen.queryByTestId("autonomy-events-empty")).not.toBeInTheDocument();
  });

  it("renders a readable but empty store as real zeros and explicit empty text", async () => {
    apiMock.autonomyStatus.mockResolvedValue(
      status({
        paper: {
          configured: true,
          readable: true,
          sessions: 0,
          open_positions: 0,
          closed_trades: 0,
        },
        orchestrator: {
          configured: true,
          readable: true,
          queued: 0,
          running: 0,
          failed: 0,
          paused: false,
          safe_mode: false,
          research_disabled: false,
          last_event: null,
        },
        champions: [],
        discoveries: { configured: true, readable: true, runs: 0, decisions: 0 },
        data: { configured: true, readable: true, symbols: [] },
      }),
    );
    apiMock.autonomyJobs.mockResolvedValue(jobsPage({ items: [] }));
    apiMock.autonomyEvents.mockResolvedValue(eventsPage({ items: [] }));

    render(<AutonomyPage />);

    // Zero is shown only because the API answered with a read zero.
    expect(await screen.findByTestId("autonomy-paper-sessions")).toHaveTextContent("0");
    expect(screen.getByTestId("autonomy-orchestrator-queued")).toHaveTextContent("0");
    expect(screen.getByTestId("autonomy-orchestrator-paused")).toHaveTextContent("Hayır");
    expect(screen.getByTestId("autonomy-discoveries-runs")).toHaveTextContent("0");

    expect(screen.getByTestId("autonomy-champions-empty")).toHaveTextContent("NO TRADE");
    expect(screen.getByTestId("autonomy-data-empty")).toHaveTextContent(
      "Ay partition’ı olan sembol yok.",
    );
    expect(await screen.findByTestId("autonomy-jobs-empty")).toBeInTheDocument();
    expect(screen.getByTestId("autonomy-events-empty")).toBeInTheDocument();
  });

  it("renders a dash when a readable count is absent, not a zero", async () => {
    apiMock.autonomyStatus.mockResolvedValue(
      status({
        paper: { configured: true, readable: true },
        discoveries: { configured: true, readable: true },
      }),
    );

    render(<AutonomyPage />);

    expect(await screen.findByTestId("autonomy-paper-sessions")).toHaveTextContent("—");
    expect(screen.getByTestId("autonomy-paper-open-positions")).toHaveTextContent("—");
    expect(screen.getByTestId("autonomy-discoveries-runs")).toHaveTextContent("—");
    expect(screen.getByTestId("autonomy-paper-sessions")).not.toHaveTextContent("0");
  });

  it("announces errors and lets the operator retry every window", async () => {
    const failure = new Error("backend down");
    apiMock.autonomyStatus.mockRejectedValueOnce(failure);
    apiMock.autonomyJobs.mockRejectedValueOnce(failure);
    apiMock.autonomyEvents.mockRejectedValueOnce(failure);

    render(<AutonomyPage />);

    expect(await screen.findByTestId("autonomy-status-error")).toHaveTextContent(
      "Durum okunamadı",
    );
    expect(screen.getByTestId("autonomy-jobs-error")).toHaveTextContent("İşler okunamadı");
    expect(screen.getByTestId("autonomy-events-error")).toHaveTextContent(
      "Olaylar okunamadı",
    );

    fireEvent.click(screen.getByRole("button", { name: "Yenile" }));

    expect(await screen.findByTestId("autonomy-paper-sessions")).toHaveTextContent("2");
    expect(await screen.findByTestId("autonomy-job-status-12")).toBeInTheDocument();
    expect(screen.getByTestId("autonomy-events-table")).toBeInTheDocument();
    expect(screen.queryByTestId("autonomy-status-error")).not.toBeInTheDocument();
  });

  it("labels champion modes and warns that only evidence may trade paper", async () => {
    apiMock.autonomyStatus.mockResolvedValue(
      status({
        champions: [
          {
            scope: "XAUUSD:M5",
            candidate_id: "cand-evidence",
            mode: "evidence",
            promoted_at: NOW,
            policy_id: "policy-1",
            policy_digest: "digest-1",
          },
          {
            scope: "XAUUSD:M15",
            candidate_id: "cand-exploratory",
            mode: "exploratory",
            promoted_at: NOW,
          },
          {
            scope: "XAUUSD:H1",
            candidate_id: "cand-manual",
            mode: "manual",
            promoted_at: null,
          },
          {
            scope: "XAUUSD:H4",
            candidate_id: "cand-unknown",
            mode: "dsl",
            promoted_at: NOW,
          },
        ],
      }),
    );

    render(<AutonomyPage />);

    const evidence = await screen.findByTestId("autonomy-champion-mode-XAUUSD:M5");
    expect(evidence).toHaveAttribute("data-mode", "evidence");
    expect(evidence).toHaveTextContent("kanıt");

    const exploratory = screen.getByTestId("autonomy-champion-mode-XAUUSD:M15");
    expect(exploratory).toHaveAttribute("data-mode", "exploratory");
    expect(exploratory).toHaveTextContent("keşif");

    const manual = screen.getByTestId("autonomy-champion-mode-XAUUSD:H1");
    expect(manual).toHaveAttribute("data-mode", "manual");
    expect(manual).toHaveTextContent("manuel");

    const unknown = screen.getByTestId("autonomy-champion-mode-XAUUSD:H4");
    expect(unknown).toHaveAttribute("data-mode", "unknown");
    expect(unknown).toHaveTextContent("kanıtsız");

    const champions = screen
      .getByRole("heading", { name: "Şampiyonlar" })
      .closest("section") as HTMLElement;
    expect(champions).toHaveTextContent("Yalnızca evidence modundaki şampiyon paper işlem");
    expect(within(champions).getByTestId("autonomy-champion-row-XAUUSD:M5")).not.toHaveTextContent(
      "paper dışı",
    );
    expect(
      within(champions).getByTestId("autonomy-champion-row-XAUUSD:M15"),
    ).toHaveTextContent("paper dışı");
    expect(
      within(champions).getByTestId("autonomy-champion-row-XAUUSD:H1"),
    ).toHaveTextContent("paper dışı");
    expect(
      within(champions).getByTestId("autonomy-champion-row-XAUUSD:H4"),
    ).toHaveTextContent("paper dışı");
  });

  it("keeps the refresh control focusable while a read is in flight", async () => {
    let release: (value: AutonomyStatus) => void = () => {};
    apiMock.autonomyStatus.mockReturnValueOnce(
      new Promise<AutonomyStatus>((resolve) => {
        release = resolve;
      }),
    );

    render(<AutonomyPage />);

    const refresh = screen.getByTestId("autonomy-refresh");
    expect(refresh).not.toBeDisabled();
    expect(refresh).toHaveAttribute("aria-disabled", "true");

    refresh.focus();
    fireEvent.click(refresh);
    // The inflight guard runs instead of disabling the button and dropping focus.
    expect(apiMock.autonomyStatus).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(refresh);

    release(status());
    await waitFor(() => expect(refresh).toHaveAttribute("aria-disabled", "false"));
    expect(screen.getByTestId("autonomy-paper-sessions")).toHaveTextContent("2");
    expect(apiMock.autonomyStatus).toHaveBeenCalledTimes(1);
  });

  it("uses readable contrast classes and no text-loss utility", async () => {
    const { container } = render(<AutonomyPage />);

    await screen.findByTestId("autonomy-paper-sessions");

    // slate-500 was 3.93:1 on the surface; slate-400 is 7.29:1. Badge
    // backgrounds may keep loss/amber fills, but body text must not.
    expect(container.querySelectorAll(".text-slate-500")).toHaveLength(0);
    expect(container.querySelectorAll(".text-loss")).toHaveLength(0);
  });

  it("keeps the status table focused, captioned and live-announced", async () => {
    render(<AutonomyPage />);

    await screen.findByTestId("autonomy-paper-sessions");

    const live = screen.getByTestId("autonomy-status-live");
    expect(live).toHaveAttribute("role", "status");
    // `role="status"` already implies a polite live region; a second attribute
    // would be redundant and can double-announce in some screen readers.
    expect(live).not.toHaveAttribute("aria-live");
    expect(live).toHaveTextContent("Otonomi durumu güncellendi");

    const region = screen.getByRole("region", { name: /İşler — en yeni önce/ });
    expect(region).toHaveAttribute("tabindex", "0");
    expect(within(region).getByRole("table", { name: /İşler — en yeni önce/ })).toBeInTheDocument();
  });
});
