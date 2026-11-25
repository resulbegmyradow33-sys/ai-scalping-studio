import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const navigation = vi.hoisted(() => ({ pathname: "/" }));

vi.mock("next/navigation", () => ({
  usePathname: () => navigation.pathname,
}));

const { Nav } = await import("./nav");

afterEach(cleanup);

describe("Nav", () => {
  it("links to the autonomy page", () => {
    navigation.pathname = "/paper";
    render(<Nav />);

    expect(screen.getByRole("link", { name: "Otonomi" })).toHaveAttribute(
      "href",
      "/autonomy",
    );
  });

  it("marks the autonomy page as current", () => {
    navigation.pathname = "/autonomy";
    render(<Nav />);

    expect(screen.getByRole("link", { name: "Otonomi" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("link", { name: "Paper Trading" })).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("links to the strategy intelligence page", () => {
    navigation.pathname = "/strategies";
    render(<Nav />);

    const link = screen.getByRole("link", { name: "Strateji Zekâsı" });
    expect(link).toHaveAttribute("href", "/strategies");
    expect(link).toHaveAttribute("aria-current", "page");
  });

  it("links to the backtesting page", () => {
    navigation.pathname = "/performance";
    render(<Nav />);

    expect(screen.getByRole("link", { name: "Backtesting" })).toHaveAttribute(
      "href",
      "/backtesting",
    );
  });
});
