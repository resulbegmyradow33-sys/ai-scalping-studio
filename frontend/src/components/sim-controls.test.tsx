import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SimControls } from "./sim-controls";

// vitest runs without globals, so testing-library cannot auto-register cleanup.
afterEach(cleanup);

describe("SimControls", () => {
  it("starts with the selected speed", () => {
    const onStart = vi.fn();
    render(<SimControls status="idle" onStart={onStart} onControl={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /100×/ }));
    fireEvent.click(screen.getByRole("button", { name: "Başlat" }));
    expect(onStart).toHaveBeenCalledWith(100);
  });
  it("disables start while running", () => {
    render(<SimControls status="running" onStart={vi.fn()} onControl={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Başlat" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Durdur" })).toBeEnabled();
  });
  it("shows the synthetic data warning", () => {
    render(<SimControls status="idle" onStart={vi.fn()} onControl={vi.fn()} />);
    expect(screen.getByText(/veri kaynağı: sentetik/i)).toBeInTheDocument();
  });
  it("stops a running episode and can continue it afterwards", () => {
    const onControl = vi.fn();
    const { rerender } = render(
      <SimControls status="running" onStart={vi.fn()} onControl={onControl} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Durdur" }));
    expect(onControl).toHaveBeenCalledWith("stop");

    rerender(<SimControls status="stopped" onStart={vi.fn()} onControl={onControl} />);
    const resume = screen.getByRole("button", { name: "Devam" });
    expect(resume).toBeEnabled();
    fireEvent.click(resume);
    expect(onControl).toHaveBeenCalledWith("resume");
  });
});
