#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def generate_sentence():
    sw = pd.read_csv("docs/sweep_results.csv", comment="#")

    b64 = sw[(sw["filter"] == "Noisy_Baseline") & (sw["size"] == "64^3")].iloc[0]
    b128 = sw[(sw["filter"] == "Noisy_Baseline") & (sw["size"] == "128^3")].iloc[0]

    g64 = sw[(sw["filter"] == "Gaussian") & (sw["mode"] == "3d") & (sw["size"] == "64^3") & (sw["param_str"] == "sigma=0.4;r=2")].iloc[0]
    g128 = sw[(sw["filter"] == "Gaussian") & (sw["mode"] == "3d") & (sw["size"] == "128^3")].iloc[0]

    g64_05 = sw[(sw["filter"] == "Gaussian") & (sw["mode"] == "3d") & (sw["size"] == "64^3") & (sw["param_str"] == "sigma=0.5;r=2")].iloc[0]
    g64_08 = sw[(sw["filter"] == "Gaussian") & (sw["mode"] == "3d") & (sw["size"] == "64^3") & (sw["param_str"] == "sigma=0.8;r=3")].iloc[0]
    g64_12 = sw[(sw["filter"] == "Gaussian") & (sw["mode"] == "3d") & (sw["size"] == "64^3") & (sw["param_str"] == "sigma=1.2;r=4")].iloc[0]

    sentence = (
        f"- **Gaussian Blur:** While the minimal smoothing setting ($\\sigma=0.4, r=2$) achieved a modest improvement in both PSNR "
        f"(${g64['psnr_mean']:.2f}\\text{{ dB}}$ at $64^3$ and ${g128['psnr_mean']:.2f}\\text{{ dB}}$ at $128^3$ vs. baseline ${b64['psnr_mean']:.2f}\\text{{ dB}}$ / ${b128['psnr_mean']:.2f}\\text{{ dB}}$) "
        f"and structural similarity ($64^3$: {b64['ssim_mean']:.4f} $\\to$ {g64['ssim_mean']:.4f}; $128^3$: {b128['ssim_mean']:.4f} $\\to$ {g128['ssim_mean']:.4f}), "
        f"it expanded the planar edge width beyond the discrete sampling floor (${g64['edge_width_mean']:.3f}$ / ${g128['edge_width_mean']:.3f}$ vx vs. baseline ${b64['edge_width_mean']:.3f}$ / ${b128['edge_width_mean']:.3f}$ vx). "
        f"All moderate and stronger blurs ($\\sigma \\ge 0.5$) degraded PSNR significantly below the noisy input "
        f"(e.g., $\\sigma=0.5 \\to {g64_05['psnr_mean']:.2f}\\text{{ dB}}$, $\\sigma=0.8 \\to {g64_08['psnr_mean']:.2f}\\text{{ dB}}$, $\\sigma=1.2 \\to {g64_12['psnr_mean']:.2f}\\text{{ dB}}$), "
        f"confirming that linear smoothing wipes out fine CT boundaries."
    )
    return sentence

def update_report():
    report_path = Path("docs/EVALUATION_REPORT.md")
    content = report_path.read_text(encoding="utf-8")
    
    new_sentence = generate_sentence()
    
    # Target existing Gaussian Blur sentence under Section 2
    pattern = r"- \*\*Gaussian Blur:\*\* While the minimal smoothing setting .*?wipes out fine CT boundaries\."
    
    match = re.search(pattern, content)
    if not match:
        raise RuntimeError("Could not find Gaussian Blur sentence pattern in report")
        
    updated = content[:match.start()] + new_sentence + content[match.end():]
    report_path.write_text(updated, encoding="utf-8")
    print("Successfully updated EVALUATION_REPORT.md with sentence generated from CSV:")
    print(new_sentence)

if __name__ == "__main__":
    import re
    update_report()
