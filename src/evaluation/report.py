import os
from datetime import datetime
import logging


def generate_report(metrics: dict, config: dict, output_dir: str, fmt: str = "txt"):
    os.makedirs(output_dir, exist_ok=True)

    model_name = config.get("model_name") or config.get("model", {}).get("name") or "unknown_model"
    output_path = config.get("training", {}).get("output_dir") if isinstance(config.get("training"), dict) else None
    predictions_file = config.get("predictions_file", "N/A")
    generated_at = config.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    lines = [
        f"Evaluation Report — {model_name}",
        "=" * 50,
        f"Generated: {generated_at}",
        f"Predictions file: {predictions_file}",
        f"Output directory: {output_path or output_dir}",
        "",
        "Metrics:",
        "-" * 50,
    ]
    for k, v in metrics.items():
        lines.append(f"{k.capitalize():15}: {v:.4f}")

    report_name = f"evaluation_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{fmt}"
    report_path = os.path.join(output_dir, report_name)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logging.info(f"Отчёт сохранён в {report_path}")
    return report_path
