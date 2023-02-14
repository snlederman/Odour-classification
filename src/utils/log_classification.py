"""
logging model experiments metrics
"""

# packages
import os
import pandas as pd

def log_metrics(report, model, PROJECT_DIR):
    """
    takes the classification_report dict output
    and saves the metrics to the experiment logger
    """
    
    log_file = os.path.join(PROJECT_DIR, "data", "metrics_log.csv")
    metrics_log = pd.read_csv(log_file)
    
    report_df = pd.DataFrame(report)
    report_df["model"] = model.__class__.__name__
    report_df["log_id"] = max(metrics_log["log_id"]) + 1
    report_df.reset_index(inplace=True)
    report_df.rename(columns={"index" : "metric"}, inplace=True)
    
    metrics_log = pd.concat([metrics_log,report_df])
    metrics_log.to_csv(log_file, index=False)
    
    accuracy = report["accuracy"]
    report_df = pd.DataFrame(report)
    report_summary = report_df.transpose()[["precision","recall","f1-score"]].mean(axis=0)
    
    report_print = f"""Report:
    Accuracy: {accuracy:.0%}
    Precision: {report_summary["precision"]:.0%}
    Sensitivity: {report_summary["recall"]:.0%}
    F1 score: {report_summary["f1-score"]:.0%}
    """
    
    return report_print
