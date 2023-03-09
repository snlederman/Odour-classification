"""
logging model experiments metrics
"""

# packages
import os
import pandas as pd

def log_metrics(report, args, project_dir):
    """
    takes the classification_report dict output
    and saves the metrics to the experiment logger
    """
    
    log_file = os.path.join(project_dir, "data", "stats", "metrics_log.csv")
    metrics_log = pd.read_csv(log_file)
    
    report_df = pd.DataFrame(report)
    report_df["log_id"] = max(metrics_log["log_id"]) + 1
    for key, value in args.items():
<<<<<<< HEAD
        report_df[key] = value
=======
        report[key] = value
>>>>>>> 727a9d9e0952ec00afa0a0373e6e0b9c9884c47d
        
    report_df.reset_index(inplace=True)
    report_df.rename(columns={"index" : "metric"}, inplace=True)
    
    metrics_log = pd.concat([metrics_log,report_df])
    metrics_log.to_csv(log_file, index=False)

    # returns nothing
