"""
summeries classification report for multiclass preblem

Returns:
    str: summary metrics
"""

import pandas as pd

def summeries_multiclass_report(report):
    """
    takes the classification_report dict output
    and summeries it for metrics comparison
    """
    accuracy = report["accuracy"]
    report_df = pd.DataFrame(report)
    report_summary = report_df.iloc[:,:8].transpose()[["precision","recall","f1-score"]].mean(axis=0)

    report_print = f"""Report:
    Accuracy: {accuracy:.0%}
    Precision: {report_summary["precision"]:.0%}
    Sensitivity: {report_summary["recall"]:.0%}
    F1 score: {report_summary["f1-score"]:.0%}
    """
    
    return report_print