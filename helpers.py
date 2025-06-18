def format_months(months: int): 
    if months % 12 == 0:
        return f"{months // 12} year{'s' if months > 12 else ''}"
    else:
        return f"{months} months"
    
def format_proba(proba: float):
    return f"{round(proba, 3) * 100}%"