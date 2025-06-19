def format_months(months: int): 
    if months % 12 == 0:
        return f"{months // 12} year{'s' if months > 12 else ''}"
    else:
        return f"{months} months"
    
def format_proba(proba: float):
    return f"{round(proba * 100, 1)}%"

def proba_to_phrase(proba: float):
    if proba < 0.25:
        return ":green[very unlikely]"
    elif proba < 0.5:
        return ":yellow[unlikely]"
    elif proba < 0.75:
        return ":orange[somewhat likely]"
    else:
        return ":red[likely]"