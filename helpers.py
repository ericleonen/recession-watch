def format_months(months: int) -> str: 
    """
    Returns a formatted phrase given months. Turns 12, 24, etc. months into 1, 2 and 3 years
    respectively.
    """
    if months % 12 == 0:
        return f"{months // 12} year{'s' if months > 12 else ''}"
    else:
        return f"{months} months"
    
def format_proba(proba: float) -> str:
    """
    Returns a formatted phrase given a probability. For example, 0.345 becomes 34.5%.
    """
    return f"{round(proba * 100, 1)}%"

def proba_to_phrase(proba: float):
    """
    Returns a formatted phrase describing a recession's likelihood given a probability.
    """
    if proba < 0.25:
        return ":green[very unlikely]"
    elif proba < 0.5:
        return ":blue[unlikely]"
    elif proba < 0.75:
        return ":orange[somewhat likely]"
    else:
        return ":red[likely]"
    
def format_pred_phrase(proba: float, top_features: list[str]):
    """
    Returns a formatted phrase describing a recession's likelihood and the most influential
    features.
    """
    top_features = [feature[0].lower() + feature[1:] for feature in top_features]
    if len(top_features) == 1:
        features_phrase = top_features[0]
    else:
        features_phrase = ", ".join(top_features[:-1]) + f", and {top_features[-1]}"
    return f"A recession is **{proba_to_phrase(proba)}** due to trends in {features_phrase}."