def to_camel(string: str) -> str:
    components = string.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def to_snake(string: str) -> str:
    return ''.join(['_' + c.lower() if c.isupper() else c for c in string]).lstrip('_')