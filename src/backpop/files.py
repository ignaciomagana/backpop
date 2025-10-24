import ast
from configparser import ConfigParser

__all__ = ["parse_inifile"]

def _eval_div_only(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_div_only(node.operand)
        if not isinstance(v, (int, float)):
            raise ValueError("Unary +/- only allowed on numbers")
        return +v if isinstance(node.op, ast.UAdd) else -v

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval_div_only(node.left)
        right = _eval_div_only(node.right)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Division operands must be numeric")
        return left / right

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_div_only(elt) for elt in node.elts]

    if isinstance(node, ast.Expr):
        return _eval_div_only(node.value)

    raise ValueError(f"Unsupported construct: {ast.dump(node, include_attributes=False)}")

def parse_inifile(ini_file):
    """Parse BackPop and COSMIC configurations from an ini file

    Parameters
    ----------
    ini_file : str
        Path to the ini file

    Returns
    -------
    config : dict
        Dictionary of BackPop configuration parameters
    flags : dict
        Dictionary of BSE flags
    obs : dict
        Dictionary of observations with keys "mean", "sigma", "name", and "out_name"
    var : dict
        Dictionary of variable parameters with keys "min", "max", and "name"
    fixed : dict
        Dictionary of fixed parameters with parameter names as keys and values as values
    """
    config_file = ConfigParser()
    config_file.read(ini_file)
    config_dict = {section: dict(config_file.items(section)) for section in config_file.sections()}

    config = config_dict["backpop"]
    for k in ["n_threads", "n_eff", "n_live"]:
        config[k] = int(config[k])
    for k in ["verbose", "resume", "use_bcm"]:
        config[k] = config[k].lower() in ["true", "1", "yes"]

    # make sure all flags are the correct type
    flags = config_dict["bse"]
    for k, v in flags.items():
        flags[k] = _eval_div_only(ast.parse(v, mode='eval').body)

    # convert ini file inputs to observations, variables, and fixed parameters
    obs = {
        "mean": [],
        "sigma": [],
        "name": [],
        "log": []
    }
    var = {
        "min": [],
        "max": [],
        "name": [],
        "log": []
    }
    fixed = {}
    for k in config_dict:
        if k.startswith("backpop.var::"):
            var_name = k.split("backpop.var::")[-1]
            var["name"].append(var_name)
            var["min"].append(float(config_dict[k]["min"].strip()))
            var["max"].append(float(config_dict[k]["max"].strip()))
            var["log"].append(config_dict[k].get("log", "False").strip().lower() == "true")
        if k.startswith("backpop.obs::"):
            obs_name = k.split("backpop.obs::")[-1]
            obs["name"].append(obs_name)
            obs["mean"].append(float(config_dict[k]["mean"].strip()))
            obs["sigma"].append(float(config_dict[k]["sigma"].strip()))
            obs["log"].append(config_dict[k].get("log", "False").strip().lower() == "true")
        if k.startswith("backpop.fixed::"):
            fixed_name = k.split("backpop.fixed::")[-1]
            fixed[fixed_name] = float(config_dict[k]["value"].strip())

    return config, flags, obs, var, fixed