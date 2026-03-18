import ast
from configparser import ConfigParser
from .consts import BPP_COLUMNS, BCM_COLUMNS

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
        
    Raises
        ------
        ValueError
            If 'bpp_columns' or 'bcm_columns' provided are not in BPP_COLUMNS or BCM_COLUMNS 
            and 'bpp_columns' do not include the observable constraints
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
        "log": [],
        "out_name": []
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
            obs["out_name"].append(config_dict[k]["out_name"])
            obs["mean"].append(float(config_dict[k]["mean"].strip()))
            obs["sigma"].append(float(config_dict[k]["sigma"].strip()))
            obs["log"].append(config_dict[k].get("log", "False").strip().lower() == "true")
        if k.startswith("backpop.fixed::"):
            fixed_name = k.split("backpop.fixed::")[-1]
            fixed[fixed_name] = float(config_dict[k]["value"].strip())
    
    if config["bpp_columns"] != "" and config["bpp_columns"] != "None":
        config["bpp_columns"] = eval(config["bpp_columns"])
        for k in config["bpp_columns"]:
            if k not in BPP_COLUMNS:
                raise ValueError(f'Invalid column name: {k}. '
                                 f'Not found in BPP columns: {BPP_COLUMNS}')

        # make sure bpp_columns includes observables
        for k in obs["out_name"]:
            if k not in config["bpp_columns"]:
                raise ValueError(f'Missing column: {k}. You must provide BPP column names '
                                 f'that match observables: {obs["out_name"]}')
    else:
        config["bpp_columns"] = BPP_COLUMNS
        
    if use_bcm:
        if config["bcm_columns"] != "" and config["bcm_columns"] != "None":
            # make sure bcm_columns names are found in BCM_COLUMNS
            config["bcm_columns"] = eval(config["bcm_columns"])
            for k in config["bcm_columns"]:
                if k not in BCM_COLUMNS:
                    raise ValueError(f'Invalid column name: {k}. '
                                     f'Not found in BPP columns: {BCM_COLUMNS}')

            # make sure bcm_columns includes observables
            for k in obs["out_name"]:
                if k not in config["bcm_columns"]:
                    raise ValueError(f'Missing column: {k}. You must provide BCM column names '
                                     f'that match observables: {obs["out_name"]}')
        else:
            config["bcm_columns"] = BCM_COLUMNS
        

    return config, flags, obs, var, fixed