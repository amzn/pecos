#!/usr/bin/env python3
import os
import json

def input_with_default(user_input, default_val):
    if not user_input:
        return default_val
    return user_input

def int_input_value_check(user_input, default_val, max_val, min_val):
    int_input = int(input_with_default(user_input, default_val))
    if int_input > max_val or int_input < min_val:
        raise ValueError(f"Input should be in [{min_val}, {max_val}], got: {int_input}")
    return int_input

def get_parameters():
    param_dict = {}
    param_dict["account"] = None
    while not param_dict["account"]:
        param_dict["account"] = input("Please enter AWS 12-digit account ID (cannot be empty): ")
        if not (param_dict["account"].isdigit() and len(param_dict["account"]) == 12):
            print(f"AWS account ID should be integer and have 12 digits, got: {param_dict['account']}")
            param_dict["account"] = None

    param_dict["region"] = input_with_default(
        input("Please enter AWS region. The default is us-east-1: "),
        "us-east-1"
    )

    param_dict["user_name"] = input_with_default(
        input(
            f"Please enter your name for tagging AWS stacks. "
            f"The default is current OS user: "
        ),
        os.getlogin()
    )

    param_dict["user_disk_gb_req"] = int_input_value_check(
        input(
            f"Please enter disk size requirement(GB) for each node. Range 1GB ~ 15TB.\n"
            f"PECOS training recommendations: >=1000GB(1TB).\n"
            f"The default is 1000: "
        ),
        default_val=1000,
        max_val=15000,
        min_val=1
    )

    # dump json
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "param_config.json"), "w") as fp:
        json.dump(param_dict, fp)


if __name__ == "__main__":
    get_parameters()
