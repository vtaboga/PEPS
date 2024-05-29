import re
import ast


def add_id_to_config(file_path, model_name, zone_id):
    modified_lines = []
    prefix = f"zone{zone_id}/"  # Changed to use prefix instead of suffix

    try:
        # Open and read the configuration file
        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith(model_name) and '=' in stripped_line:
                    # Replace the model name with config{n}/model_name
                    parts = stripped_line.split('=', 1)
                    modified_line = parts[0].replace(model_name, prefix + model_name) + '=' + parts[1]
                    modified_lines.append(modified_line)
                else:
                    # Add unmodified line
                    modified_lines.append(stripped_line)

        # Join all modified lines into a single string with newlines
        return "\n".join(modified_lines) + "\n"

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def parse_controller_config(file_path, controller_name, model_name, zone_id):
    """
    Parses a GIN configuration file and updates the controller settings by appending .config{config_id}
    to each controller_name occurrence and modifies the prediction_model parameter to add config{n}/Model.

    :param file_path: Path to the original GIN configuration file.
    :param controller_name: Name of the controller in the configuration file.
    :param model_name: Name of the model referenced in the configuration.
    :param config_id: Integer to append in the format config{config_id}/.
    :return: A string containing the modified configuration.
    """
    modified_lines = []
    config_prefix = f"zone{zone_id}/"  # Use a prefix instead of suffix for the scope

    try:
        # Open and read the configuration file
        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith(controller_name) and '=' in stripped_line:
                    # Modify the controller name with config{n}/controller_name
                    parts = stripped_line.split('=', 1)
                    modified_line = parts[0].replace(controller_name, config_prefix + controller_name) + '='
                    if 'prediction_model' in parts[0]:
                        # Also prepend config{n}/ to the model name in prediction_model parameter
                        modified_line += f' @{config_prefix + model_name}'
                    else:
                        modified_line += parts[1]
                    modified_lines.append(modified_line)
                else:
                    # Add unmodified line
                    modified_lines.append(stripped_line)

        # Join all modified lines into a single string with newlines
        return "\n".join(modified_lines) + "\n"

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def query_gin_parameter(file_path, parameter_name):
    """
    Searches for a specific parameter in a GIN configuration file and returns its value.

    :param file_path: Path to the GIN configuration file.
    :param parameter_name: The name of the parameter to search for.
    :return: The value of the parameter if found, otherwise None.
    """
    # Define a regular expression pattern to match the parameter and capture its value
    # This pattern handles possible spaces around '=', and captures values even with inline comments
    pattern = re.compile(r'\b{}\b\s*=\s*(.+?)(\s*#.*)?$'.format(re.escape(parameter_name)))

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    # Found the parameter, return its value (stripping potential inline comments)
                    return match.group(1).strip()
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return None


def convert_string_to_list(string_representation):
    """
    Converts a string representation of a list into a Python list using ast.literal_eval.

    :param string_representation: String representation of a list.
    :return: The actual Python list object.
    """
    try:
        # Convert the string representation to a Python list
        result = ast.literal_eval(string_representation)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("The input string does not represent a list.")
    except (SyntaxError, ValueError) as e:
        print(f"Error converting string to list: {e}")
        return None
