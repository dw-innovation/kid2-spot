import yaml
# from jsonschema import validate

SCHEMA = {
    'type': 'object',
    'properties': {
        'area': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'type': {'type': 'string'},
            },
            'required': ['name', 'type']
        }
    },
    'required': ['area']
}


def validate_and_fix_yaml(yaml_text):
    try:
        result = yaml.safe_load(yaml_text)
        # validate(instance=result, schema=SCHEMA)
        return result
    except yaml.parser.ParserError as e:
        print(f"fixing error: {e}")
        line_num = e.problem_mark.line
        # column_num = e.problem_mark.column
        lines = yaml_text.split('\n')

        misformatted_line = lines[line_num]
        if "entities" or "relations" in lines[line_num]:
            corrected_line = misformatted_line.strip()
            yaml_text = yaml_text.replace(misformatted_line, corrected_line)
            return validate_and_fix_yaml(yaml_text)
    except yaml.composer.ComposerError as e:
        print(f"fixing error: {e}")
        line_num = e.problem_mark.line
        # column_num = e.problem_mark.column
        lines = yaml_text.split('\n')

        if "value" in lines[line_num]:
            tag = lines[line_num].split(":")
            tag_value = tag[1].strip()
            fixed_tag_value = "\"" + tag_value + "\""
            yaml_text = yaml_text.replace(tag_value, fixed_tag_value)
            return validate_and_fix_yaml(yaml_text)

    except yaml.scanner.ScannerError as e:
        print(f"fixing error: {e}")
        line_num = e.problem_mark.line

        # column_num = e.problem_mark.column
        lines = yaml_text.split('\n')

        misformatted_line = lines[line_num]
        if "value" and "id" in lines[line_num]:
            corrected_line = misformatted_line.replace("id:", "\n id:")
            yaml_text = yaml_text.replace(misformatted_line, corrected_line)
            return validate_and_fix_yaml(yaml_text)


