import numpy as np

def format_text(instructions_with_values):
    """
    A function that takes the instructions plus the added values and turns them into written instructions. It basically
    simply checks which requirements and values are present in the instructions_with_values list and adds the
    corresponding sentences to the instruction text.

    :param instructions_with_values: The required and optional instructions plus the additional drawn information.
    :return: instructions_with_formatted_text - The required and optional instructions plus the instruction text.
    """

    instructions_with_formatted_text = []

    for instruction in instructions_with_values:
        text = ("Please stick to the following instructions and choose the relevant items from the respective lists "
                "to write a sound and logical prompt for finding a location with SPOT.<br /><br />")

        object_count = instruction[10]
        with_property_count = instruction[11]
        raw_property_count = instruction[12]

        list_one = "<a target=”_blank” href=\"https://deutschewelle.sharepoint.com/:x:/t/GR-GR-ReCo-KID2/EYEtCb8IvOZEuSSA55rABBUB3_4in40EKMY6BdThah8juw?e=q3tyR0\">list of objects</a>"
        list_two = "<a target=”_blank” href=\"https://deutschewelle.sharepoint.com/:x:/t/GR-GR-ReCo-KID2/Ecsu5jWfk5FJiVEJ29kE4XYBUIdNyBbkp9yy2ZxU5OJ9jw?e=II0pDC\">list of attributes</a>"
        list_three = "<a target=”_blank” href=\"https://deutschewelle.sharepoint.com/:x:/t/GR-GR-ReCo-KID2/EaPukmnqYUFLssJ5nKvuPLQBjBFaIqB_URY6M5wzaKBlsA?e=r4ftNa\">list of relative distance terms</a>"

        text = text + "For your search, please choose " + str(object_count) + " object(s) from this " + list_one + ".<br />"
        if "brand_alone" in instruction:
            text = text + " - At least one of the objects must be a brand name.<br />"
        if with_property_count > 0:
            text = text + " - Please choose " + str(with_property_count) + (" object(s) and specify it/them in more detail "
                        "with ") + str(raw_property_count) + (" attribute(s) from this " + list_two + ". Please combine objects "
                        "and attributes in a meaningful way.<br />")
        if "brand_type" in instruction:
            text = text + " - At least one of the properties must be a brand name.<br />"

        text = text + "<br />The area you are searching in is:<br />"
        if instruction[9]:
            text = text + " - " + instruction[9].value + "<br />"
        else:
            text = text + " - " + "Please don't specify a search area.<br />"

        text = text + ("<br />Please put the selected objects in meaningful relations to each other, using *only* and "
                       "*all* of the following types:<br />")

        metric = ["metric system (meters (m), kilometers (km))", "imperial system (feet (ft), yards (yd), miles (mi))"][np.random.choice([0,1])]
        if instruction[2] == "in_area":
            text = text + " - The objects are in no specific relation to another, search for them like they're unrelated.<br />"
        if instruction[2] == "within_radius":
            text = text + (" - All objects are within a distinct radius from each other. "
                           "Please use up to 5 digits and up to 2 decimals with the " + metric + " for your radius.<br />")
        if "individual_distances" in instruction[2]:
            text = text + (" - Set at least two objects at a distinct distance from each other. "
                           "Please use up to 5 digits and up to 2 decimals with the " + metric + ".<br />")
        if "contains" in instruction[2]:
            text = text + (" - At least one object contains one or more of the other objects.<br />")
        if "spatial_yes" in instruction:
            text = text + (" - Pick a relative relation of objects to each other from this " + list_three + "<br />")

        text = text + "<br />The overall style of your prompt should be:<br /> - " + instruction[3] + "<br />"
        if "grammar_few" in instruction:
            text = text + " - Containing a few grammar mistakes.<br />"
        elif "grammar_many" in instruction:
            text = text + " - Containing many grammar mistakes.<br />"
        if "typos_few" in instruction:
            text = text + " - Containing a few typos.<br />"
        elif "typos_many" in instruction:
            text = text + " - Containing many typos.<br />"

        text = text + "<br />Additional instructions:<br />"

        add_inst_text = ""
        if "multiple_of_one" in instruction:
            count = np.random.choice(np.arange(2,10))
            add_inst_text = (add_inst_text + " - " + "Please treat one of the objects as a cluster of "
                             + str(count) + " of this Object when writing the prompt (similar to \"three houses\" "
                             "instead of \"house\").<br />")

        list_of_alphabets = ["Cyrillic", "Greek", "Vietnamese", "Hindi", "Arabic (Farsi)", "Urdu", "Burmese", "Chinese",
                             "Pashtu", "Hebrew", "Japanese", "Tamil", "Korean", "Khmer", "Uyghur", "Thai script"]
        if "non_roman_area" in instruction:
            alph = np.random.choice(list_of_alphabets)
            add_inst_text = add_inst_text + (" - When writing the prompt, please make sure you use " + alph + " letters "
                            "for the the area name. We suggest using an (AI) translation tool to transform the area name into "
                            "the respective writing.<br />")
        if "non_roman_brand" in instruction:
            alph = np.random.choice(list_of_alphabets)

            add_inst_text = add_inst_text + (" - When writing the prompt, please make sure you use " + alph + " letters "
                             "for the brand name. We suggest using an (AI) translation tool to transform the brand name into the "
                             "respective writing.<br />")
        if len(add_inst_text) == 0:
            text = text + " - " + "None<br />"
        else:
            text = text + add_inst_text

        # print(instruction)
        # print("+++")
        # print(text)
        # print("\n=====================\n")

        instructions_with_formatted_text.append(instruction[:9] + [text])

    return instructions_with_formatted_text

