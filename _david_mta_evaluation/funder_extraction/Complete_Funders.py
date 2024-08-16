import re
import json
import spacy
from spacy.matcher import Matcher

debug = True #Set to False for actual run (no print statements/debug file)
hasRan = True #Do not print detailed output past first entry

#method to print conditionally for debug purposes + store result in more readable file
def debug_Print(message):
    global debug, hasRan
    if debug:
        debug_file = open('debug.txt', 'a', encoding='utf-8')
        debug_file.write(message + '\n')
        if not hasRan:
            print(message)
    
def extract_acknowledgements(prompt):
    # Extract the acknowledgements section
    acknowledgements_section = re.search(r'<div.*?Acknowledgments.*?</div>.*?<div.*?><p>(.*?)</p></div>', prompt, re.IGNORECASE | re.DOTALL)
    if not acknowledgements_section:
        return None
    acknowledgements = acknowledgements_section.group(1)
    acknowledgements = acknowledgements.strip()
    
    debug_Print('Acknowledgements' + '\n' + acknowledgements + '\n')
    
    return acknowledgements

def sift_with_regex(text):
    global hasRan
    debug_Print('Sifting Data w/ Regex')

    # Extract the phrases where parties acknowledged
    mega_regex = r'(support([a-z]*\b) (?:in part )*(by|from) (.*)|provided by(.*)|funded by(.*)|financial contribution from(.*)|funding (.*)|monetary support(.*)|.\s((.*) for funding\s.*)(?! through)|(.*)for financial support\s\.*)\.'
    phrases_pattern = re.compile(mega_regex, re.IGNORECASE | re.DOTALL)
    phrases = phrases_pattern.findall(text)
    debug_Print('Step 1: Phrases with funders/grants' + '\n' + str(phrases))
    
    # Parse the phrases into distinct parties (funders and grants)
    parties = []
    for match in phrases:
        # Split by commas or periods followed by space or end of string
        parts = re.split(r'([,;]\s+|\sand the\s)', str(match))
        for part in parts:
            # Find the substring starting from the first capital letter
            pattern = re.compile(r'[A-Z].*')
            substring_match = pattern.search(part)
            
            if substring_match:
                substring = substring_match.group(0)
                parties.append(substring)  # Append the processed substring to the funder_list
    debug_Print('Step 2: Associated Parties List' + '\n' + str(parties))
    
    # split and divide each party into funders or grants into a dictionary of the form {FUNDER: [Grant1, Grant2, ...]}
    funders = dict()
    for party in parties:
        grant_pattern = re.compile(r'\b[A-Za-z]*[0-9\/]+[^) .,;\'\"]*\b')
        grant_matches = grant_pattern.findall(party)
        funder_name = grant_pattern.sub('', party)
        # clean up funder names list from leftover characters
        clean_funder_name = clean_funder_names_regex(funder_name)
        funders[clean_funder_name] = grant_matches
    
    debug_Print('Step 3: Separating Grants and Funders')
    debug_Print('Funders: ' + str(funders.keys()))
    debug_Print('Grants: ' + str(funders.values()) + '\n')

    #hasRan = True #comment out to see print outputs for all entries
    return funders

def clean_funder_names_regex(name):
    # Remove text patterns that don't contribute to funder names
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'[+,.\(\);:\'\"\/!@#$%\^&\*_~`={}\[\]<>\?/]', '', name)
    name = re.sub(r'(((grant)\s(agreement)*)\s*(no)*)(?!at)', '', name, re.IGNORECASE)
    name = name.strip()
    return name

def sift_with_spacy(text):
    debug_Print('Sifting with Spacy: ')
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    funder_grant_dict = {}
    patterns = [
        ("GRANT1", [{"SHAPE": "XXddddddddd"}]),  # Matches patterns like LY21E030011
        ("GRANT2", [{"SHAPE": "XXXXdddddddddddddd"}]),  # Matches longer patterns like RCBS20200714114910141
        ("MONEY_GRANT", [{"ENT_TYPE": "MONEY"}]),  # Matches entities labeled as MONEY
        ("WORK_OF_ART_GRANT", [{"ENT_TYPE": "WORK_OF_ART"}]),  # Matches entities labeled as WORK_OF_ART
        ("CARDINAL_GRANT", [{"ENT_TYPE": "CARDINAL"}]),  # Matches entities labeled as CARDINAL
        ("PRODUCT_GRANT", [{"ENT_TYPE": "PRODUCT"}]),  # Matches entities labeled as PRODUCT
        ("DATE_GRANT", [{"ENT_TYPE": "DATE"}])  # Matches entities labeled as DATE
    ]


    debug_Print("Step 1: Match grant patterns using phrase matcher")
    matcher = Matcher(nlp.vocab)
    for label, pattern in patterns:
        matcher.add(label, [pattern])
    debug_Print('Matcher: ' + str(matcher))

    debug_Print("Step 2: Split acknowledgement section into parts (sentences), find funder org, add matched grants to respective funder org")
    matches = matcher(doc)
    debug_Print('Matches: ' + str(matches))
    current_funder = None

    #First Pass: Iterate over all org entities
    for ent in doc.ents:
        if ent.label_ == "ORG":
            current_funder = ent.text
            if current_funder not in funder_grant_dict:
                funder_grant_dict[current_funder] = []

    for _, start, end in matches:
        span = doc[start:end]

        for ent in reversed(doc.ents):
            if ent.label_ == "ORG" and ent.start < start:
                current_funder = ent.text
                break
        
        if current_funder and current_funder in funder_grant_dict:
            funder_grant_dict[current_funder].append(span.text)

    debug_Print('Funders: ' + str(funder_grant_dict.keys()))
    debug_Print('Grants: ' + str(funder_grant_dict.values()) + '\n')

    return funder_grant_dict

def combine_results(acknowledgements):
    combined_dict = {}
    #retrieve extraction results
    regex_result = sift_with_regex(acknowledgements)
    spacy_result = sift_with_spacy(acknowledgements)

    #Get (unique) keys from both results
    all_funders = set(regex_result.keys()).union(set(spacy_result.keys()))

    #consolidate grants
    for funder in all_funders:
        combined_grants = set()
        if funder in regex_result:
            combined_grants.update(regex_result[funder])
        if funder in spacy_result:
            combined_grants.update(spacy_result[funder])
        combined_dict[funder] = list(combined_grants)

    #clean up overlaps
    combined_dict = clean_up_overlaps(combined_dict)

    debug_Print('Combined results')
    debug_Print('Funders: ' + str(combined_dict.keys()))
    debug_Print('Grants: ' + str(combined_dict.values()) + '\n')

    return combined_dict

def clean_up_overlaps(combined_results):
    cleaned_results = dict()

    #sort funders by length to handle longer names first; we want to consolidate overlaps into the shorter name (arbitrary choice)
    sorted_funders = sorted(combined_results.keys(), key=len, reverse=True)

    merged_funders = set()

    for funder in sorted_funders:
        if funder in merged_funders:
            continue

        #Flag to check if funder should be added to cleaned_results
        add_funder = True
        
        for other_funder in sorted_funders:
            if funder != other_funder and (
                normalize_funder_name(funder) in normalize_funder_name(other_funder)):
                #Merge grants
                combined_results[funder] += combined_results[other_funder]
                #Mark other funder as merged
                merged_funders.add(other_funder)
                #Set flag to false since funder is merged
                add_funder = False
        
        #Add to cleaned dictionary if not merged
        if add_funder and funder not in merged_funders:
            cleaned_results[funder] = combined_results[funder]

    return cleaned_results

def normalize_funder_name(funder):
    """
    Normalize funder name by converting to lowercase and removing leading articles.
    """
    normalized = funder.lower()
    for article in ['the', 'an', 'a']:
        if normalized.startswith(article + ' '):
            normalized = normalized[len(article) + 1:]  # Remove article and space
            break
    return normalized

def format_completion(funders):
    completion_text = "Funder:"
    for funder in funders:
        completion_text += f"\n- name: {funder}\n grant: "
        for grant in funders[funder]:
            completion_text += f"{grant}, "
        if len(funders[funder]) > 0:
            completion_text = completion_text[:-2] #get rid of that last pesky comma
    completion_text += "\n#END#"
    return completion_text

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            entry = json.loads(line.strip())  # Load each line as a JSON object
            prompt = entry['prompt']
            debug_Print('Prompt: ' + str(prompt))
            acknowledgements = extract_acknowledgements(prompt)
            funders = combine_results(acknowledgements)
            if funders:
                completion = format_completion(funders)
                debug_Print('Completion: \n' + str(completion) + '\n')
                entry['completion'] = completion
                outfile.write(json.dumps(entry) + '\n')
            else:
                outfile.write(json.dumps(entry) + '\n')
    infile.close()
    outfile.close()

process_file('funders_prompts.txt', 'funders_completed.txt')
print('Done!')