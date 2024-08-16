import requests
import csv

SEARCH_QUERY = 'PUBL=ACSNANO'
APIKEY = 'c6399dd4b73e9606f2af39a35b34a7ff269051db'

def retrieve_key_fields(document):
    try:
        return ({'UID': document['UID'],
                        'Title': document['static_data']['summary']['titles']['title'][-1]['content'],
                        'DOI': document['dynamic_data']['cluster_related']['identifiers']['identifier'][2]['value'],
                        'Keywords': getAllKeyWords(document)
                        })
    except IndexError:
        print("Something wrong during request, search produced no results?")
        return ({})

def getAllKeyWords(doc):
    keywords = []
    #Keywords plus is optional, handle accordingly:
    try:
        keywords += doc['static_data']["item"]["keywords_plus"]["keyword"] #Keywords from the "plus" bar right below search that would yield this document
    except:
        keywords
    #Add keywords that will always be there
    subjects = doc['static_data']["fullrecord_metadata"]['category_info']['subjects']['subject'] #subject keywords
    for s in subjects:
        keywords.append(s['content'])
    keywords += doc['static_data']["fullrecord_metadata"]['keywords']['keyword'] #metadata keywords
    keywords = tuple(keywords) #remove duplicates by converting list to tuple
    return keywords
        
def main():
    url = 'https://api.clarivate.com/api/wos?databaseId=WOS&usrQuery={SEARCH_QUERY}'
    headers={'X-ApiKey': APIKEY}
    initial_request = requests.get(url, headers=headers)
    print(initial_request)
    initial_json = initial_request.json()
    print(initial_json)
    
    records = []
    i = 0
    data = initial_json['Data']['Records']['records']['REC']
    for wos_document in data:
        records.append(retrieve_key_fields(wos_document))
        i += 1
    #write results into a csv
    keys = records[0].keys()
    with open('documents.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, keys)
        writer.writeheader()
        writer.writerows(records)
        


main()