# David Hu MTA Project Work
All Projects are listed below. Also check out other helpful files in this repo like the Kaggle ML Guide especially.


# MTA Topic Modeling and Visualization Pipeline

This Python script implements a comprehensive pipeline for text preprocessing, topic modeling, and visualization of topic distributions using a collection of documents in JSON Lines (`.jsonl`) format. The code is designed to preprocess text data, apply topic modeling, and generate various visualizations to explore and understand the underlying topics within the text corpus.

## Features

- **Text Preprocessing**: The script includes a pipeline for cleaning, tokenizing, lemmatizing, and normalizing text data using the spaCy NLP library.
- **Topic Modeling**: A topic modeling pipeline is implemented using Latent Dirichlet Allocation (LDA) to discover the underlying topics in the text corpus.
- **Coherence Score Evaluation**: The script iteratively tests different numbers of topics and evaluates the coherence scores to determine the optimal number of topics for the dataset.
- **Visualizations**: The script generates multiple visualizations, including t-SNE plots for dimensionality reduction, word clouds for individual topics, term frequency heatmaps, and document similarity matrices.

## How It Works

### 1. Loading the Data

The script loads a JSON Lines file (`.jsonl`) where each line is a JSON object representing a document. It extracts the text data (e.g., abstracts) for further processing.

### 2. Text Preprocessing Pipeline

The text data is passed through a preprocessing pipeline that performs the following steps:

- **Cleaning**: Removal of stop words and other unnecessary characters.
- **Tokenization**: Splitting text into individual tokens (words).
- **Lemmatization**: Reducing words to their base or root form.
- **Normalization**: Converting text to a consistent format (e.g., lowercasing, ASCII conversion).

### 3. Topic Modeling Pipeline

The preprocessed text is used to fit an LDA topic model. The script iterates over a range of topic numbers (from 2 to 20) to find the optimal number of topics based on coherence scores. The best model is then selected and refitted on the entire dataset.

### 4. Visualizations

- **t-SNE Visualization**: A t-SNE plot is generated to visualize the document-topic distributions in a 2D space.
- **Word Clouds**: Word clouds are created for each topic, illustrating the most frequent terms associated with each topic.
- **Term Frequency Heatmap**: A heatmap displays the frequency of the most common terms across the corpus.
- **Document Similarity Matrix**: A similarity matrix is computed and visualized to show the similarity between documents based on their topic distributions.

## Usage

### 1. Dependencies

The script relies on several Python libraries including `spaCy`, `scikit-learn`, `matplotlib`, `seaborn`, `WordCloud`, and `Gensim`. Make sure to install these dependencies using `pip`:

```bash
pip install spacy sklearn matplotlib seaborn wordcloud gensim
```
### 2. Running the Script

Place your `.jsonl` file containing the documents in the same directory as the script or specify the correct file path.  
Run the script to preprocess the text, perform topic modeling, and generate visualizations:

```bash
python your_script_name.py
```

### 3. Output

The script will generate and save the following visualizations in the current directory:

- `tsne_visualization.png`: A 2D t-SNE plot of document-topic distributions.
- `wordcloud_topic_X.png`: Word clouds for each identified topic (X represents the topic number).
- `term_frequency_heatmap.png`: A heatmap of the most frequent terms.
- `document_similarity_matrix.png`: A document similarity matrix based on topic distributions.

### Customization

- **Adjusting the Number of Topics**: You can modify the range of topics tested in the script by changing `range(2, 21)` to another range that fits your dataset.
- **Text Keys**: If your JSON Lines file uses a different key for the text data (e.g., `"content"` instead of `"abstract"`), adjust the `load_jsonl` function accordingly.
- **Stop Words**: The list of stop words used during preprocessing can be customized by modifying the `stop_words` parameter in the `CleanTextTransformer`.

### Example JSONL Structure

The .jsonl file should have a structure similar to this:

```json
{"title": "Title of Document 1", "abstract": "This is the abstract of document 1."}
{"title": "Title of Document 2", "abstract": "This is the abstract of document 2."}
...
```

Each line represents a single document, with the text data (e.g., "abstract") used for topic modeling.

### Conclusion

This script provides a robust framework for topic modeling and visualizing topics within a text corpus. It can be easily adapted to different datasets and customized to suit specific needs.

---

# Funder Extraction Script

This Python script is designed to process text data and extract information about funders and grants from a specified input file. It leverages regular expressions and spaCy's natural language processing capabilities to identify and categorize acknowledgments related to financial support.

#### Key Functions

- **`debug_Print(message)`**: Conditionally prints debug messages to both the console and a debug file (`debug.txt`). This function helps trace the execution of the script and understand intermediate results.

- **`extract_acknowledgements(prompt)`**: Extracts the acknowledgements section from HTML-formatted text. It uses regular expressions to locate and clean the acknowledgements content.

- **`sift_with_regex(text)`**: Utilizes regular expressions to identify phrases related to funding and grants. It parses these phrases to separate funders and grants, then organizes them into a dictionary.

- **`clean_funder_names_regex(name)`**: Cleans up funder names by removing unnecessary characters and patterns.

- **`sift_with_spacy(text)`**: Uses spaCy's NLP capabilities to identify funders and grants based on pre-defined patterns. This function matches various types of entities related to grants and funding.

- **`combine_results(acknowledgements)`**: Combines results from both regex and spaCy methods. It merges and cleans up any overlapping funders and grants to produce a consolidated result.

- **`clean_up_overlaps(combined_results)`**: Handles overlapping funder names by merging entries where necessary and removing redundancies.

- **`normalize_funder_name(funder)`**: Normalizes funder names by converting them to lowercase and removing leading articles to standardize comparisons.

- **`format_completion(funders)`**: Formats the extracted funders and grants into a readable text format.

- **`process_file(input_file_path, output_file_path)`**: Reads an input file line by line, processes each line to extract acknowledgements, identifies funders and grants, and writes the results to an output file.

#### Usage

1. **Set Debug Mode**: 
   - Set `debug = True` to enable debug print statements and log information to `debug.txt`.
   - Set `debug = False` for the actual run to suppress debug output.

2. **Run the Script**:
   - Ensure that the `funders_prompts.txt` file is in the same directory as the script.
   - Execute the script to process the file and generate the `funders_completed.txt` output file:
     ```bash
     python your_script_name.py
     ```

3. **Output**:
   - The script will generate an output file `funders_completed.txt` with processed acknowledgements and extracted funders and grants.

4. **Customization**:
   - Adjust regex patterns and spaCy match patterns as needed for different types of funding acknowledgements.
   - Modify input and output file paths in the `process_file` function if required.

This script provides a comprehensive solution for extracting and organizing funder and grant information from text data, leveraging both regex and NLP techniques to ensure accurate and detailed results. Please manually review them before feeding the results into BERT or using the YAML conversion script. This is meant to make the manual doccano evaluation easier and speed up the process.

---

# JSON to YAML Script

This script performs several data transformation tasks:

1. **Read and Parse JSON Lines from Text File:**
   - `read_txt_file(file_path)`: Reads a text file where each line is a JSON object. It parses these JSON lines into Python dictionaries and appends them to a list.

2. **Write JSON Data to File:**
   - `write_json_file(entries, output_path)`: Writes a list of JSON objects to a specified JSON file with pretty-printed formatting.

3. **Clean JSON Lines Formatting Issues:**
   - `clean_jsonl_file(input_file, output_file)`: Cleans up a JSON Lines file by removing unwanted characters and ensuring proper formatting. It reads each line, parses valid JSON objects, and writes them to a cleaned JSON Lines file.

4. **Convert JSON to YAML:**
   - `json_to_yaml(json_file_path, yaml_file_path)`: Converts a JSON file to a YAML file. It reads the JSON file and writes its content to a YAML file.

5. **Process and Clean YAML Entries:**
   - `convert_text(text)`: Converts escaped characters in text to their UTF-8 equivalents.
   - `modify_yaml_entry(entry)`: Updates the 'label' field to ensure it contains integer values for start and end positions, and converts the 'text' field using `convert_text`.
   - `process_yaml_file(input_file, output_file)`: Reads a YAML file, modifies each entry using `modify_yaml_entry`, and writes the processed data to a new YAML file.

### Usage

- **Read and Write JSON:**
  - To convert a text file containing JSON lines to a JSON file, use:
    ```python
    input_file_path = 'funders_completed_checked.txt'
    output_file_path = 'funders.json'
    entries = read_txt_file(input_file_path)
    write_json_file(entries, output_file_path)
    ```

- **Clean JSON Lines File:**
  - To clean a JSON Lines file and remove formatting issues:
    ```python
    input_file = 'funders_doccano.jsonl'
    output_file = 'cleaned_doccano.jsonl'
    clean_jsonl_file(input_file, output_file)
    ```

- **Convert JSON to YAML:**
  - To convert a cleaned JSON file to a YAML file:
    ```python
    json_file_path = 'cleaned_doccano.jsonl'
    yaml_file_path = 'funders.yaml'
    json_to_yaml(json_file_path, yaml_file_path)
    ```

- **Process and Clean YAML File:**
  - To clean and process a YAML file:
    ```python
    input_file = 'funders.yaml'
    output_file = 'funders_final.yaml'
    process_yaml_file(input_file, output_file)
    ```

This script ensures proper formatting and conversion between different data formats, making it suitable for preparing data for further analysis or integration.

---

# Web of Science API Extraction Script

This script retrieves data from the Web of Science API and processes it into a CSV file. It performs the following tasks:

1. **Define Constants:**
   - `SEARCH_QUERY`: The query parameter used for searching the Web of Science database.
   - `APIKEY`: The API key required to authenticate requests to the Web of Science API.

2. **Retrieve Key Fields from Document:**
   - `retrieve_key_fields(document)`: Extracts key information from a document, including:
     - UID
     - Title
     - DOI
     - Keywords (by calling the `getAllKeyWords` function)

3. **Get All Keywords:**
   - `getAllKeyWords(doc)`: Gathers all relevant keywords from the document, including:
     - Keywords from the "Keywords Plus" section
     - Subject keywords
     - Metadata keywords
   - Converts the list of keywords to a tuple to remove duplicates.

4. **Main Function:**
   - Constructs the API request URL using `SEARCH_QUERY`.
   - Sends a GET request to the Web of Science API with the provided `APIKEY`.
   - Parses the JSON response from the API.
   - Iterates through the returned records and retrieves key fields for each document.
   - Writes the extracted data to a CSV file (`documents.csv`).

### Usage

1. **Set Up Constants:**
   - Ensure `SEARCH_QUERY` and `APIKEY` are correctly set for your API query and authentication.

2. **Run the Script:**
   - The `main()` function makes an API request, processes the response, and saves the extracted data to `documents.csv`.

This script automates the process of querying the Web of Science API, extracting relevant fields from each document, and exporting the results to a CSV file for further analysis. Modify the search query words to the ones you desire, the official API documentation and helpful keywords are located in "API_HELP.txt".

---

This was made by David Hu. ACS Intern 2024. Please reach out to who.is.david101@gmail.com for questions, concerns, or debugging.
Thanks for the great summer of learning! Cheers.