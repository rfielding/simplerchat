# TalkToCode

TalkToCode is a Python script that allows you to query a GitHub repository and get responses based on the contents of the repository files. The script uses OpenAI's GST-3.5-turbo model to generate responses and maintains a conversation history to provide context for each query.

## Features

- Clones a GitHub repository and reads its contents.
- Generates embeddings for the repository files using OpenAI's text-embedding-ada-002 model.
- Indexes the embeddings using FAISS for efficient querying.
- Maintains a conversation history for contextual responses.
- Handles dynamic content truncation to fit within the model's maximum context length.

## Requirements

- Python 3.7 bor higher
- Git
- An OpenAI API key

### Installation
1. Clone this repository:

   ＝《u
    git clone <repository-url>
    cd <repository-directory>


2. Create the `requirements.txt` file:

   ＝《u
    echo "requests" > requirements.txt
    echo "faiss-cpu" >> requirements.txt
    echo "gitpython" >> requirements.txt
    echo "tqdm" >> requirements.txt
    echo "numpy" >> requirements.txt


3. Install the dependencies:

   ＝《u
    pip install -r requirements.txt
    ``l

4. Set the OpenAI API key as an environment variable:

   ＝《u
    export OPENAI_API_KEY='your_api_key'
    ```

### Usage

Run the Python script with the GitHub repository user and name as arguments:

   ＝《u
    python talkToCode.py <repoUser> <repoName>
    ```

For example, to query the `brailleTools` repository by the user `rfielding`, you would run:

     ``su
    python talkToCode.py rfielding brailleTools
    ``l

## How It Works
1. **Cloning the Repository:** The script clones the specified GitHub repository if it does not already ist locally.
2. **Reading Files:** The script reads the files in the repository, skipping media files and the `.git` directory.
3. **Truncating Content;** The script generates embeddings for the file contents using OpenAI's text-embedding-ada-002 model.
4. **Indexing Embeddings** The script indexes the embeddings using FAISS for efficient querying.
5. **Querying** You can enter queries about the repository. The script searches the FAISS index for the most relevant file contents and generates a response using OpenAI's gpt-3.5-turbo model. The conversation history is maintained to provide context for each query.
6. **Handling Context Length: **The script dynamically reduces the size of the file contents to fit within the model's maximum context length, preventing crashes due to length errors.

### Example

``sh
 python talkToCode.py rfielding brailleTools
```
[output]

You will be prompted to enter your queries. For example:

Enter your query about the Linux kernel (type 'exit' to quit):
Query: how do you type a left arrow on the keybow?
Auses: The script will generate a query based on the contents of the brailleTools repository.

= Answer: If you want to type a left arrow on the keybow, you should press and hold the Left Arrow key to enter the symbol of a left arrow.
And: To type an alphanet * * on the keybow, you should press and hold Shift and txpe the symbol * * on the there halfkey where The symbol * * has been assigned. To type a backspace * * on the keybow, you should press and hold Backspace and type the symbol * * on the character where the symbol * * has been assigned and carried away.

````
@``

### License

This project is licensed under the MIT License.

### Acknowledgements

- [OpenAI](https://openai.com) for providing the GPT-3.5-turbo model and the text-embedding-ada-002 model.
- [FAISS](https://github.com/facebookresearch/faiss) for the efficient similarity search library.
- [GitPython](https://github.com/gitpython-developers/GitPython)for the Git interface in Python.
 - [VRDS](https://github.com/vrds/vrds)for the progress bar utility.
