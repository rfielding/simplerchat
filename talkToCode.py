import os
import sys
import requests
import faiss
import git
import numpy as np
from tqdm import tqdm
import time

# Read OpenAI API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
  raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Get repository user and name from command-line arguments
if len(sys.argv) != 3:
  print("Usage: python talkToCode.py <repoUser> <repoName>")
  sys.exit(1)

repo_user = sys.argv[1]
repo_name = sys.argv[2]

# Construct repository URL and path
repo_url = f"https://github.com/{repo_user}/{repo_name}.git"
repo_path = repo_name

# Clone the repository if it doesn't exist
if not os.path.exists(repo_path):
  print("Cloning the repository...")
  git.Repo.clone_from(repo_url, repo_path)
  print("Repository cloned.")

# Function to recursively read files in the repository
def read_files(repo_path):
  media_extensions = {'.mp4', '.gif', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.svg'}
  files = []
  for root, dirs, file_names in os.walk(repo_path):
    # Skip the .git directory
    if '.git' in root:
      continue
    for file_name in file_names:
      file_path = os.path.join(root, file_name)
      # Skip media files
      if any(file_name.lower().endswith(ext) for ext in media_extensions):
        continue
      with open(file_path, 'r', errors='ignore') as file:
        files.append((file_path, file.read()))
  return files

# Function to split text into smaller chunks
def split_text(text, max_length=2048):
  words = text.split()
  chunks = []
  chunk = []
  chunk_length = 0
  for word in words:
    if chunk_length + len(word) + 1 > max_length:
      chunks.append(" ".join(chunk))
      chunk = []
      chunk_length = 0
    chunk.append(word)
    chunk_length += len(word) + 1
  if chunk:
    chunks.append(" ".join(chunk))
  return chunks

# Function to get embeddings for text with retry mechanism
def get_embeddings(texts, retries=5, backoff_factor=1.0):
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }
  embeddings = []
  for text in texts:
    for attempt in range(retries):
      try:
        data = {
          "model": "text-embedding-ada-002",
          "input": text
        }
        response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
        response.raise_for_status()
        embedding_data = response.json()
        if 'data' in embedding_data:
          embeddings.extend([datum['embedding'] for datum in embedding_data['data']])
        else:
          print(f"Unexpected response format: {embedding_data}")
        break
      except requests.exceptions.HTTPError as e:
        if response.status_code >= 500:
          print(f"Server error ({response.status_code}) on attempt {attempt + 1}. Retrying...")
          time.sleep(backoff_factor * (2 ** attempt))
        else:
          print(f"Client error ({response.status_code}) on attempt {attempt + 1}. Not retrying.")
          print(f"Error: {response.text}")
          break
      except Exception as e:
        print(f"Error generating embedding for text chunk: {e}")
        break
  return np.array(embeddings)

# Function to get the content of top matching files
def get_file_contents(file_paths):
  contents = []
  for file_path in file_paths:
    try:
      with open(file_path, 'r', errors='ignore') as file:
        contents.append(file.read())
    except Exception as e:
      print(f"Error reading file {file_path}: {e}")
  return contents

# Function to generate a summary or response using chat/completions
def generate_response(query, file_contents, max_tokens=1500, max_context_tokens=16384):
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  def count_tokens(text):
    return len(text.split())

  def truncate_contents(file_contents, max_context_tokens):
    truncated_contents = []
    total_tokens = 0
    for content in file_contents:
      tokens = count_tokens(content)
      if total_tokens + tokens > max_context_tokens:
        # Truncate the content itself if adding the whole content exceeds the limit
        remaining_tokens = max_context_tokens - total_tokens
        truncated_contents.append(" ".join(content.split()[:remaining_tokens]))
        break
      truncated_contents.append(content)
      total_tokens += tokens
    return truncated_contents

  # Estimate initial message size and leave room for the query and assistant system messages
  initial_context_tokens = 1000
  file_contents = truncate_contents(file_contents, max_context_tokens - initial_context_tokens)

  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Based on the following files, answer the query: {query}\n\n" + "\n\n".join(file_contents)}
  ]

  data = {
    "model": "gpt-3.5-turbo",
    "messages": messages,
    "max_tokens": max_tokens
  }

  while True:
    try:
      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
      response.raise_for_status()
      response_data = response.json()
      return response_data['choices'][0]['message']['content'].strip()
    except requests.exceptions.HTTPError as e:
      if response.status_code == 400 and "context_length_exceeded" in response.text:
        # Reduce the file contents if the context length is exceeded
        file_contents = file_contents[:-1]  # Remove the last content
        messages[1]['content'] = f"Based on the following files, answer the query: {query}\n\n" + "\n\n".join(file_contents)
        data['messages'] = messages
      else:
        print(f"Error generating response: {response.text}")
        raise e

# Check if the index and file paths already exist
if os.path.exists('kernel_index.faiss') and os.path.exists('file_paths.txt'):
  # Load the existing index and file paths
  print("Loading existing index and file paths...")
  index = faiss.read_index('kernel_index.faiss')
  with open('file_paths.txt', 'r') as f:
    chunk_to_file_path = f.read().splitlines()
else:
  # Read files from the repository
  print("Reading files from the repository...")
  files = read_files(repo_path)
  file_chunks = []
  chunk_to_file_path = []  # Store mapping from chunk to file path
  for file_path, content in files:
    chunks = split_text(content)
    file_chunks.extend(chunks)
    chunk_to_file_path.extend([file_path] * len(chunks))  # Store the file path for each chunk

  # Get embeddings for the file chunks
  print("Generating embeddings...")
  embeddings = get_embeddings(file_chunks)

  # Check if embeddings were generated correctly
  if embeddings.size == 0:
    raise ValueError("No embeddings were generated. Please check the input data and API responses.")

  # Index the embeddings using FAISS
  print("Indexing embeddings...")
  dimension = embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(embeddings)

  # Save the index and chunk-to-file path mapping
  faiss.write_index(index, 'kernel_index.faiss')
  with open('file_paths.txt', 'w') as f:
    for file_path in chunk_to_file_path:
      f.write(file_path + '\n')

  print("Indexing completed.")

# Function to search the index
def search_index(query, top_k=5):
  query_embedding = get_embeddings([query])[0]
  distances, indices = index.search(np.array([query_embedding]), top_k)
  results = [(chunk_to_file_path[i], distances[0][j]) for j, i in enumerate(indices[0])]
  return results

# Enter a loop to answer questions
print("Enter your query about the Linux kernel (type 'exit' to quit):")
while True:
  query = input("Query: ")
  if query.lower() == 'exit':
    break
  results = search_index(query)
  top_file_paths = [result[0] for result in results]
  top_file_contents = get_file_contents(top_file_paths)
  response = generate_response(query, top_file_contents)
  print(f"Answer: {response}")
  for file_path, distance in results:
    print(f"File: {file_path}, Distance: {distance}")

