import os
import re
import logging
import argparse
import torch
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_distances

# Set up logging to a file
logging.basicConfig(filename='semantic_linking.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Argument parser for command line options
parser = argparse.ArgumentParser(description="Semantic Linking of Obsidian Notes")
parser.add_argument('--clusters', type=int, default=30, help='Number of clusters for Agglomerative Clustering')
parser.add_argument('--similarity', type=float, default=0.7, help='Similarity threshold for linking notes')
parser.add_argument('--title-weight', type=float, default=0.5, help='Weight for title embeddings in combination')
parser.add_argument('--content-weight', type=float, default=0.5, help='Weight for content embeddings in combination')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing embeddings')
parser.add_argument('--force-embeddings', action='store_true', help='Force regenerating embeddings even if they exist')

args = parser.parse_args()

# Path to your Obsidian vault
obsidian_vault_path = "C:/Users/epi_c/OneDrive/Documents/ObsidianVault/GPTs3"  # Adjust to your actual path

# Load pre-trained model for generating embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device=device)


# Function to remove existing links from markdown files
def remove_existing_links(vault_path):
    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r+', encoding='utf-8') as f:
                    content = f.read()
                    cleaned_content = re.sub(r'\[\[.*?\]\]', '', content)  # Remove Obsidian-style links
                    f.seek(0)
                    f.write(cleaned_content)
                    f.truncate()
                logging.info(f"Links removed from {file_path}")

# Call the function to remove all existing links before each run
remove_existing_links(obsidian_vault_path)

# Function to collect all markdown notes from the vault
def load_notes_from_vault(vault_path):
    note_files = []
    notes = []
    
    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    note_content = f.read()
                    note_files.append(file_path)
                    notes.append(note_content)

    logging.info(f"Loaded {len(notes)} notes from the vault.")
    return note_files, notes

# Function to save embeddings to avoid recomputing them
def save_embeddings(embeddings, filename="embeddings.pt"):
    torch.save(embeddings, filename)

# Function to load precomputed embeddings
def load_embeddings(filename="embeddings.pt"):
    return torch.load(filename)

# Load all notes
note_files, notes = load_notes_from_vault(obsidian_vault_path)

# Extract titles from note file names
titles = [os.path.basename(file).replace('.md', '') for file in note_files]

# Check if embeddings are already saved and if force-embeddings is not set
embeddings_filename = "note_embeddings_combined.pt"
if not args.force_embeddings and os.path.exists(embeddings_filename):
    logging.info("Loading precomputed combined embeddings...")
    note_embeddings_combined = load_embeddings(embeddings_filename)
else:
    logging.info("Generating title and content embeddings...")

    # Generate content embeddings with batching
    content_embeddings = model.encode(notes, convert_to_tensor=True, device=device, batch_size=args.batch_size)

    # Generate title embeddings with batching
    title_embeddings = model.encode(titles, convert_to_tensor=True, device=device, batch_size=args.batch_size)

    # Combine title and content embeddings using the specified weights
    note_embeddings_combined = (args.title_weight * title_embeddings) + (args.content_weight * content_embeddings)
    
    # Save the combined embeddings
    save_embeddings(note_embeddings_combined, embeddings_filename)

# Convert embeddings to numpy array
note_embeddings_np = note_embeddings_combined.cpu().numpy()

# Calculate the pairwise cosine distances between the note embeddings
distance_matrix = cosine_distances(note_embeddings_np)

# Perform hierarchical clustering (Agglomerative Clustering) with command line input for clusters
hierarchical_cluster = AgglomerativeClustering(n_clusters=args.clusters, metric='precomputed', linkage='average')
cluster_labels = hierarchical_cluster.fit_predict(distance_matrix)

# Log cluster assignments
for i, note_file in enumerate(note_files):
    logging.info(f"Note: {note_file} - Assigned to Cluster {cluster_labels[i]}")

# Function to insert links into markdown files
def insert_links_to_file(file_path, linked_notes):
    with open(file_path, 'r+', encoding='utf-8') as file:
        content = file.read()
        # Use regex to find existing links
        existing_links = re.findall(r'\[\[(.*?)\]\]', content)
        
        # Create new links, avoiding duplicates
        new_links = [f'[[{os.path.basename(note_path).replace(".md", "")}]]' for note_path in linked_notes if os.path.basename(note_path).replace(".md", "") not in existing_links]
        
        if new_links:
            link_lines = '\n'.join(new_links)
            content = f'{link_lines}\n\n{content}'
            file.seek(0)
            file.write(content)
            file.truncate()

    logging.info(f"Links written to {file_path}")

# Handle Orphans - Identify and link isolated notes
def load_orphans(filename="no_link_notes.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        orphans = [line.strip() for line in f.readlines()]
    return orphans

orphans = load_orphans()

# Linking based on the input similarity threshold from command line
for orphan_path in orphans:
    try:
        orphan_idx = note_files.index(orphan_path)
    except ValueError:
        logging.warning(f"Orphan note {orphan_path} not found in note_files.")
        continue

    similar_notes = sorted([(i, cosine_distances(note_embeddings_np[orphan_idx].reshape(1, -1), note_embeddings_np[i].reshape(1, -1)).item()) for i in range(len(notes)) if i != orphan_idx],
                           key=lambda x: x[1])[:3]  # Top 3 similar notes

    linked_note_paths = [note_files[note_idx] for note_idx, _ in similar_notes if cosine_distances(note_embeddings_np[orphan_idx].reshape(1, -1), note_embeddings_np[note_idx].reshape(1, -1)).item() <= args.similarity]

    insert_links_to_file(orphan_path, linked_note_paths)
    
    logging.info(f"Links written to {orphan_path}")
