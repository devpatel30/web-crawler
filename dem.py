import os
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from afinn import Afinn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def can_crawl(url, domain):
    try:
        robots_url = urljoin(domain, '/robots.txt')
        response = requests.get(robots_url)

        if response.status_code == 200:
            robots_content = response.text
            # Check if the URL is allowed based on the rules in robots.txt
            for line in robots_content.split('\n'):
                if line.lower().startswith('disallow:') and '/' in line.lower():
                    disallowed_path = line.split(' ')[1].strip()
                    if disallowed_path and url.endswith(disallowed_path):
                        return False
            return True
        else:
            return True  # If there is no robots.txt, assume it's okay to crawl
    except Exception as e:
        print(f"Error checking robots.txt for {domain}: {e}")
        return False


def crawl_domain(url, max_files, output_directory):
    visited_urls = set()
    crawled_data = []  # List to store the text content of each page

    def dfs_crawl(current_url, depth):
        nonlocal max_files  # Allow modification of the outer variable

        if depth <= 0 or len(crawled_data) >= max_files or not can_crawl(current_url, url):
            return
        try:
            # Check if the URL has been visited before
            if current_url in visited_urls:
                return
            visited_urls.add(current_url)

            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content from the page
            text_content = ' '.join([p.get_text() for p in soup.find_all('p')])

            crawled_data.append({'url': current_url, 'text': text_content})

            # Save data to a file
            save_to_file(current_url, text_content, output_directory)

            for link in soup.find_all('a', href=True):
                next_url = urljoin(current_url, link['href'])
                # Check if the next URL is under the specified domain
                if urlparse(next_url).netloc == urlparse(url).netloc:
                    if can_crawl(next_url, url):
                        dfs_crawl(next_url, depth - 1)

                        # Check if the maximum number of files has been reached after each crawl
                        if len(crawled_data) >= max_files:
                            return

        except Exception as e:
            print(f"Error crawling {current_url}: {e}")

    dfs_crawl(url, depth=3)  # You can adjust the depth as needed
    return crawled_data


def save_to_file(url, text_content, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Replace invalid characters in the URL to create a valid filename
    filename = url.replace('/', '_').replace(':', '').replace('.', '_')

    # Save data to a file
    with open(os.path.join(output_directory, f"{filename}.txt"), 'w', encoding='utf-8') as file:
        file.write(f"{text_content}")


def preprocess_text(text):
    # Remove numeric values
    text = re.sub(r'\b\d+\b', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


def get_top_terms_per_cluster(corpus, labels, num_clusters, top_n=20):
    terms_per_cluster = {}

    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_corpus = [corpus[i] for i in cluster_indices]

        # Preprocess the cluster corpus
        preprocessed_corpus = [preprocess_text(doc) for doc in cluster_corpus]

        # Vectorize the preprocessed corpus
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(preprocessed_corpus)

        # Get feature names (terms)
        feature_names = vectorizer.get_feature_names_out()

        # Sum the TF-IDF scores for each term across documents in the cluster
        term_scores = X.sum(axis=0).A1

        # Get indices of top terms
        top_term_indices = term_scores.argsort()[-top_n:][::-1]

        # Map indices to terms
        top_terms = [feature_names[idx] for idx in top_term_indices]

        # Store top terms for the cluster
        terms_per_cluster[cluster_id] = top_terms

    return terms_per_cluster


def save_top_terms_to_file(terms_per_cluster, filename):
    with open(filename, 'w') as file:
        for cluster_id, top_terms in terms_per_cluster.items():
            file.write(f"Cluster {cluster_id + 1}: {', '.join(top_terms)}\n")


def cluster_documents(corpus, num_clusters, save_plot=False, plot_filename=None):
    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    # Scatter plot
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5, s=50)
    plt.title(f'Clustering for k={num_clusters}')

    # Save plot
    plt.savefig(plot_filename)


    # Get top terms for each cluster
    terms_per_cluster = get_top_terms_per_cluster(corpus, kmeans.labels_, num_clusters)
    top_n = 20
    # Print top terms for each cluster
    print(f"Top {top_n} Terms for Each Cluster (k={num_clusters}):")
    for cluster_id, top_terms in terms_per_cluster.items():
        print(f"Cluster {cluster_id + 1}: {', '.join(top_terms)}")

    save_top_terms_to_file(terms_per_cluster, f"top_terms_{num_clusters}nn.txt")
    return kmeans.labels_


def calculate_cluster_sentiment(clusters, afinn, documents):
    cluster_sentiments = []

    # Iterate over unique cluster labels
    for cluster_label in np.unique(clusters):
        # Find documents belonging to the current cluster
        cluster_docs_indices = np.where(clusters == cluster_label)[0]

        total_score = 0
        total_length = 0

        for doc_index in cluster_docs_indices:
            score = afinn.score(documents[doc_index])
            doc_length = len(documents[doc_index].split())  # Assuming words as tokens
            total_score += score * doc_length
            total_length += doc_length

        cluster_sentiment = total_score / total_length if total_length > 0 else 0
        cluster_sentiments.append(cluster_sentiment)

    return cluster_sentiments


if __name__ == "__main__":
    domain_url = 'https://www.concordia.ca'
    max_files = 10
    output_directory = 'output_files'
    crawled_data = crawl_domain(domain_url, max_files, output_directory)

    # Extract text content from crawled pages
    corpus = [data['text'] for data in crawled_data]

    # Perform clustering for k=3
    clusters_k3 = cluster_documents(corpus, num_clusters=3, save_plot=True, plot_filename='kmeans_plot_k3.png')

    # Assess clusters for k=3
    print("Clusters for k=3:")
    for i, label in enumerate(clusters_k3):
        print(f"URL: {crawled_data[i]['url']}, Cluster Label: {label}")

    # Perform clustering for k=6
    clusters_k6 = cluster_documents(corpus, num_clusters=6, save_plot=True, plot_filename='kmeans_plot_k6.png')

    # Assess clusters for k=6
    print("\nClusters for k=6:")
    for i, label in enumerate(clusters_k6):
        print(f"URL: {crawled_data[i]['url']}, Cluster Label: {label}")

    # Initialize Afinn sentiment analyzer
    afinn = Afinn()
    # Calculate sentiment values for clusters
    cluster_sentiments_k3 = calculate_cluster_sentiment(clusters_k3, afinn, corpus)
    cluster_sentiments_k6 = calculate_cluster_sentiment(clusters_k6, afinn, corpus)

    # Print cluster sentiment values
    print("\nCluster Sentiments for k=3:")
    for i, sentiment in enumerate(cluster_sentiments_k3):
        print(f"Cluster {i + 1}: {sentiment}")

    print("\nCluster Sentiments for k=6:")
    for i, sentiment in enumerate(cluster_sentiments_k6):
        print(f"Cluster {i + 1}: {sentiment}")
