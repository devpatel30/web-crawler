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


# can_crawl checks the robots.txt to check if crawling is allowed/disallowed to the links in the domain
def can_crawl(url, domain):
    try:
        robots_url = urljoin(domain, '/robots.txt')
        response = requests.get(robots_url)
        '''check url rules in the robots.txt file'''
        if response.status_code == 200:
            robots_content = response.text
            for line in robots_content.split('\n'):
                if line.lower().startswith('disallow:') and '/' in line.lower():
                    disallowed_path = line.split(' ')[1].strip()
                    if disallowed_path and url.endswith(disallowed_path):
                        return False
            return True
        else:
            '''if no robots.txt crawl the webpage'''
            return True
    except Exception as e:
        print(f"Error checking robots.txt for {domain}: {e}")
        return False


# crawl_domain crawls the domain and saves the data from the webpages in a specified directory
def crawl_domain(url, maximum_files, pages_directory):
    # set to crawl unique links
    visited_urls = set()
    # stores data from webpage
    crawled_page_data = []

    def dfs_crawl(current_url, depth):
        """allows modification for outer variables"""
        nonlocal maximum_files

        if depth <= 0 or len(crawled_page_data) >= maximum_files or not can_crawl(current_url, url):
            return
        try:
            # check if url is visited before
            if current_url in visited_urls:
                return
            visited_urls.add(current_url)
            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # extract text form the webpage
            text_content = ' '.join([p.get_text() for p in soup.find_all('p')])
            crawled_page_data.append({'url': current_url, 'text': text_content})

            # save data to a file
            save_to_file(current_url, text_content, pages_directory)

            for link in soup.find_all('a', href=True):
                next_url = urljoin(current_url, link['href'])
                # check for url under the domain
                if urlparse(next_url).netloc == urlparse(url).netloc:
                    if can_crawl(next_url, url):
                        dfs_crawl(next_url, depth - 1)

                        # check for max_files
                        if len(crawled_page_data) >= maximum_files:
                            return

        except Exception as e:
            print(f"Error crawling {current_url}: {e}")

    # specify the depth for crawling
    dfs_crawl(url, depth=3)
    return crawled_page_data


# saves webpage data to specified directory
def save_to_file(url, text_content, pages_directory):
    os.makedirs(pages_directory, exist_ok=True)
    #  naming the file with appropriate characters
    filename = url.replace('/', '_').replace(':', '').replace('.', '_')

    # saving data to the file
    with open(os.path.join(pages_directory, f"{filename}.txt"), 'w', encoding='utf-8') as file:
        file.write(f"{text_content}")


# preprocessing text for importance in the cluster
def preprocess_text(text):
    # remove numerbers
    text = re.sub(r'\b\d+\b', '', text)
    # remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


# function to get top terms from cluster
def get_top_terms_per_cluster(corpus, labels, num_clusters, top_n=20):
    terms_per_cluster = {}

    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_corpus = [corpus[i] for i in cluster_indices]

        # preprocessing the cluster corpus
        preprocessed_corpus = [preprocess_text(doc) for doc in cluster_corpus]

        # vectorize preprocessed corpus
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(preprocessed_corpus)

        # get feature names (terms)
        feature_names = vectorizer.get_feature_names_out()

        # add tf-idf scores for each terms
        term_scores = X.sum(axis=0).A1

        # get indices for top terms
        top_term_indices = term_scores.argsort()[-top_n:][::-1]

        # map indices to terms
        top_terms = [feature_names[idx] for idx in top_term_indices]

        # save top terms for the cluster
        terms_per_cluster[cluster_id] = top_terms

    return terms_per_cluster


# save top terms to the file
def save_top_terms_to_file(terms_per_cluster, filename):
    with open(filename, 'w') as file:
        for cluster_id, top_terms in terms_per_cluster.items():
            file.write(f"Cluster {cluster_id + 1}: {', '.join(top_terms)}\n")


def cluster_documents(corpus, num_clusters, plot_filename=None):
    # vectorize the text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    # make scatter plot
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5, s=50)
    plt.title(f'Clustering for k={num_clusters}')

    # save the generated plot
    plt.savefig(plot_filename)

    # get top terms for each cluster
    terms_per_cluster = get_top_terms_per_cluster(corpus, kmeans.labels_, num_clusters)
    top_n = 20

    # Print top terms for each cluster
    print(f"Top {top_n} Terms for Each Cluster (k={num_clusters}):")
    for cluster_id, top_terms in terms_per_cluster.items():
        print(f"Cluster {cluster_id + 1}: {', '.join(top_terms)}")

    # save top n terms to a file
    save_top_terms_to_file(terms_per_cluster, f"top_terms_{num_clusters}_clusters.txt")
    return kmeans.labels_


# calculates the sentiment for clusters
def calculate_cluster_sentiment(clusters, afinn, documents):
    cluster_sentiments = []

    # iterate over unique cluster labels
    for cluster_label in np.unique(clusters):
        # find documents which for the current cluster
        cluster_docs_indices = np.where(clusters == cluster_label)[0]
        total_score = 0
        total_length = 0
        # get afinn score for each document
        for doc_index in cluster_docs_indices:
            score = afinn.score(documents[doc_index])
            doc_length = len(documents[doc_index].split())
            total_score += score * doc_length
            total_length += doc_length

        cluster_sentiment = total_score / total_length if total_length > 0 else 0
        cluster_sentiments.append(cluster_sentiment)

    return cluster_sentiments


if __name__ == "__main__":
    # change to any url
    domain_url = 'https://www.concordia.ca'
    max_files = 10
    output_directory = 'output_files'
    crawled_data = crawl_domain(domain_url, max_files, output_directory)

    # extract text content from crawled pages
    corpus = [data['text'] for data in crawled_data]

    # do k-means clustering for k=3
    clusters_k3 = cluster_documents(corpus, num_clusters=3, plot_filename='kmeans_plot_k3.png')
    print("Clusters for k=3:")
    for i, label in enumerate(clusters_k3):
        print(f"URL: {crawled_data[i]['url']}, Cluster Label: {label}")

    # do k-means clustering for k=6
    clusters_k6 = cluster_documents(corpus, num_clusters=6, plot_filename='kmeans_plot_k6.png')
    print("\nClusters for k=6:")
    for i, label in enumerate(clusters_k6):
        print(f"URL: {crawled_data[i]['url']}, Cluster Label: {label}")

    # initialize afinn sentiment analyzer
    afinn = Afinn()
    # calculate sentiment values for 3nn and 6nn clusters
    cluster_sentiments_k3 = calculate_cluster_sentiment(clusters_k3, afinn, corpus)
    cluster_sentiments_k6 = calculate_cluster_sentiment(clusters_k6, afinn, corpus)

    # print cluster sentiment values
    print("\nCluster Sentiments for k=3:")
    for i, sentiment in enumerate(cluster_sentiments_k3):
        print(f"Cluster {i + 1}: {sentiment}")
    print("\nCluster Sentiments for k=6:")
    for i, sentiment in enumerate(cluster_sentiments_k6):
        print(f"Cluster {i + 1}: {sentiment}")
