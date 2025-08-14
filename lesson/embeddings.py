import numpy as np
import matplotlib.pyplot as plt
from simple_tokenizer import SimpleTokenizer


class SimpleEmbedding:
    def __init__(self, vocab_size, embedding_dim=2):
        """
        Simple embedding layer that maps token IDs to dense vectors.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialize embeddings with small random values
        self.embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    
    def forward(self, token_ids):
        """
        Get embeddings for given token IDs.
        
        Args:
            token_ids (list): List of token IDs
            
        Returns:
            np.ndarray: Embeddings for the tokens
        """
        return self.embeddings[token_ids]
    
    def get_sentence_embedding(self, token_ids, method='mean'):
        """
        Get sentence-level embedding by aggregating token embeddings.
        
        Args:
            token_ids (list): List of token IDs for the sentence
            method (str): Aggregation method ('mean', 'sum', 'max')
            
        Returns:
            np.ndarray: Sentence embedding
        """
        token_embeddings = self.forward(token_ids)
        
        if method == 'mean':
            return np.mean(token_embeddings, axis=0)
        elif method == 'sum':
            return np.sum(token_embeddings, axis=0)
        elif method == 'max':
            return np.max(token_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


def visualize_embeddings(sentences, embeddings, tokenizer):
    """
    Visualize 2D embeddings using matplotlib.
    
    Args:
        sentences (list): List of original sentences
        embeddings (np.ndarray): 2D embeddings
        tokenizer (SimpleTokenizer): Tokenizer instance for vocabulary info
    """
    plt.figure(figsize=(10, 8))
    
    # Plot sentence embeddings
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c='red', s=100, alpha=0.7, label='Sentences')
    
    # Add labels for each sentence
    for i, sentence in enumerate(sentences):
        plt.annotate(f"'{sentence}'", 
                    (embeddings[i, 0], embeddings[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('2D Text Embeddings Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Example sentences
    sentences = [
        "hello world",
        "world hello",
        "good morning",
        "morning good",
        "hello good morning",
        "world is beautiful",
        "beautiful world",
        "good day today",
        "today is good",
        "hello beautiful day"
    ]
    
    print("=== Text Embedding Example ===\n")
    
    # Tokenize all sentences to build vocabulary
    tokenized_sentences = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokenized_sentences.append(tokens)
        print(f"'{sentence}' -> {tokens}")
    
    print(f"\nVocabulary size: {tokenizer.vocab_size()}")
    print(f"Vocabulary: {tokenizer.word_to_id}")
    
    # Initialize embedding layer
    embedding_layer = SimpleEmbedding(tokenizer.vocab_size(), embedding_dim=2)
    
    print(f"\nEmbedding matrix shape: {embedding_layer.embeddings.shape}")
    
    # Generate sentence embeddings
    sentence_embeddings = []
    print("\n=== Sentence Embeddings ===")
    
    for i, (sentence, tokens) in enumerate(zip(sentences, tokenized_sentences)):
        # Get sentence embedding (mean of token embeddings)
        sent_emb = embedding_layer.get_sentence_embedding(tokens, method='mean')
        sentence_embeddings.append(sent_emb)
        print(f"'{sentence}' -> [{sent_emb[0]:.3f}, {sent_emb[1]:.3f}]")
    
    sentence_embeddings = np.array(sentence_embeddings)
    
    # Show individual token embeddings for the first sentence
    print(f"\n=== Token Embeddings for '{sentences[0]}' ===")
    first_tokens = tokenized_sentences[0]
    token_embs = embedding_layer.forward(first_tokens)
    
    for token_id, embedding in zip(first_tokens, token_embs):
        word = tokenizer.id_to_word[token_id]
        print(f"Token '{word}' (ID: {token_id}) -> [{embedding[0]:.3f}, {embedding[1]:.3f}]")
    
    # Calculate similarity between sentences
    print("\n=== Sentence Similarities (Cosine) ===")
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Show similarities for first few sentences
    for i in range(min(3, len(sentences))):
        for j in range(i+1, min(3, len(sentences))):
            sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])
            print(f"'{sentences[i]}' <-> '{sentences[j]}': {sim:.3f}")
    
    # Visualize embeddings
    print("\n=== Visualizing Embeddings ===")
    visualize_embeddings(sentences, sentence_embeddings, tokenizer)
    
    # Demonstrate different aggregation methods
    print("\n=== Different Aggregation Methods ===")
    test_sentence = "hello beautiful world"
    test_tokens = tokenizer.tokenize(test_sentence)
    
    for method in ['mean', 'sum', 'max']:
        emb = embedding_layer.get_sentence_embedding(test_tokens, method=method)
        print(f"'{test_sentence}' ({method}): [{emb[0]:.3f}, {emb[1]:.3f}]")


if __name__ == "__main__":
    main()
