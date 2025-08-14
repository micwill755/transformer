import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 0
    
    def tokenize(self, text):
        """Convert text to lowercase and split into words, removing punctuation"""
        # Convert to lowercase and split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def build_vocab(self, sentences):
        """Build vocabulary from a list of sentences"""
        all_tokens = []
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        for token in special_tokens:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        
        # Add regular tokens to vocabulary
        for token, count in token_counts.items():
            if token not in self.token_to_id:
                self.token_to_id[token] = self.next_id
                self.id_to_token[self.next_id] = token
                self.next_id += 1
        
        self.vocab = token_counts
        return self.vocab
    
    def encode(self, text):
        """Convert text to sequence of token IDs"""
        tokens = self.tokenize(text)
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id['<UNK>'])  # Unknown token
        return token_ids
    
    def decode(self, token_ids):
        """Convert sequence of token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<UNK>')
        return ' '.join(tokens)

# Example usage
def main():
    # Sample sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process natural language.",
        "Tokenization breaks text into smaller meaningful units called tokens."
    ]
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Build vocabulary from sample sentences
    vocab = tokenizer.build_vocab(sentences)
    
    print("=== VOCABULARY ===")
    print(f"Vocabulary size: {len(vocab)}")
    print("Token frequencies:")
    for token, count in vocab.most_common():
        print(f"  '{token}': {count}")
    
    print(f"\nToken to ID mapping (first 10):")
    for i, (token, token_id) in enumerate(tokenizer.token_to_id.items()):
        if i < 10:
            print(f"  '{token}' -> {token_id}")
    
    print("\n=== TOKENIZATION EXAMPLES ===")
    
    # Process each sentence
    for i, sentence in enumerate(sentences, 1):
        print(f"\nSentence {i}: \"{sentence}\"")
        
        # Tokenize into words
        tokens = tokenizer.tokenize(sentence)
        print(f"Tokens: {tokens}")
        
        # Convert to numerical IDs
        token_ids = tokenizer.encode(sentence)
        print(f"Token IDs: {token_ids}")
        
        # Decode back to text
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded: \"{decoded}\"")
        
        print("-" * 50)
    
    # Test with a new sentence (containing unknown words)
    print("\n=== TESTING WITH NEW SENTENCE ===")
    new_sentence = "Python programming requires understanding algorithms."
    print(f"New sentence: \"{new_sentence}\"")
    
    tokens = tokenizer.tokenize(new_sentence)
    print(f"Tokens: {tokens}")
    
    token_ids = tokenizer.encode(new_sentence)
    print(f"Token IDs: {token_ids}")
    
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: \"{decoded}\"")

if __name__ == "__main__":
    main()
