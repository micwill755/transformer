import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

'''
The SimpleTokenizer takes a sentence, splits it into individual words, 
and converts each unique word into a numeric ID (starting with special 
tokens like PAD and UNK), creating a dictionary that maps words to IDs and vice versa, 
allowing us to transform text into numbers that our neural network can process.
'''
class SimpleTokenizer:
    def __init__(self):
        # We use two dictionaries for bidirectional mapping
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Add special tokens
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<UNK>"] = 1
        self.id_to_word[0] = "<PAD>"
        self.id_to_word[1] = "<UNK>"
        
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1

    def tokenize(self, text):
        words = text.lower().split()
        current_id = len(self.word_to_id)
        
        tokens = []
        for word in words:
            # If word not in dictionary, add it
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
            
            tokens.append(self.word_to_id[word])
            
        return tokens
    
    def decode(self, token_ids):
        return [self.id_to_word.get(id, "<UNK>") for id in token_ids]

    def vocab_size(self):
        return len(self.word_to_id)
    
# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Example sentences
    sentences = [
        "hello world",
        "attention is awesome",
        "deep learning is fun"
    ]
    
    # Tokenize sentences
    tokenized = [tokenizer.tokenize(sentence) for sentence in sentences]

    print("\nTokenized sequences:")
    for sent, tokens in zip(sentences, tokenized):
        print(f"Original: {sent}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {' '.join(tokenizer.decode(tokens))}\n")