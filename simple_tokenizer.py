class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        
    def tokenize(self, text):
        words = text.lower().split()
        current_id = len(self.word_to_id)
        
        tokens = []
        for word in words:
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

if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = SimpleTokenizer()
    
    text = "hello world hello"
    tokens = tokenizer.tokenize(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {' '.join(decoded)}")
    print(f"Vocabulary: {tokenizer.word_to_id}")