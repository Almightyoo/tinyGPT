from collections import Counter, defaultdict
import regex
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import heapq
import time
from termcolor import colored

class BPE:
    def __init__(
            self, 
            vocab: Dict[str, int], 
            id_to_token: Dict[int, str], 
            merges: List[Tuple[int,int]],
            regex_pattern: str,
            show_progress: bool = True
            ) -> None:
        self.vocab = vocab
        self.id_to_token = id_to_token
        self.merges = merges
        self.regex_pattern = regex_pattern
        self.show_progress = show_progress
        self.len_vocab = len(vocab)

    def encode(self, text: str) -> List[int]:
        split_text = self._preprocess_text(text)
        # print(split_text)

        tokenized_words = self._tokenize_words(split_text)
        # print(tokenized_words)
        for part_a, part_b in self.merges:
            new_token = self.id_to_token[part_a] + self.id_to_token[part_b]
            for i, word in enumerate(tokenized_words):
                tokenized_words[i] = self._merge_in_word(word, part_a, part_b, self.vocab[new_token])
        flattened_ids = [token_id for word in tokenized_words for token_id in word]
        return flattened_ids

    
    def _preprocess_text(self, text: str) -> List[str]:
        return [word for word in tqdm(regex.findall(self.regex_pattern, text), disable = not self.show_progress)]

    def _tokenize_words(self, split_text: List[str])-> List[List[int]]:
        tokenized_words = [[self.vocab.setdefault(char, self.len_vocab) for char in word] for word in split_text]
        return tokenized_words

    def _merge_in_word(self, word: List[int], part_a: int, part_b: int, new_token_id: int) -> List[int]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == part_a and word[i + 1] == part_b:
                new_word.append(new_token_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word
    
    def color_code_tokens(self, text: str) -> None:
        tokens = self.encode(text)
        token_texts = [self.id_to_token[t] for t in tokens]
        print(token_texts)

        highlight_colors = ["on_red", "on_green", "on_blue","on_yellow","on_light_grey", "on_magenta", "on_white", "on_cyan",]

        highlighted_text = ""
        for i,token in enumerate(token_texts):
            highlighted_text += colored(token, "black", highlight_colors[i%len(highlight_colors)])
        print("tokens highlighted:")
        print(highlighted_text)

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        text = "".join(tokens)        
        return text






class BPETrainer:
    def __init__(
            self,
            vocab_size: int = 50000,
            regex_pattern: str = '',
            show_progress: bool = True,
            min_frequency: int = 0,
            max_token_length: int = 20
            ) -> None:
        self.vocab_size = vocab_size
        self.show_progress = show_progress
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = []
        self.word_counts = defaultdict(int)
        self.initial_alphabet = set()
        self.regex_pattern=regex_pattern
        self.min_frequency= min_frequency
        self.max_token_length = max_token_length

    def feed(self, data: str) -> None:
        words= self._preprocess_text(data)
        for word in tqdm(words, disable = not self.show_progress):
            self.word_counts[word]+=1
            self.initial_alphabet.update(word)

    def _preprocess_text(self, data: str) -> List[str]:
        return [word for word in tqdm(regex.findall(self.regex_pattern, data), disable = not self.show_progress)]

    def _initialize_vocab(self) -> Dict[str, int]:
        #TODO: add special tokens later here
        vocab = {char: idx for idx, char in enumerate(self.initial_alphabet)}
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        return vocab
    
    def _tokenize_words(self, vocab: Dict[str, int])-> Tuple[List[List[int]], List[int]]:
        tokenized_words = [[vocab.setdefault(char, len(vocab)) for char in word] for word in self.word_counts]
        counts = list(self.word_counts.values())
        return tokenized_words, counts

    def _count_pairs(self, tokenized_words: List[List[int]], counts: List[int]) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Set[int]]]:
        pair_counts = Counter()
        where_to_update = defaultdict(set)
        
        for i, (word, count) in enumerate(zip(tokenized_words, counts)):
            for pair in zip(word[:-1], word[1:]):
                pair_counts[pair] += count
                where_to_update[pair].add(i)
        
        return pair_counts, where_to_update
    
    def _build_priority_queue(self, pair_counts: Dict[Tuple[int, int], int], where_to_update: Dict[Tuple[int, int], Set[int]]) -> List[Tuple[int, Tuple[int, int]]]:
        queue = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(queue)
        return queue
    
    def _merge_pair(self, part_a: int, part_b: int) -> str:
        """Merge two tokens into a new token."""
        if part_a not in self.reverse_vocab or part_b not in self.reverse_vocab:
            raise KeyError(f"Token ID {part_a} or {part_b} not found in reverse_vocab.")
        return f"{self.reverse_vocab[part_a]}{self.reverse_vocab[part_b]}"

    def _merge_in_word(self, word: List[int], part_a: int, part_b: int, new_token_id: int) -> List[int]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == part_a and word[i + 1] == part_b:
                new_word.append(new_token_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    

    def train(self) -> Dict[str, int]:
        vocab = self._initialize_vocab()
        # print(vocab)
        # print(self.reverse_vocab)
        tokenized_words, counts = self._tokenize_words(vocab)

        pair_counts, where_to_update = self._count_pairs(tokenized_words, counts)
        queue = self._build_priority_queue(pair_counts,where_to_update)

        while len(vocab) < self.vocab_size and queue:
            count, pair = heapq.heappop(queue)
            if -count != pair_counts[pair]: continue
            if -count<self.min_frequency: break
            part_a, part_b = pair
            new_token = self._merge_pair(part_a, part_b)
            

            if len(new_token) > self.max_token_length: continue

            if new_token not in vocab:
                vocab[new_token] = len(vocab)
                self.merges.append((part_a, part_b))
                self.reverse_vocab[vocab[new_token]] = new_token

            for i in where_to_update[pair]:
                word =tokenized_words[i]
                new_word = self._merge_in_word(word, part_a, part_b, vocab[new_token])
                tokenized_words[i]=new_word

                for j in range(len(new_word) - 1):
                    new_pair = (new_word[j], new_word[j + 1])
                    pair_counts[new_pair] += counts[i]
                    where_to_update[new_pair].add(i)
                    heapq.heappush(queue, (-pair_counts[new_pair], new_pair))

        return vocab
    



# ======================================= TESTING THE BPE TOKENIZER ========================================== #

with open('data/input.txt', 'r', encoding='utf-8') as f:
    dataset= f.read()

data=dataset[:100000000]
regex_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
text = "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose"


time1=time.time()
trainer = BPETrainer(regex_pattern=regex_pattern, show_progress=True)
trainer.feed(data)
vocab = trainer.train()
print(len(vocab))
print(time.time()-time1)
time2 = time.time()
bpe = BPE(vocab, trainer.reverse_vocab, trainer.merges, trainer.regex_pattern, show_progress=True)

encoded_ids = bpe.encode(text)
print("Encoded IDs:", encoded_ids)
print("Encoded length :", len(encoded_ids))
print(time.time()-time2)
decoded_text = bpe.decode(encoded_ids)
print("Decoded Text:", decoded_text)
print(bpe.color_code_tokens(text))
print(trainer.initial_alphabet)

# for i,(k, v) in enumerate(vocab.items()):
#     # if i>=1000: break
#     print(f'{k} | {v}')
# print(trainer.vocab_size)
# print(trainer.merges[:10])





# ======================================= TESTING THE BPE TOKENIZER ========================================== #