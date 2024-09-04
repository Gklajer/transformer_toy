from collections import defaultdict


class BasicTokenizer:
    def __init__(self, vocab_size: int, train_text: str, verbose=False):
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges = [None] * (vocab_size - 256)

        self._counts = defaultdict(int)
        
        self._train(train_text, verbose)

                
    def encode(self, text: str) -> list[int]:
        """transform text into a list of tokens"""
        tokens = list(text.encode("utf-8"))
        for i, pair in enumerate(self.merges):
            self._merge(tokens, pair, 256 + i)

        return tokens

    def decode(self, tokens: list[int]):
        """return the corresponding text from a list of tokens"""
        btokens = b"".join(self.vocab[token] for token in tokens)
        return btokens.decode("utf8", errors="replace")
    
    def _train(self, text: str, verbose):
        """train the tokenizer on the given text"""
        self._in_training = True
                
        tokens = list(text.encode("utf-8"))

        self._count_pairs(tokens)

        if verbose:
            print("Start merging")

        for i in range(len(self.merges)):
            (i1, i2), top_count = max(self._counts.items(), key=lambda itm: itm[1])

            if top_count == 1:
                break

            self.merges[i] = (i1, i2)
            self.vocab.append(self.vocab[i1] + self.vocab[i2])

            self._merge(tokens, (i1, i2), (new_token := 256 + i))

            if verbose:
                print(f"\t-merge {self.decode([-1])!r}")
                
        self._in_training = False

    def _count_pairs(self, tokens: list[int]):
        """count the number of pairs of tokens"""
        for pair in zip(tokens, tokens[1:]):
            self._counts[pair] += 1

    def _update_counts(self, old_pair: tuple[int, int], new_pair: tuple[int, int]):
        """update the counts of pairs of tokens"""
        self._counts[old_pair] -= 1
        self._counts[new_pair] += 1

    def _merge(self, seq: list[int], pair: tuple[int, int], new: int):
        """replace all occurences of a given pair of tokens in the sequence by a single new token"""
        if len(seq) <= 1:
            return

        if self._in_training:
            self._counts[pair] = 0

        read_ptr = -1
        write_ptr = -1

        while (read_ptr := read_ptr + 1) < len(seq):
            write_ptr += 1
            seq[write_ptr] = seq[read_ptr]

            if read_ptr == len(seq) - 1:
                break

            if (seq[read_ptr], seq[read_ptr + 1]) != pair:
                continue

            if self._in_training and read_ptr >= 1:
                self._update_counts(
                    (seq[read_ptr - 1], seq[read_ptr]), (seq[read_ptr - 1], new)
                )

            if self._in_training and read_ptr + 2 < len(seq):
                self._update_counts(
                    (seq[read_ptr + 1], seq[read_ptr + 2]), (new, seq[read_ptr + 2])
                )

            read_ptr += 1
            seq[write_ptr] = new

        while len(seq) - 1 > write_ptr:
            seq.pop()
