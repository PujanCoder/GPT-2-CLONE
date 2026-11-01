
class BytePairEncoder:
    def __init__(self, merges):
        """
        merges: ordered list of pairs, such as:
            [("t", "h"), ("th", "e")]
        """
        self.bpe_ranks = {
            tuple(merge): rank
            for rank, merge in enumerate(merges)
        }

        self.cache = {}

    @staticmethod
    def get_pairs(word):
        """Return all adjacent symbol pairs."""
        pairs = set()

        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))

        return pairs

    def encode(self, token):
        """Apply BPE merges to one pre-tokenized string."""

        if token in self.cache:
            return self.cache[token]

        word = tuple(token)

        if len(word) <= 1:
            return [token]

        while True:
            pairs = self.get_pairs(word)

            if not pairs:
                break

            # Select the available pair with the lowest merge rank.
            best_pair = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(
                    pair, float("inf")
                ),
            )

            # Stop if no pair is present in the merge vocabulary.
            if best_pair not in self.bpe_ranks:
                break

            first, second = best_pair
            merged_word = []
            i = 0

            # Merge every occurrence of the selected adjacent pair.
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == first
                    and word[i + 1] == second
                ):
                    merged_word.append(first + second)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1

            word = tuple(merged_word)

            if len(word) == 1:
                break

        result = list(word)
        self.cache[token] = result

        return result
