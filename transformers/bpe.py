from typing import List
from queue import PriorityQueue
from collections import defaultdict
from utils import *

def bpe(dataset: List[str], target_vocab_size):
    heads = []
    tokens_dict = defaultdict(list)
    basic_chars = set()

    for line in dataset:
        tokens = token_split(line.strip())
        for token in tokens:
            head = None
            last = None
            for byte in token.encode("utf-8"):
                byte = ByteSeq(byte)
                basic_chars.add(byte)
                if head == None:
                    head = Node(byte)
                    last = head
                    continue
                new = Node(byte)
                tokens_dict[last.val + byte].append(new)
                new.prev = last
                last.next = new
                last = new
            heads.append(head)

    counter = 0 # for distinguishing pq values with same count
    pq: PriorityQueue[str] = PriorityQueue()
    for key, value in tokens_dict.items():
        # neg len value cuz we want MOST common bi-bytes
        pq.put((-len(value), counter, key))
        counter += 1

    merged_tokens = []
    while len(merged_tokens) + len(basic_chars) < target_vocab_size:
        _, _, key = pq.get() # get and remove
        nodes = tokens_dict[key]
        del tokens_dict[key]

        # verify validity since we're intentionally lazy
        valid_nodes = [n for n in nodes if n.prev and n.prev.val + n.val == key]
        if len(valid_nodes) == 0: # skip nodes with no more valid
            continue

        merged_tokens.append(key)
        for node in valid_nodes:
            prev = node.prev
            # delete node, change prev to key
            prev.next = node.next
            if node.next:
                node.next.prev = prev
            prev.val = key
            if prev.prev is not None:
                new_pair = prev.prev.val + key
                tokens_dict[new_pair].append(prev)
                pq.put((-len(tokens_dict[new_pair]), counter, new_pair))
                counter += 1
            if node.next is not None:
                new_pair = key + prev.next.val
                tokens_dict[new_pair].append(prev.next)
                pq.put((-len(tokens_dict[new_pair]), counter, new_pair))
                counter += 1
            # note that we don't look for and remove old pairs, that's the 'laziness' from above
            # this means the priority queue is not 'perfect', but it's close enough
    print(f"{len(basic_chars)} single character tokens")
    print(f"{len(merged_tokens)} additional tokens")
    print(f"{len(merged_tokens) + len(basic_chars)} total tokens")
    return basic_chars, merged_tokens