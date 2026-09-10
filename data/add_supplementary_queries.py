import json
import os

QUERIES_FILE = "data/queries.json"

new_coding_queries = [
    # trees
    ("coding", "trees", "Write a function to find the height of a binary tree."),
    ("coding", "trees", "Write code to check if a binary tree is a valid binary search tree."),
    ("coding", "trees", "Write a function to perform an in-order traversal of a binary tree without recursion."),
    ("coding", "trees", "Write code to find the lowest common ancestor of two nodes in a binary tree."),
    # bit manipulation
    ("coding", "bit_manipulation", "Write a function to count the number of set bits in an integer."),
    ("coding", "bit_manipulation", "Write code to check if a number is a power of two using bitwise operations."),
    ("coding", "bit_manipulation", "Write a function to find the single number in an array where every other number appears twice."),
    # stacks and queues
    ("coding", "stacks_queues", "Write code to check if a string of parentheses is balanced using a stack."),
    ("coding", "stacks_queues", "Implement a queue using two stacks."),
    ("coding", "stacks_queues", "Write a function to evaluate a postfix expression using a stack."),
    ("coding", "stacks_queues", "Implement a min-stack that supports push, pop, and retrieving the minimum in O(1)."),
    # hashmaps
    ("coding", "hashmaps", "Write a function to find the first non-repeating character in a string using a hashmap."),
    ("coding", "hashmaps", "Write code to group anagrams together from a list of strings."),
    ("coding", "hashmaps", "Write a function to find two numbers in an array that sum to a target value."),
    # matrix
    ("coding", "matrix", "Write a function to rotate an N x N matrix 90 degrees in place."),
    ("coding", "matrix", "Write code to search for a target value in a row-wise and column-wise sorted matrix."),
    ("coding", "matrix", "Write a function to find the number of islands in a 2D grid of 1s and 0s."),
    # greedy
    ("coding", "greedy", "Write a greedy algorithm to solve the activity selection problem."),
    ("coding", "greedy", "Write code to find the minimum number of coins needed using a greedy approach, and explain when it fails."),
    ("coding", "greedy", "Write a function to determine the maximum profit from at most one stock transaction."),
    # additional variety
    ("coding", "arrays", "Write a function to find all pairs in an array that sum to a given target, without using extra space for a hashmap."),
    ("coding", "recursion", "Write a recursive function to generate all permutations of a string."),
    ("coding", "dynamic_programming", "Write a DP solution to find the length of the longest increasing subsequence."),
    ("coding", "debugging", "Why does this recursive Fibonacci implementation become extremely slow for n > 35, and how would you fix it?"),
]

with open(QUERIES_FILE, encoding="utf-8") as f:
    existing = json.load(f)

max_id = max(item["id"] for item in existing)
added = 0
for category, subtopic, text in new_coding_queries:
    max_id += 1
    existing.append({
        "id": max_id,
        "category": category,
        "subtopic": subtopic,
        "query": text
    })
    added += 1

with open(QUERIES_FILE, "w", encoding="utf-8") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

print(f"Added {added} new supplementary coding queries.")
print(f"New total queries in {QUERIES_FILE}: {len(existing)}")