import json
import os

queries = []


def add(category, subtopic, text):
    queries.append({
        "id": len(queries) + 1,
        "category": category,
        "subtopic": subtopic,
        "query": text
    })


# ---------- SIMPLE / FACTUAL ----------
geo = [
    "What is the capital of Japan?",
    "Which country has the largest population in the world?",
    "What is the longest river in the world?",
    "How many continents are there?",
    "What is the capital of Australia?",
    "Which ocean is the largest?",
    "What is the smallest country in the world?",
    "Name the tallest mountain on Earth.",
]
for q in geo:
    add("simple", "geography", q)

science_facts = [
    "What is the chemical symbol for gold?",
    "How many planets are there in our solar system?",
    "What gas do plants absorb from the atmosphere?",
    "What is the boiling point of water in Celsius?",
    "What is the speed of light in a vacuum?",
    "How many bones are there in the adult human body?",
]
for q in science_facts:
    add("simple", "science_fact", q)

math_basic = [
    "What is 15 multiplied by 12?",
    "Convert 5 kilometers to miles.",
    "What is the square root of 144?",
    "How many minutes are there in a day?",
    "What is 7 squared?",
    "Convert 100 Fahrenheit to Celsius.",
]
for q in math_basic:
    add("simple", "basic_math", q)

history_dates = [
    "In which year did World War II end?",
    "Who was the first President of the United States?",
    "When was the Declaration of Independence signed?",
    "Who invented the telephone?",
    "In what year did India gain independence?",
]
for q in history_dates:
    add("simple", "history_fact", q)

definitions = [
    "What does the acronym NASA stand for?",
    "What is the definition of photosynthesis?",
    "What does HTTP stand for?",
    "What is the meaning of the word 'ubiquitous'?",
    "What does GDP stand for in economics?",
]
for q in definitions:
    add("simple", "definition", q)

general_knowledge = [
    "Who wrote the play Romeo and Juliet?",
    "What is the currency used in Japan?",
    "How many strings does a standard guitar have?",
    "What is the national sport of Japan?",
    "Which planet is known as the Red Planet?",
    "What is the largest mammal on Earth?",
    "How many colors are there in a rainbow?",
]
for q in general_knowledge:
    add("simple", "general_knowledge", q)

conversions_units = [
    "Convert 1 hour into seconds.",
    "How many grams are there in a kilogram?",
    "Convert 50 miles per hour into kilometers per hour.",
    "How many centimeters are in a meter?",
]
for q in conversions_units:
    add("simple", "conversions", q)


# ---------- COMPLEX / REASONING ----------
economics = [
    "Explain why increasing interest rates tends to reduce inflation.",
    "Why does a trade deficit not necessarily indicate a weak economy?",
    "Explain the relationship between unemployment and inflation using the Phillips curve.",
    "Why might a country choose to devalue its own currency?",
    "Explain how supply chain disruptions can cause inflation even without demand changes.",
]
for q in economics:
    add("complex", "economics", q)

science_reasoning = [
    "Explain why vaccines sometimes require multiple doses to be effective.",
    "Why does ice float on water despite being a solid form of it?",
    "Explain why nuclear fusion produces more energy than nuclear fission per unit mass.",
    "Why is it harder to boil water at higher altitudes?",
    "Explain how greenhouse gases trap heat in the atmosphere.",
]
for q in science_reasoning:
    add("complex", "science_reasoning", q)

philosophy_ethics = [
    "Is it ethical to use animals for medical testing? Present arguments on both sides.",
    "Explain the trolley problem and what it reveals about moral decision-making.",
    "Discuss whether free will can coexist with a fully deterministic universe.",
    "What is the difference between moral relativism and moral absolutism?",
]
for q in philosophy_ethics:
    add("complex", "philosophy", q)

comparative_analysis = [
    "Compare the advantages and disadvantages of renewable energy versus nuclear power.",
    "Compare capitalism and socialism in terms of economic incentives.",
    "Compare the long-term effects of remote work versus in-office work on productivity.",
    "Compare monarchies and democracies in terms of long-term political stability.",
]
for q in comparative_analysis:
    add("complex", "comparative_analysis", q)

multi_step_reasoning = [
    "If a train leaves City A at 60 km/h and another leaves City B at 90 km/h toward each other 300 km apart, when do they meet?",
    "A company's revenue grows 10% each year. If it starts at $1 million, what will it be after 5 years, and why does this differ from simple 50% growth?",
    "Explain step by step why a Ponzi scheme inevitably collapses.",
    "Walk through the reasoning for why compound interest grows faster than simple interest over time.",
]
for q in multi_step_reasoning:
    add("complex", "multi_step_reasoning", q)

cause_effect = [
    "Explain how deforestation contributes to changes in regional rainfall patterns.",
    "Why did the 2008 financial crisis spread from the housing market to the global economy?",
    "Explain how overfishing can collapse an entire marine ecosystem.",
    "Why can a small change in interest rates have a large effect on stock markets?",
    "Explain why antibiotic overuse leads to drug-resistant bacteria over time.",
]
for q in cause_effect:
    add("complex", "cause_effect", q)

technology_reasoning = [
    "Explain why increasing a neural network's size doesn't always improve its accuracy.",
    "Why do distributed systems need to make trade-offs described by the CAP theorem?",
    "Explain why encryption alone doesn't guarantee complete data security.",
    "Why can caching improve performance but also introduce data consistency issues?",
    "Explain why more CPU cores don't always make a program run proportionally faster.",
]
for q in technology_reasoning:
    add("complex", "technology_reasoning", q)

social_reasoning = [
    "Explain why raising the minimum wage can have mixed effects on employment.",
    "Discuss why social media algorithms tend to amplify polarizing content.",
    "Explain why urbanization often leads to declining birth rates.",
]
for q in social_reasoning:
    add("complex", "social_reasoning", q)


# ---------- CODING ----------
arrays = [
    "Write a Python function to find the maximum subarray sum (Kadane's algorithm).",
    "Write a function to rotate an array to the right by k steps.",
    "Write code to find the second largest element in an array.",
    "Write a function to remove duplicates from a sorted array in place.",
]
for q in arrays:
    add("coding", "arrays", q)

linked_lists = [
    "Write a Python function to reverse a singly linked list.",
    "How do I detect a cycle in a linked list?",
    "Write code to find the middle node of a linked list in one pass.",
    "Write a function to merge two sorted linked lists.",
]
for q in linked_lists:
    add("coding", "linked_lists", q)

recursion = [
    "Write a recursive function to compute the nth Fibonacci number.",
    "Write a recursive function to calculate the factorial of a number.",
    "Write a recursive function to check if a string is a palindrome.",
    "Explain and implement recursive binary search.",
]
for q in recursion:
    add("coding", "recursion", q)

dynamic_programming = [
    "Write a dynamic programming solution for the 0/1 knapsack problem.",
    "Write code to find the longest common subsequence between two strings.",
    "Solve the coin change problem using dynamic programming.",
    "Write a DP solution to find the minimum number of edits to convert one string to another.",
]
for q in dynamic_programming:
    add("coding", "dynamic_programming", q)

sorting_searching = [
    "Implement quicksort in Python.",
    "Implement merge sort and explain its time complexity.",
    "Write a function to perform binary search on a rotated sorted array.",
    "Explain the difference between quicksort and mergesort in terms of worst-case complexity.",
]
for q in sorting_searching:
    add("coding", "sorting_searching", q)

graphs = [
    "Write code to perform a breadth-first search on a graph.",
    "Write a function to detect a cycle in a directed graph.",
    "Implement Dijkstra's algorithm for shortest path.",
    "Write code to check if a graph is bipartite.",
]
for q in graphs:
    add("coding", "graphs", q)

sql_databases = [
    "Write a SQL query to find the second highest salary from an Employees table.",
    "Write a SQL query to find duplicate rows in a table.",
    "Write a SQL query to join three tables and filter by a date range.",
    "Explain the difference between INNER JOIN and LEFT JOIN with an example.",
]
for q in sql_databases:
    add("coding", "sql", q)

debugging_general = [
    "Why does this Python code throw an IndexError: list index out of range?",
    "Debug this function that is supposed to reverse a string but returns the same string.",
    "Why is my recursive function causing a stack overflow?",
    "Explain why this code has an off-by-one error in a for loop.",
]
for q in debugging_general:
    add("coding", "debugging", q)

oop_design = [
    "Explain the difference between abstraction and encapsulation with a code example.",
    "Write a Python class hierarchy demonstrating inheritance and polymorphism for shapes.",
    "Implement the singleton design pattern in Python.",
    "Explain when to use composition over inheritance, with an example.",
]
for q in oop_design:
    add("coding", "oop_design", q)

strings_regex = [
    "Write a regular expression to validate an email address.",
    "Write a function to check if two strings are anagrams of each other.",
    "Write code to find the longest palindromic substring in a string.",
]
for q in strings_regex:
    add("coding", "strings_regex", q)


# ---------- CREATIVE / WRITING ----------
poems = [
    "Write a short poem about autumn rain.",
    "Write a haiku about the ocean at sunset.",
    "Write a poem about a lighthouse guiding lost sailors home.",
    "Write a short poem about the feeling of starting a new city.",
]
for q in poems:
    add("creative", "poem", q)

short_stories = [
    "Write a short story about a robot discovering emotions for the first time.",
    "Write a short story about two strangers stuck in an elevator during a storm.",
    "Write a short story about a child who finds a door to another world in their attic.",
    "Write a short mystery story set in a small isolated village.",
]
for q in short_stories:
    add("creative", "short_story", q)

dialogue_scripts = [
    "Write a short dialogue between a detective and a suspect who is clearly lying.",
    "Write a conversation between two old friends meeting after 20 years.",
    "Write a dialogue between a scientist and an alien meeting for the first time.",
]
for q in dialogue_scripts:
    add("creative", "dialogue", q)

taglines_naming = [
    "Suggest a creative tagline for a coffee shop that opens only at night.",
    "Suggest a name and tagline for a startup that delivers plants to apartments.",
    "Write a catchy slogan for an eco-friendly clothing brand.",
]
for q in taglines_naming:
    add("creative", "tagline_naming", q)

descriptive_writing = [
    "Describe a bustling street market in vivid sensory detail.",
    "Describe an abandoned amusement park at night.",
    "Describe the feeling of standing at the edge of a cliff overlooking the sea.",
    "Write a vivid description of a thunderstorm rolling into a quiet town.",
]
for q in descriptive_writing:
    add("creative", "descriptive_writing", q)

letters_speeches = [
    "Write a heartfelt letter from a soldier to their family during wartime, in a fictional voice.",
    "Write a short motivational speech for a team about to compete in a big tournament.",
    "Write a farewell letter from a retiring teacher to their students.",
    "Write a toast speech for a friend's wedding.",
]
for q in letters_speeches:
    add("creative", "letters_speeches", q)

metaphor_analogy = [
    "Write a metaphor comparing time to a river, and expand it into a short paragraph.",
    "Describe love using an extended analogy to gardening.",
    "Write a short piece comparing life to a marathon.",
]
for q in metaphor_analogy:
    add("creative", "metaphor_analogy", q)

fictional_worldbuilding = [
    "Describe a futuristic city where it never stops raining.",
    "Invent a fictional creature and describe its habitat and behavior.",
    "Describe a hidden village that exists inside a giant tree.",
    "Write a short piece introducing a fantasy kingdom on the edge of extinction.",
]
for q in fictional_worldbuilding:
    add("creative", "worldbuilding", q)

song_style_verses = [
    "Write an original short verse (not from any existing song) about chasing dreams in a big city.",
    "Write an original short verse about missing home while traveling far away.",
    "Write an original short verse celebrating friendship and good times.",
]
for q in song_style_verses:
    add("creative", "original_verse", q)

humor_satire = [
    "Write a humorous short piece about a cat who thinks it runs the household.",
    "Write a satirical news headline and short blurb about Mondays being banned.",
    "Write a funny short story about a Wi-Fi router that develops a personality.",
]
for q in humor_satire:
    add("creative", "humor_satire", q)


print(f"Total queries generated: {len(queries)}")

from collections import Counter
counts = Counter(q["category"] for q in queries)
for cat, n in counts.items():
    print(f"  {cat}: {n}")

os.makedirs("data", exist_ok=True)
with open("data/queries.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2, ensure_ascii=False)

print("Saved to data/queries.json")