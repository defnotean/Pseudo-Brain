"""Curated high-frequency English subword vocabulary for conversational fluency.

Provides whole-word and common morpheme tokens with space-prefixed variants,
enabling Pseudo-Brain to read and write fluent English at 4-6x token compression
without falling into character-level autoregressive loops.
"""

from __future__ import annotations
from typing import List

_RAW_SUBWORDS: List[str] = [
    # Basic punctuation & symbols
    " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "\n", "\t",
    # Contractions, punctuation clusters, and sentence boundaries
    "'s", "'re", "'ve", "'ll", "'d", "'m", "n't", "...", "?!", "--", ",\"", ".\"", "!\n", "?\n", ": ", ", ", ". ", "? ", "! ",
    # Most common English words (bare form)
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on", "with",
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
    "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
    "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    "has", "had", "been", "am", "doing", "done", "did", "does", "said", "says", "made", "makes",
    "went", "goes", "came", "comes", "took", "takes", "knew", "knows", "saw", "sees", "thought", "thinks",
    # Most common English words (space-prefixed form)
    " the", " be", " to", " of", " and", " a", " in", " that", " have", " I", " it", " for", " not",
    " on", " with", " he", " as", " you", " do", " at", " this", " but", " his", " by", " from",
    " they", " we", " say", " her", " she", " or", " an", " will", " my", " one", " all", " would",
    " there", " their", " what", " so", " up", " out", " if", " about", " who", " get", " which",
    " go", " me", " when", " make", " can", " like", " time", " no", " just", " him", " know",
    " take", " people", " into", " year", " your", " good", " some", " could", " them", " see",
    " other", " than", " then", " now", " look", " only", " come", " its", " over", " think",
    " also", " back", " after", " use", " two", " how", " our", " work", " first", " well", " way",
    " even", " new", " want", " because", " any", " these", " give", " day", " most", " us",
    " is", " are", " was", " were", " has", " had", " been", " am", " doing", " done", " did",
    " does", " said", " says", " made", " makes", " went", " goes", " came", " comes", " took",
    " takes", " knew", " knows", " saw", " sees", " thought", " thinks",
    # Capitalized common words
    "The", "To", "Of", "And", "A", "In", "That", "Have", "It", "For", "Not", "On", "With",
    "He", "As", "You", "Do", "At", "This", "But", "His", "By", "From", "They", "We", "Say",
    "Her", "She", "Or", "An", "Will", "My", "One", "All", "Would", "There", "Their", "What",
    "So", "Up", "Out", "If", "About", "Who", "Get", "Which", "Go", "Me", "When", "Make",
    "Can", "Like", "Time", "No", "Just", "Him", "Know", "Take", "People", "Into", "Year",
    "Your", "Good", "Some", "Could", "Them", "See", "Other", "Than", "Then", "Now", "Look",
    "Only", "Come", "Its", "Over", "Think", "Also", "Back", "After", "Use", "Two", "How",
    "Our", "Work", "First", "Well", "Way", "Even", "New", "Want", "Because", "Any", "These",
    "Give", "Day", "Most", "Us", "Is", "Are", "Was", "Were", "Has", "Had", "Been",
    # Conversational Greetings, Salutations & Phrases
    "Hello", "hello", " Hello", " hello", "Hi", "hi", " Hi", " hi", "Hey", "hey", " Hey", " hey",
    "Good morning", "good morning", " Good morning", " good morning",
    "Good afternoon", "good afternoon", " Good afternoon", " good afternoon",
    "Good evening", "good evening", " Good evening", " good evening",
    "gamers", "gamer", " gamers", " gamer", "game", "games", " game", " games",
    "how are you", "How are you", " how are you", " How are you",
    "I am", "I'm", "I've", "I'd", "I'll", " I am", " I'm", " I've", " I'd", " I'll",
    "doing well", "doing great", " doing well", " doing great",
    "great to hear", "good to see you", "nice to meet you", "pleasure to meet you",
    "thank you", "Thank you", " thank you", " Thank you", "thanks", "Thanks", " thanks", " Thanks",
    "welcome", "Welcome", " welcome", " Welcome", "you're welcome", "You're welcome",
    "please", "Please", " please", " Please", "sure", "Sure", " sure", " Sure",
    "absolutely", "certainly", "definitely", "probably", "perhaps", "maybe",
    "yes", "Yes", " yes", " Yes", "yeah", "Yeah", " yeah", " Yeah",
    "no", "No", " no", " No", "nope", "Nope",
    "okay", "Okay", " okay", " Okay", "ok", "Ok", " ok", " Ok",
    "alright", "Alright", " alright", " Alright",
    "help", "Help", " help", " Help", "assist", "assist you", " assist", " assist you",
    "how can I help", "How can I help", "What can I do for you",
    "question", "questions", " question", " questions", "answer", "answers", " answer", " answers",
    "tell me", "Tell me", " tell me", " Tell me", "explain", "Explain", " explain", " Explain",
    # Agent Persona & Cognitive Architecture Terms
    "Pseudo", "Brain", "Pseudo-Brain", " Pseudo", " Brain", " Pseudo-Brain",
    "recurrent", "cognitive", "neural", "network", "recurrent neural network", "RNN",
    " recurrent", " cognitive", " neural", " network", " RNN",
    "memory", "working memory", "slots", "thought", "thoughts", "thread", "threads",
    " memory", " working memory", " slots", " thought", " thoughts", " thread", " threads",
    "preemption", "interruption", "resumption", "preempt", "resume",
    " preemption", " interruption", " resumption", " preempt", " resume",
    "persistent", "persistence", "latching", "plastic", "synapse", "synapses", "synaptic",
    "buffer", "replay", "zero replay", "streaming", "latency", "real-time", "60 Hz",
    " buffer", " replay", " zero replay", " streaming", " latency", " real-time", " 60 Hz",
    "DirectML", "GPU", "AMD", "Radeon", "hardware", "accelerator", "VRAM",
    " DirectML", " GPU", " AMD", " Radeon", " hardware", " accelerator", " VRAM",
    "RX 9070 XT", "RX 9070", "9070 XT", "RDNA",
    # Knowledge Domain: Science & Life
    "photosynthesis", "chlorophyll", "plants", "sunlight", "carbon", "oxygen", "energy",
    " photosynthesis", " chlorophyll", " plants", " sunlight", " carbon", " oxygen", " energy",
    "water", "glucose", "leaves", "green", "cells", "cellular", "organism", "living",
    " water", " glucose", " leaves", " green", " cells", " cellular",
    "physics", "chemistry", "biology", "science", "scientific", "experiment", "evidence",
    " gravity", " mass", " velocity", " planet", " earth", " moon", " sun", " solar",
    # Entities, Names, and Factual Anchors
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace",
    " Alice", " Bob", " Charlie", " David", " Eve", " Frank", " Grace",
    "coffee", "tea", "water", "juice", "milk", "pizza", "apple", "bread",
    " coffee", " tea", " water", " juice", " milk", " pizza", " apple", " bread",
    "Tokyo", "Paris", "London", "Toronto", "Berlin", "New York", "San Francisco",
    " Tokyo", " Paris", " London", " Toronto", " Berlin", " New York", " San Francisco",
    "Alpha", "Beta", "Gamma", "Delta", "project", "Friday", "Monday", "Wednesday",
    " Alpha", " Beta", " Gamma", " Delta", " project", " Friday", " Monday", " Wednesday",
    "due", "due on", " due", " due on", "deadline", "scheduled", "completed",
    "likes", "prefers", "enjoys", "loves", "hates", "owns", "lives in",
    " likes", " prefers", " enjoys", " loves", " hates", " owns", " lives in",
    # Common English Adjectives, Adverbs & Connectives
    "great", "Great", " great", " Great", "fine", "Fine", " fine", " Fine",
    "good", "better", "best", "bad", "worse", "worst",
    " good", " better", " best", " bad", " worse", " worst",
    "important", "special", "unique", "simple", "complex", "clear", "natural", "fluent",
    " important", " special", " unique", " simple", " complex", " clear", " natural", " fluent",
    "fast", "slow", "high", "low", "deep", "wide", "large", "small", "huge",
    " fast", " slow", " high", " low", " deep", " wide", " large", " small",
    "very", "really", "much", "many", "little", "few", "more", "less",
    " very", " really", " much", " many", " little", " few", " more", " less",
    "always", "never", "sometimes", "often", "usually", "rarely",
    " always", " never", " sometimes", " often", " usually",
    "today", "now", "soon", "later", "before", "while", "during", "since", "until",
    " today", " now", " soon", " later", " before", " while", " during", " since", " until",
    "here", "there", "where", "everywhere", "anywhere", "nowhere",
    " here", " there", " where", " everywhere", " anywhere",
    "something", "anything", "nothing", "everything", "someone", "anyone", "everyone",
    " because", " since", " although", " though", " however", " therefore", " furthermore",
    "talk", "speak", "chat", "conversation", "discuss", "discussion",
    "understand", "understanding", "knowledge", "reasoning", "logic", "intelligence",
    "world", "human", "humans", "person", "people", "friend", "friends", "team",
    "computer", "computers", "program", "programming", "software", "code", "algorithm",
    "ing", "ed", "ly", "tion", "able", "ment", "ness", "er", "est", "ful", "less", "ize", "ise",
    "al", "ic", "ous", "ive", "ity", "ies", "es",
]

# De-duplicate while preserving exact order
_SEEN = set()
CONVERSATIONAL_SUBWORDS: List[str] = []
for _w in _RAW_SUBWORDS:
    if _w and _w not in _SEEN:
        _SEEN.add(_w)
        CONVERSATIONAL_SUBWORDS.append(_w)
