"""High-Density Synthetic Reasoning Distillation Generator for Pseudo-Brain."""
import json
import math
import random
from typing import Any, Dict, Generator, List, Optional, Tuple

class SyntheticDistillationEngine:
    """Procedural generation engine for high-purity reasoning traces."""
    def __init__(self, seed: int = 42, max_threads: int = 16):
        self.rng = random.Random(seed)
        self.max_threads = max_threads

        self.geography = [
            ("Canada", "Ottawa"), ("Japan", "Tokyo"), ("France", "Paris"), ("Germany", "Berlin"),
            ("United Kingdom", "London"), ("Australia", "Canberra"), ("Italy", "Rome"), ("Spain", "Madrid"),
            ("Brazil", "Brasilia"), ("China", "Beijing"), ("India", "New Delhi"), ("South Korea", "Seoul"),
            ("Mexico", "Mexico City"), ("Egypt", "Cairo"), ("South Africa", "Pretoria"), ("Argentina", "Buenos Aires"),
            ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Switzerland", "Bern"), ("Netherlands", "Amsterdam"),
            ("Greece", "Athens"), ("Turkey", "Ankara"), ("Poland", "Warsaw"), ("Portugal", "Lisbon"),
            ("New Zealand", "Wellington"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Belgium", "Brussels"),
            ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Russia", "Moscow"), ("Ukraine", "Kyiv"),
            ("Czech Republic", "Prague"), ("Hungary", "Budapest"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
            ("Croatia", "Zagreb"), ("Serbia", "Belgrade"), ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"),
            ("Iceland", "Reykjavik"), ("Indonesia", "Jakarta"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
            ("Malaysia", "Kuala Lumpur"), ("Singapore", "Singapore"), ("Philippines", "Manila"), ("Saudi Arabia", "Riyadh"),
            ("United Arab Emirates", "Abu Dhabi"), ("Israel", "Jerusalem"), ("Chile", "Santiago"), ("Colombia", "Bogota"),
            ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("Morocco", "Rabat"), ("Ghana", "Accra"),
            ("Ethiopia", "Addis Ababa"), ("Tanzania", "Dodoma"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"),
            ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Jordan", "Amman"), ("Lebanon", "Beirut"), ("Cuba", "Havana"),
            ("Ecuador", "Quito"), ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"), ("Bolivia", "Sucre"),
            ("Venezuela", "Caracas"), ("Costa Rica", "San Jose"), ("Panama", "Panama City"), ("Jamaica", "Kingston"),
            ("Taiwan", "Taipei"), ("Mongolia", "Ulaanbaatar"), ("Nepal", "Kathmandu"), ("Kazakhstan", "Astana"),
            ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"), ("Estonia", "Tallinn"),
            ("Latvia", "Riga"), ("Lithuania", "Vilnius"), ("Luxembourg", "Luxembourg City"), ("Monaco", "Monaco"),
            ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Senegal", "Dakar"), ("Uganda", "Kampala"),
            ("Rwanda", "Kigali"), ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Botswana", "Gaborone"),
        ]

        self.science_qa = [
            ("Explain why ice floats on water.", "Ice floats on water because liquid water expands upon freezing into a crystalline solid. Water molecules form hydrogen bonds in an open hexagonal crystal lattice, making solid ice approximately 9% less dense than liquid water at 0°C. By Archimedes' principle, the buoyant force exceeds the weight of the submerged ice, causing it to float."),
            ("What is entropy in thermodynamics?", "Entropy is a fundamental thermodynamic state function that quantifies the number of accessible microscopic configurations (Omega) corresponding to a given macroscopic thermodynamic state, formulated by Ludwig Boltzmann as S = k_B * ln(Omega). According to the Second Law of Thermodynamics, the total entropy of an isolated system never decreases over time."),
            ("Why is the sky blue?", "The sky appears blue because of Rayleigh scattering of sunlight in Earth's atmosphere. Gas molecules (nitrogen and oxygen) are much smaller than visible light wavelengths and scatter shorter wavelengths (blue and violet light) much more intensely than longer wavelengths (red and yellow light), inversely proportional to the fourth power of the wavelength (1 / lambda^4)."),
            ("Explain quantum superposition.", "Quantum superposition is the physical principle that any valid quantum state can be expressed as a linear combination of two or more distinct orthogonal eigenstates: |psi> = sum c_i |phi_i>. The system simultaneously exhibits properties of all constituent states until a measurement operation collapses the state vector to a single eigenstate with probability |c_i|^2."),
            ("What is special relativity?", "Special relativity, formulated by Albert Einstein in 1905, rests on two core postulates: the laws of physics are identical across all inertial reference frames, and the speed of light in vacuum c is universal and independent of the motion of the source or observer. This leads to time dilation, Lorentz length contraction, and mass-energy equivalence E = m*c^2."),
            ("What is photosynthesis and why is it important?", "Photosynthesis is the biochemical pathway through which photoautotrophs convert light energy into chemical energy: 6CO2 + 6H2O + light -> C6H12O6 + 6O2. It is essential because it produces molecular oxygen for aerobic respiration and fixes atmospheric carbon dioxide into organic carbohydrates."),
            ("Describe the First Law of Thermodynamics.", "The First Law of Thermodynamics is the law of conservation of energy: energy can neither be created nor destroyed in an isolated system, only transformed from one form into another. The change in internal energy Delta U of a closed thermodynamic system equals the heat added Q minus the work done W: Delta U = Q - W."),
            ("What is CRISPR gene editing?", "CRISPR-Cas9 is a targeted molecular biology tool derived from bacterial adaptive immunity. A synthetic single-guide RNA (sgRNA) matches a 20-nucleotide target sequence in the genome adjacent to a Protospacer Adjacent Motif (PAM), directing the Cas9 endonuclease to introduce a double-strand break that can be repaired to silence or insert genes."),
            ("How do neurons transmit electrical signals?", "Neurons transmit signals via action potentials, which are transient reversals of electrical membrane potential. When a stimulus depolarizes the axon hillock to threshold (~ -55 mV), voltage-gated sodium channels open rapidly, causing Na+ influx. Subsequent opening of voltage-gated potassium channels allows K+ efflux, repolarizing the cell back toward resting potential (~ -70 mV)."),
            ("What causes ocean tides?", "Ocean tides are caused primarily by the gravitational gradient (differential gravitational pull) exerted on Earth by the Moon and the Sun. Because gravitational attraction decreases with the square of distance, ocean water on the side facing the Moon experiences stronger pull than Earth's center, while water on the far side experiences less pull, producing two simultaneous tidal bulges."),
            ("Explain Newton's Third Law of Motion.", "Newton's Third Law of Motion states that for every action force, there is an equal in magnitude and opposite in direction reaction force: F_AB = - F_BA. These action-reaction force pairs act on two different interacting bodies, meaning they never cancel each other out on a single object."),
            ("What is an exothermic reaction?", "An exothermic reaction is a chemical process that releases thermal energy into its surroundings, characterized by a negative enthalpy change (Delta H < 0). The energy released by forming new chemical bonds in the products exceeds the energy required to break bonds in the reactants."),
            ("What is an endothermic reaction?", "An endothermic reaction is a chemical process that absorbs thermal energy from its surroundings, characterized by a positive enthalpy change (Delta H > 0). The energy required to break bonds in the reactants is greater than the energy released when new bonds form in the products."),
            ("What is the greenhouse effect?", "The greenhouse effect is the atmospheric process where greenhouse gases (water vapor, carbon dioxide, methane) absorb and re-radiate infrared energy emitted by Earth's sun-warmed surface. This trapped thermal radiation prevents rapid heat loss to space, maintaining a habitable surface temperature."),
            ("What is nuclear fusion?", "Nuclear fusion is the nuclear process in which two light atomic nuclei (such as deuterium and tritium) combine under extreme temperature and pressure to form a heavier nucleus (such as helium-4), releasing enormous energy due to mass defect conversion governed by E = Delta m * c^2."),
            ("Why does sound travel faster in solids than in air?", "Sound is a mechanical compression wave that propagates via molecular collisions. In solids, atoms are bound by strong intermolecular forces and packed much closer together than gas molecules in air, allowing pressure fluctuations and vibrational momentum to transfer significantly faster."),
            ("What is DNA polymerase?", "DNA polymerase is the enzyme responsible for synthesizing new DNA strands during replication. It reads a template strand in the 3' to 5' direction and synthesizes a complementary daughter strand in the 5' to 3' direction by adding complementary deoxynucleotides via phosphodiester bonds."),
            ("What is the function of mitochondria?", "Mitochondria are membrane-bound organelles that generate most of the cell's chemical energy supply. Through the citric acid cycle and oxidative phosphorylation across the inner cristae membrane, they produce adenosine triphosphate (ATP), the primary energy currency of biological cells."),
            ("What is plate tectonics?", "Plate tectonics is the scientific theory explaining that Earth's outer lithosphere consists of individual tectonic plates gliding over the ductile asthenosphere. Their interactions at divergent, convergent, and transform boundaries drive earthquakes, volcanic activity, and mountain formation."),
        ]

        self.algorithms = [
            ("quicksort", 'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)'),
            ("binary_search", 'def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1'),
            ("is_prime", 'def is_prime(n):\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True'),
            ("fibonacci", 'def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b'),
            ("is_palindrome", "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"),
            ("find_max", "def find_max(arr):\n    if not arr:\n        raise ValueError('Array is empty')\n    m = arr[0]\n    for x in arr[1:]:\n        if x > m:\n            m = x\n    return m"),
            ("gcd", 'def gcd(a, b):\n    while b != 0:\n        a, b = b, a % b\n    return a'),
            ("lcm", 'def lcm(a, b):\n    def gcd(x, y):\n        while y:\n            x, y = y, x % y\n        return x\n    return abs(a * b) // gcd(a, b)'),
            ("reverse_list", 'def reverse_list(arr):\n    return arr[::-1]'),
            ("mergesort", 'def mergesort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = mergesort(arr[:mid])\n    right = mergesort(arr[mid:])\n    res, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            res.append(left[i]); i += 1\n        else:\n            res.append(right[j]); j += 1\n    res.extend(left[i:]); res.extend(right[j:])\n    return res'),
            ("bubble_sort", 'def bubble_sort(arr):\n    n = len(arr)\n    a = arr.copy()\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if a[j] > a[j + 1]:\n                a[j], a[j + 1] = a[j + 1], a[j]\n                swapped = True\n        if not swapped:\n            break\n    return a'),
            ("two_sum", 'def two_sum(nums, target):\n    lookup = {}\n    for i, num in enumerate(nums):\n        diff = target - num\n        if diff in lookup:\n            return [lookup[diff], i]\n        lookup[num] = i\n    return []'),
            ("factorial", "def factorial(n):\n    if n < 0:\n        raise ValueError('Negative input')\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res"),
            ("sieve_of_eratosthenes", 'def sieve_of_eratosthenes(limit):\n    if limit < 2:\n        return []\n    is_prime = [True] * (limit + 1)\n    is_prime[0] = is_prime[1] = False\n    for p in range(2, int(limit**0.5) + 1):\n        if is_prime[p]:\n            for multiple in range(p * p, limit + 1, p):\n                is_prime[multiple] = False\n    return [p for p, flag in enumerate(is_prime) if flag]'),
            ("count_vowels", "def count_vowels(s):\n    vowels = set('aeiouAEIOU')\n    return sum(1 for c in s if c in vowels)"),
            ("is_anagram", "def is_anagram(s1, s2):\n    return sorted(s1.replace(' ', '').lower()) == sorted(s2.replace(' ', '').lower())"),
        ]

    def sample_math(self) -> str:
        thread_id = self.rng.randint(0, self.max_threads - 1)
        prob_type = self.rng.choice(["linear", "linear_sub", "quadratic", "distance_word", "percentage", "rectangle_area"])
        if prob_type == "linear":
            a = self.rng.randint(2, 20)
            b = self.rng.randint(1, 50)
            x_ans = self.rng.randint(1, 25)
            c = a * x_ans + b
            return (
                f"[THREAD:{thread_id}]Solve for x: {a}*x + {b} = {c} "
                f"[RESP]<thought> Step 1: Subtract {b} from both sides: {a}*x = {c} - {b} = {c - b}. "
                f"Step 2: Divide both sides by {a}: x = {c - b} / {a} = {x_ans}. </thought> "
                f"<solution>x = {x_ans}</solution>[EOS]"
            )
        elif prob_type == "linear_sub":
            a = self.rng.randint(2, 20)
            b = self.rng.randint(1, 50)
            x_ans = self.rng.randint(2, 25)
            c = a * x_ans - b
            return (
                f"[THREAD:{thread_id}]Solve for x: {a}*x - {b} = {c} "
                f"[RESP]<thought> Step 1: Add {b} to both sides: {a}*x = {c} + {b} = {c + b}. "
                f"Step 2: Divide both sides by {a}: x = {c + b} / {a} = {x_ans}. </thought> "
                f"<solution>x = {x_ans}</solution>[EOS]"
            )
        elif prob_type == "quadratic":
            r = self.rng.randint(2, 15)
            c = r * r
            return (
                f"[THREAD:{thread_id}]Solve for x: x^2 - {c} = 0 "
                f"[RESP]<thought> Step 1: Add {c} to both sides: x^2 = {c}. "
                f"Step 2: Take the square root of both sides: x = sqrt({c}) or x = -sqrt({c}). "
                f"Therefore, x = {r} or x = -{r}. </thought> "
                f"<solution>x = {r}, -{r}</solution>[EOS]"
            )
        elif prob_type == "distance_word":
            speed = self.rng.choice([30, 45, 50, 60, 65, 70, 75, 80, 90, 100])
            hours = self.rng.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
            dist = int(speed * hours) if (speed * hours).is_integer() else speed * hours
            vehicle = self.rng.choice(["car", "train", "bus", "cyclist", "runner"])
            unit = self.rng.choice(["miles", "kilometers"])
            speed_unit = "mph" if unit == "miles" else "km/h"
            return (
                f"[THREAD:{thread_id}]If a {vehicle} travels at {speed} {speed_unit} for {hours} hours, how far does it travel? "
                f"[RESP]<thought> Step 1: Recall distance formula: Distance = Speed * Time. "
                f"Step 2: Substitute: Distance = {speed} * {hours} = {dist}. </thought> "
                f"<solution>{dist} {unit}</solution>[EOS]"
            )
        elif prob_type == "rectangle_area":
            w = self.rng.randint(3, 30)
            h = self.rng.randint(3, 40)
            area = w * h
            return (
                f"[THREAD:{thread_id}]What is the area of a rectangle with width {w} and height {h}? "
                f"[RESP]<thought> Step 1: Area = width * height. "
                f"Step 2: Multiply: {w} * {h} = {area}. </thought> "
                f"<solution>{area}</solution>[EOS]"
            )
        else:
            base = self.rng.choice([40, 50, 80, 100, 150, 200, 250, 400, 500, 800, 1000])
            pct = self.rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])
            val = (base * pct) // 100
            return (
                f"[THREAD:{thread_id}]What is {pct}% of {base}? "
                f"[RESP]<thought> Step 1: Convert percentage to decimal: {pct}% = {pct / 100}. "
                f"Step 2: Multiply: {base} * {pct / 100} = {val}. </thought> "
                f"<solution>{val}</solution>[EOS]"
            )

    def sample_code(self) -> str:
        thread_id = self.rng.randint(0, self.max_threads - 1)
        name, code = self.rng.choice(self.algorithms)
        stems = [
            f"Write {name} in Python.",
            f"Implement {name} in Python.",
            f"Write a Python function for {name}.",
            f"Show an implementation of {name} in Python.",
            f"How do you implement {name} in Python?",
        ]
        prompt = self.rng.choice(stems)
        return f"[THREAD:{thread_id}]{prompt} [RESP]{code}[EOS]"

    def sample_geography(self) -> str:
        thread_id = self.rng.randint(0, self.max_threads - 1)
        country, capital = self.rng.choice(self.geography)
        stems = [
            f"What is the capital of {country}?",
            f"Name the capital city of {country}.",
            f"What is {country}'s capital?",
            f"Which city is the capital of {country}?",
            f"Can you tell me the capital of {country}?",
        ]
        prompt = self.rng.choice(stems)
        answers = [
            f"The capital of {country} is {capital}.",
            f"{capital} is the capital of {country}.",
            f"The capital city of {country} is {capital}.",
        ]
        answer = self.rng.choice(answers)
        return f"[THREAD:{thread_id}]{prompt} [RESP]{answer}[EOS]"

    def sample_science(self) -> str:
        thread_id = self.rng.randint(0, self.max_threads - 1)
        q, a = self.rng.choice(self.science_qa)
        return f"[THREAD:{thread_id}]{q} [RESP]{a}[EOS]"

    def sample_item(self) -> str:
        r = self.rng.random()
        if r < 0.35:
            return self.sample_math()
        elif r < 0.65:
            return self.sample_code()
        elif r < 0.85:
            return self.sample_science()
        else:
            return self.sample_geography()

if __name__ == "__main__":
    engine = SyntheticDistillationEngine()
    print("Math:", engine.sample_math())
    print("Code:", engine.sample_code())
    print("Geo:", engine.sample_geography())
    print("Science:", engine.sample_science())