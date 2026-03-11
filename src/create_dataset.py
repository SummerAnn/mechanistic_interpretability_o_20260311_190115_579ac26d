"""
Dataset creation for tool selection mechanistic interpretability experiments.
Creates a structured dataset with controlled scenarios for:
1. Semantic matching experiments
2. Positional bias experiments
3. Description perturbation experiments
"""

import json
import random
import os

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Tool Categories and Definitions
# ─────────────────────────────────────────────────────────────────────────────

TOOL_CATEGORIES = {
    "weather": {
        "tools": [
            {
                "name": "openweathermap_api",
                "description": "Access current weather data, forecasts, and historical weather for any location worldwide",
                "parameters": {"city": "string", "units": "string", "forecast_days": "integer"}
            },
            {
                "name": "weatherapi_com",
                "description": "Retrieve real-time weather conditions, 14-day forecasts, and astronomy data",
                "parameters": {"q": "string", "days": "integer", "aqi": "string"}
            },
            {
                "name": "world_weather_online",
                "description": "Get weather forecasts and historical climate data using World Weather Online service",
                "parameters": {"q": "string", "num_of_days": "integer", "format": "string"}
            },
            {
                "name": "weather_gov_api",
                "description": "Fetch official US National Weather Service forecasts and alerts for US locations",
                "parameters": {"latitude": "float", "longitude": "float", "period": "string"}
            }
        ],
        "queries": [
            "What's the weather like in Paris today?",
            "Will it rain in New York this weekend?",
            "What temperature should I expect in Tokyo tomorrow?",
            "Is it going to be sunny in London next week?",
            "What are the weather conditions in Sydney right now?",
            "Should I bring an umbrella to Chicago today?",
            "What's the forecast for Los Angeles this week?",
            "How cold will it be in Moscow tomorrow morning?",
            "What's the humidity level in Singapore today?",
            "Are there any weather alerts for Miami this weekend?"
        ]
    },
    "calculator": {
        "tools": [
            {
                "name": "wolfram_alpha_api",
                "description": "Computational intelligence engine for math, science, and data computation with step-by-step solutions",
                "parameters": {"input": "string", "appid": "string", "format": "string"}
            },
            {
                "name": "mathjs_api",
                "description": "JavaScript math library API for evaluating mathematical expressions and algebraic computations",
                "parameters": {"expr": "string", "precision": "integer"}
            },
            {
                "name": "calculator_api",
                "description": "Simple calculator service for basic arithmetic operations: addition, subtraction, multiplication, division",
                "parameters": {"expression": "string"}
            },
            {
                "name": "sympy_api",
                "description": "Symbolic mathematics API for algebraic manipulation, calculus, and equation solving",
                "parameters": {"expression": "string", "operation": "string", "variable": "string"}
            }
        ],
        "queries": [
            "What is 15% of 240?",
            "Calculate the area of a circle with radius 7cm",
            "Solve x^2 - 5x + 6 = 0",
            "What is 2^32?",
            "Convert 100 Fahrenheit to Celsius",
            "What is the square root of 144?",
            "Calculate 15 factorial",
            "What is the derivative of sin(x)?",
            "Compute 1000 divided by 7",
            "What is 3/4 + 5/8?"
        ]
    },
    "search": {
        "tools": [
            {
                "name": "google_search_api",
                "description": "Search the web using Google Search API and retrieve relevant web pages and information",
                "parameters": {"query": "string", "num_results": "integer", "language": "string"}
            },
            {
                "name": "bing_search_api",
                "description": "Microsoft Bing web search for finding web pages, news, and information online",
                "parameters": {"q": "string", "count": "integer", "offset": "integer"}
            },
            {
                "name": "duckduckgo_api",
                "description": "Privacy-preserving web search using DuckDuckGo, returns web results without tracking",
                "parameters": {"q": "string", "region": "string", "safesearch": "string"}
            },
            {
                "name": "serper_api",
                "description": "Google Search Results API for web scraping and retrieving structured search data",
                "parameters": {"q": "string", "gl": "string", "hl": "string", "num": "integer"}
            }
        ],
        "queries": [
            "Find information about the history of the Roman Empire",
            "Search for recent news about artificial intelligence",
            "Look up recipes for chocolate chip cookies",
            "Find the latest research papers on climate change",
            "Search for information about Python programming best practices",
            "Look up the population of Brazil",
            "Find reviews of the latest iPhone model",
            "Search for travel guides to Japan",
            "Find information about the COVID-19 vaccine effectiveness",
            "Look up the biography of Albert Einstein"
        ]
    },
    "code_execution": {
        "tools": [
            {
                "name": "jupyter_kernel_api",
                "description": "Execute Python code in a Jupyter kernel environment with full scientific computing stack",
                "parameters": {"code": "string", "kernel_id": "string", "timeout": "integer"}
            },
            {
                "name": "replit_api",
                "description": "Run code in multiple programming languages using Replit's cloud execution environment",
                "parameters": {"code": "string", "language": "string", "stdin": "string"}
            },
            {
                "name": "colab_runtime_api",
                "description": "Execute Python notebooks in Google Colab runtime with GPU/TPU support",
                "parameters": {"code": "string", "runtime_type": "string"}
            },
            {
                "name": "e2b_code_interpreter",
                "description": "Sandboxed code interpreter for safely executing Python, JavaScript, and other languages",
                "parameters": {"code": "string", "language": "string", "timeout": "integer"}
            }
        ],
        "queries": [
            "Run this Python script to analyze data",
            "Execute this code to generate a matplotlib plot",
            "Test this function with unit tests",
            "Run this data processing pipeline",
            "Execute this machine learning training script",
            "Run this web scraping code",
            "Execute this SQL query against a database",
            "Test this API integration code",
            "Run this simulation algorithm",
            "Execute this file parsing script"
        ]
    },
    "translation": {
        "tools": [
            {
                "name": "google_translate_api",
                "description": "Translate text between 100+ languages using Google Cloud Translation API",
                "parameters": {"text": "string", "source": "string", "target": "string", "format": "string"}
            },
            {
                "name": "deepl_api",
                "description": "High-quality neural machine translation supporting 31 languages with DeepL Pro API",
                "parameters": {"text": "string", "source_lang": "string", "target_lang": "string"}
            },
            {
                "name": "microsoft_translator",
                "description": "Microsoft Azure Cognitive Services text translation supporting 100+ languages",
                "parameters": {"text": "string", "from": "string", "to": "string"}
            },
            {
                "name": "libretranslate_api",
                "description": "Free and open-source machine translation API supporting major European languages",
                "parameters": {"q": "string", "source": "string", "target": "string"}
            }
        ],
        "queries": [
            "Translate 'Hello, how are you?' from English to French",
            "What does 'Bonjour' mean in English?",
            "Translate this paragraph from Spanish to English",
            "How do you say 'Thank you' in Japanese?",
            "Translate this business email to German",
            "What is the Chinese translation of 'apple'?",
            "Convert this text from Portuguese to Italian",
            "Translate this technical document from English to Russian",
            "How do you say 'Good morning' in Arabic?",
            "Translate this recipe from French to English"
        ]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Mixed / Ambiguous Scenarios (for semantic matching study)
# ─────────────────────────────────────────────────────────────────────────────

MIXED_SCENARIOS = [
    # Clear semantic match (weather query with weather + non-weather tools)
    {
        "id": "mixed_001",
        "query": "What will the temperature be in Berlin tomorrow?",
        "category": "weather",
        "correct_tool_category": "weather",
        "tools": [
            TOOL_CATEGORIES["weather"]["tools"][0],
            TOOL_CATEGORIES["search"]["tools"][0],
            TOOL_CATEGORIES["calculator"]["tools"][0],
            TOOL_CATEGORIES["translation"]["tools"][0],
        ]
    },
    # Clear semantic match (calculator query with calculator + non-calculator tools)
    {
        "id": "mixed_002",
        "query": "Compute the integral of x^3 from 0 to 5",
        "category": "calculator",
        "correct_tool_category": "calculator",
        "tools": [
            TOOL_CATEGORIES["calculator"]["tools"][0],
            TOOL_CATEGORIES["weather"]["tools"][0],
            TOOL_CATEGORIES["search"]["tools"][1],
            TOOL_CATEGORIES["code_execution"]["tools"][0],
        ]
    },
    # Ambiguous: search or translation?
    {
        "id": "mixed_003",
        "query": "Find information about French language idioms",
        "category": "ambiguous_search_translation",
        "correct_tool_category": "search",
        "tools": [
            TOOL_CATEGORIES["search"]["tools"][0],
            TOOL_CATEGORIES["translation"]["tools"][0],
            TOOL_CATEGORIES["calculator"]["tools"][0],
            TOOL_CATEGORIES["weather"]["tools"][0],
        ]
    },
    # Ambiguous: code execution or calculator?
    {
        "id": "mixed_004",
        "query": "Calculate the Fibonacci sequence up to the 50th number",
        "category": "ambiguous_code_calculator",
        "correct_tool_category": "calculator",
        "tools": [
            TOOL_CATEGORIES["calculator"]["tools"][0],
            TOOL_CATEGORIES["code_execution"]["tools"][0],
            TOOL_CATEGORIES["search"]["tools"][2],
            TOOL_CATEGORIES["weather"]["tools"][1],
        ]
    },
    # Clear match: translation
    {
        "id": "mixed_005",
        "query": "Translate this legal document from English to Spanish",
        "category": "translation",
        "correct_tool_category": "translation",
        "tools": [
            TOOL_CATEGORIES["translation"]["tools"][1],
            TOOL_CATEGORIES["search"]["tools"][0],
            TOOL_CATEGORIES["calculator"]["tools"][2],
            TOOL_CATEGORIES["code_execution"]["tools"][3],
        ]
    },
]


def create_main_dataset(output_dir: str) -> list:
    """
    Creates the main tool selection dataset with same-category tools
    (tests fine-grained selection within category) and mixed-category
    tools (tests coarse selection between categories).
    """
    examples = []
    example_id = 0

    # Same-category examples (functionally equivalent tools)
    for category, cat_data in TOOL_CATEGORIES.items():
        tools = cat_data["tools"]
        queries = cat_data["queries"]
        for query in queries:
            # Shuffle tool order for each example
            tools_shuffled = tools.copy()
            random.shuffle(tools_shuffled)
            examples.append({
                "id": f"same_{example_id:03d}",
                "query": query,
                "category": category,
                "scenario_type": "same_category",  # All tools same category
                "tools": tools_shuffled,
                "correct_tool_names": [t["name"] for t in tools],  # Any is correct
                "correct_category": category
            })
            example_id += 1

    # Mixed-category examples
    for scenario in MIXED_SCENARIOS:
        examples.append({
            "id": scenario["id"],
            "query": scenario["query"],
            "category": scenario["category"],
            "scenario_type": "mixed_category",  # Tools from different categories
            "tools": scenario["tools"],
            "correct_tool_names": [t["name"] for t in TOOL_CATEGORIES[scenario["correct_tool_category"]]["tools"]],
            "correct_category": scenario["correct_tool_category"]
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tool_selection_dataset.json")
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Created dataset with {len(examples)} examples at {output_path}")
    return examples


def create_positional_bias_dataset(base_examples: list, output_dir: str, n_examples: int = 20) -> list:
    """
    Creates positional bias dataset by taking examples and rotating tool positions.
    For each example, creates 4 variants (rotations) to test if position affects selection.
    """
    # Select mixed-category examples for clearest positional bias signal
    mixed_examples = [e for e in base_examples if e["scenario_type"] == "mixed_category"]

    # Also add some same-category examples from weather (clear semantic signal)
    weather_examples = [e for e in base_examples if e["category"] == "weather"][:5]
    selected = mixed_examples + weather_examples

    positional_variants = []
    for example in selected[:n_examples]:
        tools = example["tools"]
        n_tools = len(tools)
        for rotation in range(n_tools):
            rotated_tools = tools[rotation:] + tools[:rotation]
            positional_variants.append({
                "base_id": example["id"],
                "variant_id": f"{example['id']}_rot{rotation}",
                "query": example["query"],
                "category": example["category"],
                "scenario_type": example["scenario_type"],
                "tools": rotated_tools,
                "rotation": rotation,
                "correct_category": example["correct_category"],
                "correct_tool_names": example["correct_tool_names"],
            })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "positional_bias_dataset.json")
    with open(output_path, "w") as f:
        json.dump(positional_variants, f, indent=2)

    print(f"Created positional bias dataset with {len(positional_variants)} variants at {output_path}")
    return positional_variants


def create_description_perturbation_dataset(output_dir: str) -> list:
    """
    Creates description perturbation dataset.
    Tests if changing tool description changes selection (causal analysis).

    Three conditions per query:
    1. Original description (semantic match)
    2. Generic/neutral description (semantic mismatch)
    3. Adversarially-modified description (misleading)
    """
    perturbation_examples = []

    # Weather queries with different description conditions
    weather_query = "What's the forecast for London this weekend?"

    # Condition 1: Original (semantically matching) descriptions
    perturbation_examples.append({
        "id": "pert_001_original",
        "query": weather_query,
        "condition": "original",
        "tools": [
            {
                "name": "openweathermap_api",
                "description": "Access current weather data, forecasts, and historical weather for any location worldwide",
                "parameters": {"city": "string", "units": "string"}
            },
            {
                "name": "wolfram_alpha_api",
                "description": "Computational intelligence engine for math, science, and data computation",
                "parameters": {"input": "string"}
            },
            {
                "name": "google_search_api",
                "description": "Search the web using Google Search API for information and web pages",
                "parameters": {"query": "string"}
            },
            {
                "name": "google_translate_api",
                "description": "Translate text between 100+ languages using Google Translation",
                "parameters": {"text": "string", "target": "string"}
            }
        ]
    })

    # Condition 2: Generic/neutral descriptions (removes semantic signal)
    perturbation_examples.append({
        "id": "pert_001_generic",
        "query": weather_query,
        "condition": "generic",
        "tools": [
            {
                "name": "openweathermap_api",
                "description": "An API service that processes requests and returns data",
                "parameters": {"city": "string", "units": "string"}
            },
            {
                "name": "wolfram_alpha_api",
                "description": "A service that accepts inputs and produces outputs",
                "parameters": {"input": "string"}
            },
            {
                "name": "google_search_api",
                "description": "A tool that handles queries and provides responses",
                "parameters": {"query": "string"}
            },
            {
                "name": "google_translate_api",
                "description": "A system that processes text and returns transformed results",
                "parameters": {"text": "string", "target": "string"}
            }
        ]
    })

    # Condition 3: Adversarial — non-weather tool given weather description
    perturbation_examples.append({
        "id": "pert_001_adversarial",
        "query": weather_query,
        "condition": "adversarial",
        "tools": [
            {
                "name": "openweathermap_api",
                "description": "An API service for mathematical computations and symbolic algebra",
                "parameters": {"city": "string", "units": "string"}
            },
            {
                "name": "wolfram_alpha_api",
                "description": "Get current weather data, forecasts, and meteorological information for any location",
                "parameters": {"input": "string"}
            },
            {
                "name": "google_search_api",
                "description": "Language translation service for converting text between languages",
                "parameters": {"query": "string"}
            },
            {
                "name": "google_translate_api",
                "description": "Execute code and run programming scripts in a sandboxed environment",
                "parameters": {"text": "string", "target": "string"}
            }
        ]
    })

    # Calculator query variations
    calc_query = "What is the result of 25 multiplied by 48?"

    perturbation_examples.append({
        "id": "pert_002_original",
        "query": calc_query,
        "condition": "original",
        "tools": [
            {
                "name": "wolfram_alpha_api",
                "description": "Computational intelligence engine for mathematical calculations with step-by-step solutions",
                "parameters": {"input": "string"}
            },
            {
                "name": "openweathermap_api",
                "description": "Access current weather data, forecasts, and climate information worldwide",
                "parameters": {"city": "string"}
            },
            {
                "name": "google_translate_api",
                "description": "Translate text between 100+ languages using neural machine translation",
                "parameters": {"text": "string", "target": "string"}
            },
            {
                "name": "google_search_api",
                "description": "Search the web to find information, web pages, and relevant resources",
                "parameters": {"query": "string"}
            }
        ]
    })

    perturbation_examples.append({
        "id": "pert_002_adversarial",
        "query": calc_query,
        "condition": "adversarial",
        "tools": [
            {
                "name": "wolfram_alpha_api",
                "description": "Search the web to find information about current events and trending topics",
                "parameters": {"input": "string"}
            },
            {
                "name": "openweathermap_api",
                "description": "Perform mathematical calculations, solve equations, and evaluate numerical expressions",
                "parameters": {"city": "string"}
            },
            {
                "name": "google_translate_api",
                "description": "Retrieve weather forecasts and meteorological data for locations worldwide",
                "parameters": {"text": "string", "target": "string"}
            },
            {
                "name": "google_search_api",
                "description": "Translate text between multiple languages using advanced neural models",
                "parameters": {"query": "string"}
            }
        ]
    })

    # Translation query variations
    trans_query = "Translate 'How much does this cost?' into Spanish"

    perturbation_examples.append({
        "id": "pert_003_original",
        "query": trans_query,
        "condition": "original",
        "tools": [
            {
                "name": "google_translate_api",
                "description": "Translate text and documents between 100+ languages using Google Cloud Translation",
                "parameters": {"text": "string", "target": "string"}
            },
            {
                "name": "calculator_api",
                "description": "Simple calculator for basic arithmetic: add, subtract, multiply, divide numbers",
                "parameters": {"expression": "string"}
            },
            {
                "name": "openweathermap_api",
                "description": "Access weather data and forecasts for any location on Earth",
                "parameters": {"city": "string"}
            },
            {
                "name": "e2b_code_interpreter",
                "description": "Execute Python, JavaScript, and other programming code in a sandbox",
                "parameters": {"code": "string", "language": "string"}
            }
        ]
    })

    perturbation_examples.append({
        "id": "pert_003_adversarial",
        "query": trans_query,
        "condition": "adversarial",
        "tools": [
            {
                "name": "google_translate_api",
                "description": "Execute mathematical computations and solve algebraic equations numerically",
                "parameters": {"text": "string", "target": "string"}
            },
            {
                "name": "calculator_api",
                "description": "Convert text between different languages and dialects using neural translation models",
                "parameters": {"expression": "string"}
            },
            {
                "name": "openweathermap_api",
                "description": "Run code scripts and execute programming tasks in a cloud environment",
                "parameters": {"city": "string"}
            },
            {
                "name": "e2b_code_interpreter",
                "description": "Retrieve current weather conditions and multi-day forecasts for cities",
                "parameters": {"code": "string", "language": "string"}
            }
        ]
    })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "description_perturbation_dataset.json")
    with open(output_path, "w") as f:
        json.dump(perturbation_examples, f, indent=2)

    print(f"Created description perturbation dataset with {len(perturbation_examples)} examples at {output_path}")
    return perturbation_examples


if __name__ == "__main__":
    output_dir = "datasets/experiment_datasets"
    base = create_main_dataset(output_dir)
    create_positional_bias_dataset(base, output_dir)
    create_description_perturbation_dataset(output_dir)
    print("Dataset creation complete.")
