import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANUALS_DIR = DATA_DIR / "manuals"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Web search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MOTO_DOMAINS = [
    "advrider.com",
    "reddit.com",
    "thumpertalk.com",
    "ktmforums.com",
    "husqvarnaownersgroup.com",
]

# Search
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.5"))

# Safety-critical components that trigger disclaimers
SAFETY_CRITICAL_KEYWORDS = [
    "brake", "caliper", "rotor", "master cylinder",
    "axle", "wheel", "steering head", "triple clamp",
    "suspension", "fork", "shock", "linkage",
    "chain", "sprocket", "swingarm pivot",
    "frame", "subframe", "engine mount",
]
