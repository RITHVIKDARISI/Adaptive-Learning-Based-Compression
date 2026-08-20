import random
import os
import pandas as pd
import numpy as np
from src.utils import get_data_dir

def generate_poisson_timestamps(n_messages: int, mean_interval_sec: float = 1.0) -> list[float]:
    timestamps = []
    current_time = 0.0
    for _ in range(n_messages):
        interval = np.random.exponential(scale=mean_interval_sec)
        current_time += interval
        timestamps.append(current_time)
    return timestamps

def _load_or_generate(filename: str, fallback_fn, n_samples: int) -> list[str]:
    path = os.path.join(get_data_dir(), filename)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if "text" in df.columns:
                texts = df["text"].dropna().tolist()
                if len(texts) >= n_samples:
                    n_unique = max(int(n_samples * 0.88), 5)
                    sampled = random.sample(texts, n_unique)
                    n_dups = n_samples - len(sampled)
                    duplicates = [random.choice(sampled) for _ in range(n_dups)]
                    combined = sampled + duplicates
                    random.shuffle(combined)
                    return combined[:n_samples]
        except Exception:
            pass
    return fallback_fn(n_samples)

def generate_synthetic_conversational(n_samples: int) -> list[str]:
    def _fallback(n):
        short = [
            "hi", "hello", "hey", "ok", "sure", "bye", "thanks", "got it",
            "yes", "no", "haha", "lol", "okay", "yep", "on my way",
        ]
        medium = [
            "Hey, are you free for a quick call this afternoon?",
            "Did you manage to finish the report before the deadline?",
            "Let me know when you are available to discuss the project details.",
            "I'll be a bit late today, can we push the meeting by 15 minutes?",
            "Can you send me the updated slides before the presentation starts?",
            "Just checking in — how is the implementation going so far today?",
            "Please review the pull request when you get a chance, it's urgent.",
            "Are we still on for the standup at 10am tomorrow morning?",
            "The build failed again on the CI pipeline, can you take a look?",
            "I finished the feature — testing it now before pushing to main branch.",
            "Could you double-check the API response format before we deploy?",
            "The database migration script ran successfully on the staging environment.",
        ]
        long = [
            "System alert: the database connection pool has reached its maximum limit of 100 concurrent connections. Please investigate and scale the service immediately to avoid request timeouts affecting end users.",
            "Could you please provide a detailed summary of the discussion points from today's architecture review meeting? I missed the first 20 minutes due to a conflicting call with the client team.",
            "The automated deployment pipeline failed at the Docker build stage due to a missing environment variable. The variable POSTGRES_URL needs to be set in the production secrets manager before retrying.",
            "I've completed the initial implementation of the adaptive compression scheduler. The decision tree model achieves 87% accuracy on the validation set with a mean inference latency of just 0.4 microseconds.",
            "Please make sure to run the full test suite before merging any changes to the main branch. The regression tests in particular are critical for catching any compression ratio performance regressions.",
            "The load balancer is distributing traffic unevenly across the three backend nodes. Node 2 is receiving approximately 60% of all requests while nodes 1 and 3 remain mostly idle during peak hours.",
            "We need to revisit the caching strategy for the message deduplication layer. The current LRU cache is evicting entries too aggressively under high-throughput workloads above 10,000 messages per second.",
            "The quarterly performance benchmarks show that the LinUCB bandit model outperforms all static baselines by an average of 23% on compression ratio while keeping end-to-end latency well below 5 microseconds.",
            "Error trace: NullPointerException in thread main at com.example.CompressionService.compress(CompressionService.java:142). The root cause appears to be an uninitialized codec instance during cold startup.",
            "Reminder: the sprint planning session is scheduled for tomorrow at 9am. Please review the backlog items and come prepared with effort estimates for the compression pipeline tickets assigned to you.",
        ]
        msgs = []
        for _ in range(n):
            r = random.random()
            if r < 0.20:
                msgs.append(random.choice(short))
            elif r < 0.55:
                msgs.append(random.choice(medium))
            else:
                msgs.append(random.choice(long))
        return msgs

    return _load_or_generate("dailydialog_clean.csv", _fallback, n_samples)

def generate_synthetic_tweets(n_samples: int) -> list[str]:
    def _fallback(n):
        short = [
            "same 😂", "lol", "facts", "mood", "omg", "fr fr", "no cap", "💯",
        ]
        medium = [
            "loving the weather today! finally some sunshine after weeks of rain ☀️ #blessed",
            "just shipped my first open source project and it already has 12 stars — insane feeling 🚀",
            "working on a new ML project involving real-time text compression — results are looking promising",
            "so tired but the deadline is tomorrow, coffee is the only thing keeping me going right now ☕",
            "congrats to the whole team on a successful product launch! three months of hard work paid off 🎉",
            "traffic on the highway is absolutely terrible today — 45 minutes for a 10 minute drive 😤",
            "cannot believe how fast the year is going — it's already August and I haven't read half my book list",
            "learning Rust for the first time and honestly the borrow checker is humbling me every single day",
            "just hit 1000 GitHub followers — never expected this when I started posting code snippets last year!",
            "the new transformer architecture paper dropped today and it's already breaking benchmarks on five tasks",
        ]
        long = [
            "Hot take: most ML papers overclaim results because they test on cherry-picked benchmarks that don't reflect real production workloads. We need standardized streaming evaluation protocols. #MachineLearning #MLOps",
            "After 6 months of working remotely full time I've realized that async communication is actually more productive than back-to-back meetings. The key is clear written documentation and fast response SLAs for teams.",
            "Just published a detailed benchmark comparing ZSTD, Brotli, GZIP and LZ4 on real-world chat message streams. ZSTD wins on compression ratio, LZ4 wins on latency, Brotli is the sweet spot for compressible text.",
            "The job market for ML engineers right now is tough — every posting requires 5+ years of experience with technologies that have only existed for 2 years. Companies need to get realistic about their hiring expectations.",
            "Thread on why I think contextual bandits are underrated for real-time infrastructure decisions: they adapt online, require no offline labels, and handle distribution shift naturally without retraining. 1/7 🧵",
        ]
        msgs = []
        for _ in range(n):
            r = random.random()
            if r < 0.15:
                msgs.append(random.choice(short))
            elif r < 0.50:
                msgs.append(random.choice(medium))
            else:
                msgs.append(random.choice(long))
        return msgs

    return _load_or_generate("sentiment140_clean.csv", _fallback, n_samples)

def generate_synthetic_sms(n_samples: int) -> list[str]:
    def _fallback(n):
        short = [
            "Ok", "Sure", "Yes", "No", "K", "👍", "On my way", "Be right there",
        ]
        medium = [
            "Hey, where are you? We've been waiting for about 10 minutes already.",
            "Can you call me back when you get a chance? It's about the weekend plans.",
            "I'll be late by about 15 minutes — stuck in traffic near the main highway.",
            "Please bring some milk and bread on your way home from the office tonight.",
            "Happy birthday! Hope you have an amazing day filled with great surprises 🎂",
            "Are we still meeting at the coffee shop at 3pm or did the plans change?",
            "Just checking — did you manage to book the tickets for Saturday's concert?",
            "The package was delivered to the front door, please bring it inside safely.",
            "Don't forget that dad's birthday dinner is this Friday at 7pm at the usual place.",
            "Your prescription is ready for pickup at the pharmacy whenever you're available.",
        ]
        long = [
            "Hi, just a reminder that your dentist appointment is scheduled for tomorrow at 11:30am at the downtown clinic on Main Street. Please arrive 10 minutes early to fill out the updated insurance forms at reception.",
            "Hey! Long time no talk — I wanted to reach out because I'll be in your city next week for a tech conference. Would love to grab dinner and properly catch up if you're free on Thursday or Friday evening!",
            "Your order #48291 has been dispatched from our warehouse and is expected to arrive between 2pm and 6pm tomorrow. You can track the delivery in real time using the tracking link in your confirmation email.",
            "Reminder: your car service appointment is booked for Friday at 9am at the dealership on Park Road. The service includes a full oil change, tire rotation, and brake inspection. Estimated duration is 2 to 3 hours.",
            "School notice: Tomorrow's field trip to the science museum has been rescheduled to next Wednesday due to weather forecasts. New permission slips will be sent home today. Please sign and return by Monday morning.",
        ]
        msgs = []
        for _ in range(n):
            r = random.random()
            if r < 0.20:
                msgs.append(random.choice(short))
            elif r < 0.55:
                msgs.append(random.choice(medium))
            else:
                msgs.append(random.choice(long))
        return msgs

    return _load_or_generate("nussms_clean.csv", _fallback, n_samples)

DATASET_GENERATORS = {
    "DailyDialog-style (chat)": (generate_synthetic_conversational, 0.8),
    "Sentiment140-style (tweets)": (generate_synthetic_tweets, 0.5),
    "NUS-SMS-style (sms)": (generate_synthetic_sms, 1.2),
}
