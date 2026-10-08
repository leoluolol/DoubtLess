"""Generate a reasoning trace and answer with a Qwen3 model pair."""
from doubtless import ContrastEngine, Sampling


def main():
    with ContrastEngine(
        "Qwen/Qwen3-32B",
        small="Qwen/Qwen3-0.6B",
        alpha=5.0,
        max_model_len=4096,
        max_num_seqs=1,
    ) as engine:
        results = engine.generate(
            ["What is the sum of the positive divisors of 360?"],
            Sampling(max_tokens=2048, seed=0),
        )
    for result in results:
        print(result.text)
        print(f"Thinking tokens: {result.think_tokens}; total tokens: {result.completion_tokens}")


if __name__ == "__main__":
    main()
