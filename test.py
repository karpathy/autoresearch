import os
import random
from memory import connect, save_experiment, record_verdict, get_experiment, get_all_experiments, should_run_experiment
from schema import ExperimentRecord, Verdict

def run_tests():
    print("Connecting to in-memory DB...")
    conn = connect(":memory:")

    # 0. Prepopulate DB to ensure standard scaling has a realistic mean/variance (not just 1 or 2 opposite points)
    print("Seeding database with 5 background experiments to stabilize stats...")
    for i in range(5):
        save_experiment(conn, ExperimentRecord(
            hyperparameters={"lr": random.uniform(0.005, 0.01), "batch_size": 64},
            confidence=0.5,
            last_verdict=Verdict.REJECT
        ))

    # 1. Base Experiment
    exp1 = ExperimentRecord(
        hyperparameters={"lr": 0.001, "batch_size": 32},
        confidence=0.5
    )
    print(f"\n[1] Saving Experiment 1 (ID: {exp1.id})")
    save_experiment(conn, exp1)
    
    print("    Recording verdict ACCEPT with val_bpb=1.2")
    record_verdict(conn, exp1.id, Verdict.ACCEPT, val_bpb=1.2)
    
    updated_exp1 = get_experiment(conn, exp1.id)
    print(f"    -> Updated confidence: {updated_exp1.confidence}")

    # 2. Similar Experiment, Same Verdict (Should boost confidence)
    exp2 = ExperimentRecord(
        hyperparameters={"lr": 0.00101, "batch_size": 32},
        confidence=0.5
    )
    print(f"\n[2] Saving Experiment 2 (ID: {exp2.id})")
    save_experiment(conn, exp2)
    
    print("    Recording verdict ACCEPT (Same verdict) with val_bpb=1.19")
    record_verdict(conn, exp2.id, Verdict.ACCEPT, val_bpb=1.19)
    
    updated_exp2 = get_experiment(conn, exp2.id)
    print(f"    -> Updated confidence (should increase): {updated_exp2.confidence}")

    # 3. Similar Experiment, Opposite Verdict (Should trigger LLM)
    exp3 = ExperimentRecord(
        hyperparameters={"lr": 0.00102, "batch_size": 32},
        confidence=0.5
    )
    print(f"\n[3] Saving Experiment 3 (ID: {exp3.id})")
    save_experiment(conn, exp3)
    
    print("    Recording verdict REJECT (Opposite verdict) with val_bpb=1.5")
    print("    This should trigger the LLM...")
    record_verdict(conn, exp3.id, Verdict.REJECT, val_bpb=1.5)
    
    updated_exp3 = get_experiment(conn, exp3.id)
    print(f"    -> Updated confidence (reset to 1.0): {updated_exp3.confidence}")
    print(f"    -> LLM was used: {updated_exp3.llm_used}")
    print(f"    -> Final Verdict: {updated_exp3.last_verdict.value if updated_exp3.last_verdict else 'None'}")

    # --- Test Gate Logic (should_run_experiment) ---
    print("\n--- Testing should_run_experiment Gate ---")
    
    # Test Case A: Highly similar to a REJECT with high confidence
    # For this test, updated_exp3 is a REJECT with confidence 1.0.
    query_a = {"lr": 0.001021, "batch_size": 32}
    should_run_a, reason_a = should_run_experiment(conn, query_a)
    print(f"\n[Gate A] Query (Sim to exp3 REJECT/conf 1.0): {query_a}")
    print(f"    -> Should Run? {should_run_a} | Reason: {reason_a}")

    # Test Case B: Highly similar to an ACCEPT (any confidence)
    # exp1 and exp2 are both ACCEPT.
    query_b = {"lr": 0.001005, "batch_size": 32}
    should_run_b, reason_b = should_run_experiment(conn, query_b)
    print(f"\n[Gate B] Query (Sim to exp2 ACCEPT/conf 1.0): {query_b}")
    print(f"    -> Should Run? {should_run_b} | Reason: {reason_b}")

    # Test Case C: Highly similar to a REJECT with Low confidence
    exp4 = ExperimentRecord(
        hyperparameters={"lr": 0.00502, "batch_size": 32},
        confidence=0.3,
        last_verdict=Verdict.REJECT
    )
    save_experiment(conn, exp4)
    
    query_c = {"lr": 0.005021, "batch_size": 32}
    should_run_c, reason_c = should_run_experiment(conn, query_c)
    print(f"\n[Gate C] Query (Sim to low-conf REJECT): {query_c}")
    print(f"    -> Should Run? {should_run_c} | Reason: {reason_c}")

if __name__ == '__main__':
    run_tests()
