import pickle
import numpy as np
import argparse
import os

def check_fellner_convergence(val_history, window_size=20):
    """
    Applies the Fellner/Thesis criteria to a validation history.
    Formula: |mu1 - mu2| < sigma2 / (2 * sqrt(N))
    """
    if len(val_history) < 2 * window_size:
        return None, "History too short for specified window."

    for i in range(2 * window_size, len(val_history)):
        # Extract the two sliding windows ending at the current epoch i
        recent = val_history[i - (2 * window_size) : i]
        w1 = recent[:window_size]
        w2 = recent[window_size:]
        
        mu1, mu2 = np.mean(w1), np.mean(w2)
        sigma2 = np.std(w2)
        
        # Fellner Criteria Threshold
        threshold = sigma2 / (2 * np.sqrt(window_size))
        diff = abs(mu1 - mu2)
        
        if diff < threshold:
            return i, diff, threshold
            
    return None, "Convergence criteria never met."

def main():
    parser = argparse.ArgumentParser(description="Post-hoc Convergence Checker")
    parser.add_argument('--path', type=str, required=True, help="Path to the .pkl model file")
    parser.add_argument('--window', type=int, default=20, help="Convergence window size (N)")
    args = parser.parse_args()
    if not os.path.exists(args.path):
        print(f"Error: File {args.path} not found.")
        return

    with open(args.path, 'rb') as f:
        data = pickle.load(f)

    val_history = data.get('val_history', [])
    
    print(f"\n--- Analyzing: {os.path.basename(args.path)} ---")
    print(f"Total Epochs in File: {len(val_history)}")
    
    epoch, diff, thresh = check_fellner_convergence(val_history, args.window)
    
    if epoch:
        print(f"STATUS: CONVERGED")
        print(f"Convergence Epoch: {epoch}")
        print(f"Final Delta: {diff:.8f}")
        print(f"Threshold:   {thresh:.8f}")
        print(f"Loss at Conv: {val_history[epoch]:.8f}")
    else:
        print(f"STATUS: NOT CONVERGED")
        print(f"Reason: {diff}") # In this case, 'diff' contains the string reason

if __name__ == "__main__":
    main()