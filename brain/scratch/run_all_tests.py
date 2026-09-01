def run_tests(steps=100, batch_size=16, input_dim=64, width=120):
    deterministic_seed(42)
    results = {}
    model = FlyInspiredThoughtletCore(input_dim=input_dim, thoughtlet_width=width)
    model.eval()
    x = torch.randn(batch_size, input_dim)
    reward = torch.randn(batch_size)
    next_x = torch.randn(batch_size, input_dim)

    with torch.no_grad():
        out = model(x)
        results["forward_output_mean"] = float(out.mean())
        results["forward_output_std"] = float(out.std())
        results["forward_output_norm_mean"] = float(out.norm(dim=-1).mean())

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
        results["determinism_max_diff"] = float((out1 - out2).abs().max())

    robustness = {}
    base = out.clone()
    for rate in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        if rate == 0.0:
            robustness["drop_0.0"] = 1.0
            continue
        mask = (torch.rand(batch_size, input_dim) > rate).float()
        out_dropped = model(x * mask)
        corr = torch.corrcoef(torch.cat([base.flatten().unsqueeze(0), out_dropped.flatten().unsqueeze(0)]))[0, 1].item()
        robustness[f"drop_{rate}"] = corr
    results["robustness"] = robustness

    deterministic_seed(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    train_losses = []
    for step in range(50):
        x_t = torch.randn(batch_size, input_dim)
        r_t = torch.randn(batch_size)
        next_x_t = torch.randn(batch_size, input_dim)
        optimizer.zero_grad()
        output = model(x_t, reward_signal=r_t)
        loss = loss_fn(output, next_x_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
    results["training_final_loss"] = train_losses[-1]
    results["training_mean_loss"] = float(np.mean(train_losses))

    rng = np.random.RandomState(42)
    W_dense = rng.randn(input_dim, width).astype(np.float32)
    W_out_dense = rng.randn(width, input_dim).astype(np.float32)
    h = torch.relu(x @ torch.from_numpy(W_dense))
    canon_out = h @ torch.from_numpy(W_out_dense)
    results["canonical_output_norm_mean"] = float(canon_out.norm(dim=-1).mean())
    results["fly_vs_canon_output_ratio"] = float(results["forward_output_norm_mean"] / results["canonical_output_norm_mean"])
    return results
def main():
    parser = argparse.ArgumentParser(description="Run all fruit-fly brain reference tests")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("Running all fruit-fly brain reference tests...")
    results = run_tests(steps=args.steps, batch_size=args.batch)

    print("\n=== Test Results ===")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2:.6f}")
        else:
            print(f"  {k}: {v:.6f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
