import argparse
import os

from mot.data import generate_scenario, load_pets09_detection_file, load_result, load_scenario, save_result, save_scenario, scenario_config
from mot.gnn import run_gnn
from mot.jpda import run_jpda


def main():
    parser = argparse.ArgumentParser(description="Simple GNN/JPDAF multi-object tracking demo")
    parser.add_argument("command", choices=["generate", "run", "compare", "evaluate", "plot", "animate"])
    parser.add_argument("--tracker", choices=["gnn", "jpda"], default="gnn")
    parser.add_argument("--data", default="outputs/scenario.npz")
    parser.add_argument("--result", default=None)
    parser.add_argument("--gnn", default="outputs/gnn_result.npz")
    parser.add_argument("--jpda", default="outputs/jpda_result.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frame", type=int, default=60)
    parser.add_argument("--scenario", choices=["crossing", "four", "jpda_advantage", "pets09"], default="crossing")
    parser.add_argument("--detection-file", default="data/pets09_s2l1_detection.txt")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-distance", type=float, default=60.0)
    args = parser.parse_args()

    if args.command == "generate":
        if args.scenario == "pets09":
            scenario = load_pets09_detection_file(args.detection_file, max_frames=args.steps)
        else:
            config = scenario_config(args.scenario)
            if args.steps is not None:
                config["steps"] = args.steps
            if args.seed is not None:
                config["seed"] = args.seed
            scenario = generate_scenario(config)
        save_scenario(args.data, scenario)
        print(f"saved scenario to {args.data}")

    elif args.command == "run":
        scenario = load_scenario(args.data)
        result = run_gnn(scenario["measurements"], scenario["config"]) if args.tracker == "gnn" else run_jpda(scenario["measurements"], scenario["config"])
        out = args.result or f"outputs/{args.tracker}_result.npz"
        save_result(out, result)
        print(f"saved {args.tracker} result to {out}")

    elif args.command == "compare":
        from mot.visualization import plot_track_count, plot_tracks

        scenario = load_scenario(args.data)
        if os.path.exists(args.gnn):
            gnn = load_result(args.gnn)
        else:
            gnn = run_gnn(scenario["measurements"], scenario["config"])
            save_result(args.gnn, gnn)

        if os.path.exists(args.jpda):
            jpda = load_result(args.jpda)
        else:
            jpda = run_jpda(scenario["measurements"], scenario["config"])
            save_result(args.jpda, jpda)

        out = args.out or "outputs/comparison.png"
        plot_tracks(scenario, [gnn, jpda], out)
        plot_track_count(scenario, [gnn, jpda], "outputs/track_count.png")
        print(f"saved comparison to {out}")

    elif args.command == "evaluate":
        from mot.evaluation import evaluate_detection_results, evaluate_results, print_metrics, save_metrics

        scenario = load_scenario(args.data)
        gnn = load_result(args.gnn)
        jpda = load_result(args.jpda)
        has_truth = sum(len(frame) for frame in scenario["truth"]) > 0
        if has_truth:
            metrics = evaluate_results(scenario, [gnn, jpda], args.max_distance)
        else:
            metrics = evaluate_detection_results(scenario, [gnn, jpda], args.max_distance)
        print_metrics(metrics)
        if args.out:
            save_metrics(args.out, metrics)
            print(f"saved metrics to {args.out}")

    elif args.command == "plot":
        from mot.visualization import plot_frame, plot_image_frame, plot_scenario

        scenario = load_scenario(args.data)
        if args.gnn and args.jpda and os.path.exists(args.gnn) and os.path.exists(args.jpda):
            results = [load_result(args.gnn), load_result(args.jpda)]
            out = args.out or f"outputs/image_frame_{args.frame}.png"
            plot_image_frame(scenario, results, args.frame, image_dir=args.image_dir, path=out)
        elif args.result:
            result = load_result(args.result)
            out = args.out or f"outputs/{result['tracker']}_frame_{args.frame}.png"
            plot_frame(scenario, result, args.frame, out)
        else:
            out = args.out or "outputs/scenario.png"
            plot_scenario(scenario, out)
        print(f"saved plot to {out}")

    elif args.command == "animate":
        from mot.visualization import animate, animate_image_frames, render_image_frames

        scenario = load_scenario(args.data)
        out = args.out or f"outputs/{args.tracker}_animation.mp4"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if os.path.exists(args.gnn) and os.path.exists(args.jpda):
            results = [load_result(args.gnn), load_result(args.jpda)]
            image_dir = args.image_dir or scenario["config"].get("image_dir")
            if image_dir:
                ext = os.path.splitext(out)[1].lower()
                if ext in ("", ".png"):
                    render_image_frames(scenario, results, image_dir, out, start=args.start, end=args.end)
                else:
                    animate_image_frames(scenario, results, image_dir, out, start=args.start, end=args.end)
            else:
                animate(scenario, results[0], out)
        else:
            result_path = args.result or f"outputs/{args.tracker}_result.npz"
            result = load_result(result_path)
            animate(scenario, result, out)
        print(f"saved animation to {out} or frame fallback directory")


if __name__ == "__main__":
    main()
