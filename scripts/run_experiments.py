import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mot.data import generate_scenario, load_pets09_detection_file, save_result, save_scenario, scenario_config
from mot.evaluation import evaluate_detection_results, evaluate_results, save_metrics
from mot.gnn import run_gnn
from mot.jpda import run_jpda
from mot.visualization import (
    animate,
    animate_image_frames,
    plot_image_frame,
    plot_scenario,
    plot_track_count,
    plot_tracks,
    render_image_frames,
)


def main():
    os.makedirs("outputs/report", exist_ok=True)
    run_synthetic("crossing", scenario_config("crossing"))
    run_synthetic("four", scenario_config("four"))
    run_synthetic("jpda_advantage", scenario_config("jpda_advantage"))
    if os.path.exists("data/pets09_s2l1_detection.txt") and os.path.isdir("data/pets09_s2l1/img1"):
        run_pets09()
    else:
        print("Skipping PETS09 regeneration; copy data from apennisi/jpdaf_tracking into data/ first.")


def run_synthetic(name, config):
    scenario = generate_scenario(config)
    gnn = run_gnn(scenario["measurements"], scenario["config"])
    jpda = run_jpda(scenario["measurements"], scenario["config"])
    metrics = evaluate_results(scenario, [gnn, jpda])

    save_scenario(f"outputs/{name}_scenario.npz", scenario)
    save_result(f"outputs/{name}_gnn.npz", gnn)
    save_result(f"outputs/{name}_jpda.npz", jpda)
    save_metrics(f"outputs/report/{name}_metrics.json", metrics)
    plot_scenario(scenario, f"outputs/report/{name}_measurements.png")
    plot_tracks(scenario, [gnn, jpda], f"outputs/report/{name}_comparison.png")
    plot_track_count(scenario, [gnn, jpda], f"outputs/report/{name}_track_count.png")

    animate(scenario, gnn, f"outputs/report/{name}_gnn.gif")
    animate(scenario, jpda, f"outputs/report/{name}_jpda.gif")


def run_pets09():
    scenario = load_pets09_detection_file()
    gnn = run_gnn(scenario["measurements"], scenario["config"])
    jpda = run_jpda(scenario["measurements"], scenario["config"])
    metrics = evaluate_detection_results(scenario, [gnn, jpda], max_distance=45)

    save_scenario("outputs/pets09_scenario.npz", scenario)
    save_result("outputs/pets09_gnn.npz", gnn)
    save_result("outputs/pets09_jpda.npz", jpda)
    save_metrics("outputs/report/pets09_metrics.json", metrics)
    plot_tracks(scenario, [gnn, jpda], "outputs/report/pets09_comparison.png")
    plot_track_count(scenario, [gnn, jpda], "outputs/report/pets09_track_count.png")
    plot_image_frame(
        scenario,
        [gnn, jpda],
        120,
        image_dir=scenario["config"]["image_dir"],
        path="outputs/report/pets09_frame_0121.png",
    )

    frame_dir = "outputs/report/pets09_frames_100_130"
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    render_image_frames(
        scenario,
        [gnn, jpda],
        scenario["config"]["image_dir"],
        frame_dir,
        start=100,
        end=130,
    )
    animate_image_frames(
        scenario,
        [gnn, jpda],
        scenario["config"]["image_dir"],
        "outputs/report/pets09_full.mp4",
    )


if __name__ == "__main__":
    main()
